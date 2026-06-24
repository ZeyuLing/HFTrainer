#!/usr/bin/env python3
"""Shard the 8082 production eval matrix for the 2026-05-01 refresh.

This script intentionally follows the lightweight scheduler pattern used by
the recent E3/E8/E14 reruns: one process per GPU, one task/setting/model shard
per process, and all outputs kept under a dated work_dir until merge/import.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "work_dirs" / "eval_8082_refresh_20260501"


@dataclass(frozen=True)
class Target:
    task: str
    setting: str
    n: int
    kind: str  # "hymotion", "kimodo", or "kimodo_t2m"
    models: tuple[str, ...] = ()
    use_caption: str | None = None  # KIMODO only: "yes" / "no"


# Dashboard-visible production settings, not hidden sweep/ablation settings.
HYMOTION_TARGETS: list[Target] = [
    Target("E1", "default", 240, "hymotion", ("caption_local_phase2", "caption_global_phase2")),
    *[
        # The 1-frame anchor settings are caption-required in m2m_eval_tasks.py;
        # uncond models intentionally skip them as too underspecified.
        Target("E2", s, 220, "hymotion", ("caption_local_phase2", "caption_global_phase2"))
        for s in ("start_1f", "end_1f", "both_1f")
    ],
    *[
        Target("E2", s, 220, "hymotion", ("caption_local_phase2", "caption_global_phase2", "uncond_local", "uncond_global"))
        for s in ("pre20", "post20", "mid60", "pre20_uncond", "post20_uncond", "mid60_uncond")
    ],
    *[
        Target("E3", s, 240, "hymotion", ("caption_local_phase2", "caption_global_phase2", "uncond_local", "uncond_global"))
        for s in ("every_5f", "every_10f", "every_15f", "every_30f", "every_60f", "adaptive")
    ],
    *[
        Target("E4", s, 100, "hymotion", ("caption_local_phase2", "caption_global_phase2", "uncond_local", "uncond_global"))
        for s in ("A_rhand_sparse", "B_ankles_sparse", "C_rhand_lfoot", "D_both_hands", "E_all4_sparse", "F_rhand_dense")
    ],
    *[
        Target("E5", s, 78, "hymotion", ("caption_local_phase2", "caption_global_phase2", "uncond_local", "uncond_global"))
        for s in ("A", "B", "C")
    ],
    Target("E6", "pos_contact", 50, "hymotion", ("caption_local_phase2", "caption_global_phase2", "uncond_local", "uncond_global")),
    Target("E7", "default", 50, "hymotion", ("caption_local_phase2", "caption_global_phase2")),
    Target("E8", "A", 200, "hymotion", ("caption_local_phase2", "caption_global_phase2")),
    Target("E8", "D", 200, "hymotion", ("uncond_local", "uncond_global")),
    # E9 is intentionally excluded here: the dashboard entries are repair-method
    # composites (strict mask + smoothing / MoGenDIT / QCSelect), not direct
    # eval_m2m_v2_all_tasks rows. Refresh it with the dedicated E9 repair stack.
    *[
        Target("E10", s, 50, "hymotion", ("caption_local_phase2", "caption_global_phase2", "uncond_local", "uncond_global"))
        for s in ("A_upper", "B_lower", "C_spine_only")
    ],
    *[
        Target("E13", s, 80, "hymotion", ("caption_local_phase2", "caption_global_phase2"))
        for s in ("A", "B", "C")
    ],
    *[
        Target("E14", s, 100, "hymotion", ("uncond_local", "uncond_global"))
        for s in ("L", "M")
    ],
    Target("E15", "default", 200, "hymotion", ("uncond_local", "uncond_global")),
]

KIMODO_TARGETS: list[Target] = [
    Target("E1", "default", 240, "kimodo_t2m", use_caption="yes"),
    *[
        Target("E2", s, 220, "kimodo", use_caption="yes")
        for s in ("start_1f", "end_1f", "both_1f", "pre20", "post20", "mid60")
    ],
    *[
        Target("E2", s, 220, "kimodo", use_caption="no")
        for s in ("pre20", "post20", "mid60", "pre20_uncond", "post20_uncond", "mid60_uncond")
    ],
    *[
        Target("E3", s, 240, "kimodo", use_caption=uc)
        for s in ("every_5f", "every_10f", "every_15f", "every_30f", "every_60f", "adaptive")
        for uc in ("yes", "no")
    ],
    *[
        Target("E4", s, 100, "kimodo", use_caption=uc)
        for s in ("A_rhand_sparse", "B_ankles_sparse", "C_rhand_lfoot", "D_both_hands", "E_all4_sparse", "F_rhand_dense")
        for uc in ("yes", "no")
    ],
    *[
        Target("E5", s, 78, "kimodo", use_caption=uc)
        for s in ("A", "B", "C")
        for uc in ("yes", "no")
    ],
    # E7 and E10 are not comparable in tools/run_kimodo_all_tasks.py and are
    # deliberately excluded rather than importing empty/stale dashboard rows.
    Target("E8", "A", 200, "kimodo", use_caption="yes"),
    Target("E8", "D", 200, "kimodo", use_caption="no"),
    *[
        Target("E14", s, 100, "kimodo", use_caption="no")
        for s in ("L", "M")
    ],
    Target("E15", "default", 200, "kimodo", use_caption="no"),
]


def _shards(n: int, shard_size: int) -> list[tuple[int, int]]:
    return [(s, min(s + shard_size, n)) for s in range(0, n, shard_size)]


def _hymotion_cmd(target: Target, model: str, start: int, end: int, out_dir: Path) -> list[str]:
    return [
        "python3",
        "tools/eval_m2m_v2_all_tasks.py",
        "--tasks",
        target.task,
        "--settings",
        target.setting,
        "--models",
        model,
        "--max-samples",
        str(target.n),
        "--start-index",
        str(start),
        "--end-index",
        str(end),
        "--num-steps",
        "50",
        "--replacement-guidance",
        "skip_last",
        "--text-guidance-scale",
        "2.0",
        "--output-dir",
        str(out_dir.relative_to(ROOT)),
        "--save-npz",
        "--use-rewritten",
    ]


def _kimodo_cmd(target: Target, start: int, end: int, out_dir: Path) -> list[str]:
    if target.kind == "kimodo_t2m":
        # The outer scheduler already maps this process to one physical GPU via
        # CUDA_VISIBLE_DEVICES. Keep KIMODO T2M single-GPU here to avoid fighting
        # the scheduler's GPU accounting.
        return [
            "python3",
            "tools/run_kimodo_t2m.py",
            "--num_gpus",
            "1",
            "--gpu_ids",
            "0",
            "--output_dir",
            str(out_dir.relative_to(ROOT)),
            "--steps",
            "100",
            "--data-file",
            "data/eval/m2m_v2/eval_e1_t2m_rewritten.json",
        ]
    cmd = [
        "python3",
        "tools/run_kimodo_all_tasks.py",
        "--tasks",
        target.task,
        "--settings",
        target.setting,
        "--max-samples",
        str(target.n),
        "--start-idx",
        str(start),
        "--end-idx",
        str(end),
        "--use-caption",
        str(target.use_caption),
        "--output-dir",
        str(out_dir.relative_to(ROOT)),
    ]
    if target.use_caption == "yes":
        cmd.append("--use-rewritten")
    return cmd


def build_jobs(group: str, shard_size: int, gpus: int) -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    if group in ("hymotion", "all"):
        for target in HYMOTION_TARGETS:
            for model in target.models:
                for start, end in _shards(target.n, shard_size):
                    name = f"hymotion_{model}_{target.task}_{target.setting}_{start:03d}_{end:03d}"
                    out_dir = OUT_ROOT / "hymotion" / name
                    jobs.append((name, _hymotion_cmd(target, model, start, end, out_dir)))
    if group in ("kimodo", "all"):
        for target in KIMODO_TARGETS:
            if target.kind == "kimodo_t2m":
                name = "kimodo_caption_E1_default_all"
                out_dir = OUT_ROOT / "kimodo" / name
                jobs.append((name, _kimodo_cmd(target, 0, gpus, out_dir)))
                continue
            for start, end in _shards(target.n, shard_size):
                model_tag = "caption" if target.use_caption == "yes" else "uncond"
                name = f"kimodo_{model_tag}_{target.task}_{target.setting}_{start:03d}_{end:03d}"
                out_dir = OUT_ROOT / "kimodo" / name
                jobs.append((name, _kimodo_cmd(target, start, end, out_dir)))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["hymotion", "kimodo", "all"], required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=25)
    args = parser.parse_args()

    jobs = build_jobs(args.group, args.shard_size, args.gpus)
    log_dir = OUT_ROOT / "logs" / args.group
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scheduler] group={args.group} jobs={len(jobs)} out={OUT_ROOT}", flush=True)

    running: dict[int, tuple[subprocess.Popen, str, object]] = {}
    idx = 0
    done = 0
    failed: list[str] = []
    while idx < len(jobs) or running:
        for gpu in range(args.gpus):
            if idx >= len(jobs) or gpu in running:
                continue
            name, cmd = jobs[idx]
            idx += 1
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = log_dir / f"{name}.log"
            log = log_path.open("wb")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (proc, name, log)
            print(f"[launch] {idx}/{len(jobs)} gpu={gpu} pid={proc.pid} {name}", flush=True)

        time.sleep(10)
        for gpu, (proc, name, log) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            done += 1
            print(f"[done] {done}/{len(jobs)} gpu={gpu} rc={rc} {name}", flush=True)
            if rc != 0:
                failed.append(name)
            del running[gpu]

    if failed:
        raise SystemExit(f"failed jobs: {failed}")
    print(f"[all done] group={args.group} jobs={len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
