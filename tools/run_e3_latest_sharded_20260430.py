#!/usr/bin/env python3
"""Run active E3 settings with latest checkpoints using an 8-GPU shard scheduler."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "work_dirs" / "e3_latest_20260430_1747"
SETTINGS = ["every_10f", "every_15f", "every_30f", "every_60f", "adaptive"]
SHARDS = [(i, i + 30) for i in range(0, 240, 30)]


def _cmd(model: str, setting: str, start: int, end: int, out_dir: Path) -> list[str]:
    return [
        "python3",
        "tools/eval_m2m_v2_all_tasks.py",
        "--tasks",
        "E3",
        "--settings",
        setting,
        "--models",
        model,
        "--max-samples",
        str(end),
        "--start-index",
        str(start),
        "--end-index",
        str(end),
        "--num-steps",
        "50",
        "--replacement-guidance",
        "skip_last",
        "--output-dir",
        str(out_dir.relative_to(ROOT)),
        "--save-npz",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["uncond_local", "uncond_global"], required=True)
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()

    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, list[str]]] = []
    for setting in SETTINGS:
        for start, end in SHARDS:
            name = f"e3_{args.model}_{setting}_{start:03d}_{end:03d}"
            jobs.append((name, _cmd(args.model, setting, start, end, OUT_ROOT / name)))

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
            print(f"[launch] gpu={gpu} pid={proc.pid} {name}", flush=True)

        time.sleep(5)
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
    print(f"[all done] {len(jobs)} jobs for {args.model}", flush=True)


if __name__ == "__main__":
    main()
