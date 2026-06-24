#!/usr/bin/env python3
"""Monitor the HYMotion-G1 tracker-reward verification runs.

The goal is deliberately narrow: for each tracker backend, prove that optimizing
the generator from the same frozen base checkpoint improves fixed frozen-tracker
metrics.  This watcher discovers new checkpoints, submits the matching eval job,
and writes a compact table that separates training telemetry from real eval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO = Path(
    os.environ.get(
        "PROJECT_ROOT",
        "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer",
    )
).absolute()
CKPT_RE = re.compile(r"checkpoint-iter_(\d+)$")
STEP_RE = re.compile(r"step \[(\d+)/(\d+)\]")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([+-]?(?:\d+(?:\.\d*)?|\.\d+))")

ARMS = {
    "proto": "work_dirs/physflow_verify_hymotion_g1_protomotions_130k_safe",
    "any2track": "work_dirs/physflow_verify_hymotion_g1_any2track_130k_safe",
    "humanoidgpt": "work_dirs/physflow_verify_hymotion_g1_humanoidgpt_130k_safe",
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _ckpt_iter(path: Path) -> Optional[int]:
    m = CKPT_RE.search(path.name)
    return int(m.group(1)) if m else None


def _metrics(summary_path: Path) -> Optional[Dict[str, Any]]:
    if not summary_path.is_file():
        return None
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return None
    return data.get("generated") or {}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def discover_checkpoints(root: Path) -> List[tuple[int, Path]]:
    out: List[tuple[int, Path]] = []
    if not root.is_dir():
        return out
    for ckpt in root.glob("checkpoint-iter_*"):
        it = _ckpt_iter(ckpt)
        if it is not None and (ckpt / "model.pt").is_file():
            out.append((it, ckpt))
    return sorted(out)


def latest_train_log(root: Path) -> Optional[Path]:
    logs = sorted(root.glob("20*/train.log"))
    return logs[-1] if logs else None


def parse_training_tail(log_path: Optional[Path], window: int = 100) -> Dict[str, Any]:
    if log_path is None or not log_path.is_file():
        return {"status": "no_log"}
    rows: List[Dict[str, float]] = []
    for line in log_path.open(errors="ignore"):
        if "step [" not in line or "n_good=" not in line:
            continue
        sm = STEP_RE.search(line)
        if not sm:
            continue
        rec: Dict[str, float] = {"step": float(sm.group(1)), "total": float(sm.group(2))}
        for key, value in KV_RE.findall(line):
            try:
                rec[key] = float(value)
            except ValueError:
                pass
        rows.append(rec)
    if not rows:
        return {"status": "waiting_metrics", "log": str(log_path)}
    tail = rows[-window:]

    def mean(key: str) -> Optional[float]:
        vals = [r[key] for r in tail if key in r]
        return sum(vals) / len(vals) if vals else None

    return {
        "status": "ok",
        "log": str(log_path),
        "last_step": int(rows[-1]["step"]),
        "tail_n": len(tail),
        "n_good_mean": mean("n_good"),
        "reward_best_mean": mean("reward_best_mean"),
        "loss_mean": mean("loss"),
        "joint_std_mean": mean("sel_joint_std_mean"),
    }


def launcher_record_path(state_dir: Path, tag: str) -> Path:
    return state_dir / f"{tag}.launcher.json"


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def already_submitted_or_done(out_dir: Path, state_dir: Path, tag: str) -> bool:
    if (out_dir / "summary.json").is_file():
        return True
    if (REPO / f"work_dirs/physflow_hymotion_g1_eval_{tag}_good_task.txt").is_file():
        return True
    rec_path = launcher_record_path(state_dir, tag)
    if rec_path.is_file():
        try:
            rec = json.loads(rec_path.read_text())
            pid = int(rec.get("pid", -1))
            if pid > 0 and process_alive(pid):
                return True
        except Exception:
            pass
    return False


def submit_eval(
    *,
    tag: str,
    ckpt: Path,
    out_dir: Path,
    state_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if already_submitted_or_done(out_dir, state_dir, tag):
        return {"tag": tag, "status": "already_done_or_submitted", "out": str(out_dir)}
    state_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TOKEN"] = args.token
    env.update(
        {
            "TAG": tag,
            "EVAL_CKPT": str(ckpt),
            "EVAL_OUT": str(out_dir),
            "EVAL_NUM_SAMPLES": str(args.num_samples),
            "EVAL_MAX_ITEMS": str(args.max_items),
            "EVAL_SAMPLE_STEPS": str(args.sample_steps),
            "EVAL_BATCH_SIZE": str(args.batch_size),
            "EVAL_SEED": str(args.seed),
            "MAXATT": str(args.max_attempts),
            "GATE_POLLS": str(args.gate_polls),
            "GATE_SLEEP_SEC": str(args.gate_sleep_sec),
            "EVAL_GPU": args.gpu,
            "EVAL_BUSINESS_FLAG": args.business_flag,
        }
    )
    log_path = state_dir / f"{tag}.launcher.log"
    log_f = log_path.open("a")
    proc = subprocess.Popen(
        ["bash", "scripts/embodied/submit_eval_hymotion_g1_until_good_node.sh"],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    log_f.close()
    rec = {
        "tag": tag,
        "status": "launcher_started",
        "pid": proc.pid,
        "ckpt": str(ckpt),
        "out": str(out_dir),
        "launcher_log": str(log_path),
        "ts": _now(),
    }
    launcher_record_path(state_dir, tag).write_text(json.dumps(rec, indent=2))
    return rec


def iter_eval_rows(out_root: Path) -> Iterable[tuple[str, str, Path]]:
    yield "base", "130000", out_root / "base130k_frozen_eval" / "summary.json"
    for method in ARMS:
        for p in sorted(out_root.glob(f"{method}_iter*_frozen_eval/summary.json")):
            name = p.parent.name
            m = re.search(r"_iter(\d+)_", name)
            yield method, m.group(1) if m else "?", p


def write_summary(out_root: Path, state: Dict[str, Any]) -> None:
    base = _metrics(out_root / "base130k_frozen_eval" / "summary.json") or {}
    lines = [
        "# HYMotion-G1 Tracker Reward 130k Monitor",
        "",
        f"Updated: {_now()}",
        "",
        "| Method | Iter | Done | Completion ↑ | Fall ↓ | Score ↓ | MaxJ ↓ | RootErr ↓ | Trackable ↑ | ΔScore vs base | ΔTrackable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for method, iteration, summary_path in iter_eval_rows(out_root):
        metrics = _metrics(summary_path)
        rows.append((method, iteration, summary_path, metrics))
    for method, iteration, summary_path, metrics in rows:
        done = metrics is not None
        m = metrics or {}
        d_score = None
        d_track = None
        if method != "base" and m and base:
            if m.get("adversarial_score_mean") is not None and base.get("adversarial_score_mean") is not None:
                d_score = float(m["adversarial_score_mean"]) - float(base["adversarial_score_mean"])
            if m.get("trackable_basic_rate") is not None and base.get("trackable_basic_rate") is not None:
                d_track = float(m["trackable_basic_rate"]) - float(base["trackable_basic_rate"])
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(iteration),
                    "Y" if done else "N",
                    _fmt(m.get("completion_mean")),
                    _fmt(m.get("fall_rate")),
                    _fmt(m.get("adversarial_score_mean")),
                    _fmt(m.get("max_joint_error_rad_mean")),
                    _fmt(m.get("root_trajectory_error_mean_m")),
                    _fmt(m.get("trackable_basic_rate")),
                    _fmt(d_score),
                    _fmt(d_track),
                ]
            )
            + " |"
        )
    lines += ["", "## Training Tail", ""]
    lines.append("| Method | Status | Last step | n_good | reward_best | loss | joint_std |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for method, info in state.get("training", {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(info.get("status")),
                    str(info.get("last_step", "-")),
                    _fmt(info.get("n_good_mean")),
                    _fmt(info.get("reward_best_mean")),
                    _fmt(info.get("loss_mean")),
                    _fmt(info.get("joint_std_mean")),
                ]
            )
            + " |"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "monitor_summary.md").write_text("\n".join(lines) + "\n")
    (out_root / "monitor_state.json").write_text(json.dumps(state, indent=2))


def run_once(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = REPO / args.out_root
    state_dir = REPO / args.state_dir
    submitted: List[Dict[str, Any]] = []
    base_ckpt = REPO / "work_dirs/physflow_verify_hymotion_g1_base/checkpoint-iter_130000"
    submitted.append(
        submit_eval(
            tag="verify_base130k_frozen",
            ckpt=base_ckpt,
            out_dir=out_root / "base130k_frozen_eval",
            state_dir=state_dir,
            args=args,
        )
    )
    training = {}
    for method, rel_root in ARMS.items():
        root = REPO / rel_root
        training[method] = parse_training_tail(latest_train_log(root), args.train_window)
        for it, ckpt in discover_checkpoints(root):
            if it < args.min_iter or it % args.eval_every != 0:
                continue
            submitted.append(
                submit_eval(
                    tag=f"verify_130k_{method}_it{it}_frozen",
                    ckpt=ckpt,
                    out_dir=out_root / f"{method}_iter{it}_frozen_eval",
                    state_dir=state_dir,
                    args=args,
                )
            )
    state = {"ts": _now(), "submitted": submitted, "training": training}
    write_summary(out_root, state)
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("TOKEN", ""))
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--sleep-sec", type=int, default=300)
    ap.add_argument("--out-root", default="output/physflow_verify_hymotion_g1_130k_safe")
    ap.add_argument("--state-dir", default="work_dirs/physflow_tracker_reward_130k_monitor")
    ap.add_argument("--num-samples", type=int, default=24)
    ap.add_argument("--max-items", type=int, default=4096)
    ap.add_argument("--sample-steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--gate-polls", type=int, default=120)
    ap.add_argument("--gate-sleep-sec", type=int, default=10)
    ap.add_argument("--gpu", default="V100")
    ap.add_argument("--business-flag", default="AILab_DHA")
    ap.add_argument("--min-iter", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--train-window", type=int, default=50)
    args = ap.parse_args()
    if not args.token:
        raise SystemExit("TOKEN is required")
    for i in range(args.loops):
        state = run_once(args)
        print(json.dumps(state, indent=2))
        if i + 1 < args.loops:
            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()
