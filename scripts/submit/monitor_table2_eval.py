#!/usr/bin/env python3
"""Monitor Table 2 Taiji inference/evaluation jobs.

This script is intentionally narrow: it only watches the HYMotionM2M Table 2
jobs submitted from this repository, records progress, and optionally switches
long-pending jobs to a fallback business flag.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO / "outputs" / "evaluation" / "humanml3d" / "_monitor"
STATE_PATH = LOG_ROOT / "table2_monitor_state.json"
TOKEN_FILE = Path("~/.claude-dashboard/taiji_token").expanduser()
ISO = "%Y-%m-%d %H:%M:%S"


@dataclass
class Job:
    key: str
    task: str
    pred_dir: Path
    eval_json: Path
    resubmit_cmd: list[str] | None = None
    fallback_task: str | None = None


JOBS = [
    Job(
        key="ours_smpl",
        task="m2m272_smpl_root_caption_neo_nocache",
        pred_dir=REPO / "outputs/evaluation/humanml3d/smpl_root_caption/pred272",
        eval_json=REPO / "outputs/evaluation/humanml3d/smpl_root_caption/eval_smpl_root_caption_cfg2p5_rep20.json",
    ),
    Job(
        key="ours_kimodo",
        task="m2m272_kimodo_root_caption_neo_nocache",
        pred_dir=REPO / "outputs/evaluation/humanml3d/kimodo_root_caption/pred272",
        eval_json=REPO / "outputs/evaluation/humanml3d/kimodo_root_caption/eval_kimodo_root_caption_cfg2p5_rep20.json",
        resubmit_cmd=[
            sys.executable, "scripts/submit/submit_m2m_272_eval.py",
            "--model", "kimodo_root_caption",
            "--business", "AILab_DHC_DD",
            "--flag-suffix", "_dd_nocache",
            "--no-cache",
        ],
        fallback_task="m2m272_kimodo_root_caption_dd_nocache",
    ),
    Job(
        key="flowmdm_eval",
        task="flowmdm_t2m272_full_neo_nocache_eval",
        pred_dir=REPO / "outputs/evaluation/humanml3d_hml3d263/flowmdm",
        eval_json=REPO / "outputs/evaluation/humanml3d/flowmdm/eval_flowmdm_full_rep20.json",
    ),
    Job(
        key="motionlab_full",
        task="motionlab_t2m272_full_neo_nocache_full",
        pred_dir=REPO / "outputs/evaluation/humanml3d_hml3d263/motionlab",
        eval_json=REPO / "outputs/evaluation/humanml3d/motionlab/eval_motionlab_full_rep20.json",
        resubmit_cmd=[
            sys.executable, "scripts/submit/submit_motionlab_t2m_eval.py",
            "--business", "AILab_DHC_DD",
            "--flag-suffix", "_dd_nocache",
            "--no-cache",
        ],
        fallback_task="motionlab_t2m272_full_dd_nocache",
    ),
]

LOG_FILES = [
    REPO / "outputs/evaluation/humanml3d/smpl_root_caption/_logs/run.log",
    REPO / "outputs/evaluation/humanml3d/kimodo_root_caption/_logs/run.log",
    REPO / "outputs/evaluation/humanml3d/flowmdm/_logs/run_full_eval.log",
    REPO / "outputs/evaluation/humanml3d/motionlab/_logs/run_full.log",
]


def get_token() -> str:
    token = os.environ.get("TOKEN", "")
    if token:
        return token
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    return ""


def run(cmd: list[str], timeout: int = 60) -> str:
    env = os.environ.copy()
    token = get_token()
    if not token:
        raise RuntimeError("TOKEN is not set")
    env["TOKEN"] = token
    proc = subprocess.run(
        cmd, cwd=REPO, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout, check=False,
    )
    return proc.stdout


def parse_il(text: str) -> dict[str, str | None]:
    for line in text.splitlines():
        if not line.startswith("| 20"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 6:
            return {
                "create_time": parts[0],
                "instance_id": parts[1],
                "state": parts[3],
                "time_delay": parts[4],
                "time_start": parts[5],
            }
    if "not found" in text.lower() or "no" in text.lower():
        return {"state": "MISSING", "create_time": None, "instance_id": None,
                "time_delay": None, "time_start": None}
    return {"state": "UNKNOWN", "create_time": None, "instance_id": None,
            "time_delay": None, "time_start": None}


def count_npy(path: Path) -> int:
    try:
        return sum(1 for name in os.listdir(path) if name.endswith(".npy"))
    except FileNotFoundError:
        return 0


def minutes_since(value: str | None) -> float | None:
    if not value or value in {"-", "None"}:
        return None
    try:
        return (datetime.now() - datetime.strptime(value, ISO)).total_seconds() / 60.0
    except ValueError:
        return None


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"switches": {}, "counts": {}}


def save_state(state: dict):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def tail_text(path: Path, limit: int = 2000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fp:
        fp.seek(0, os.SEEK_END)
        size = fp.tell()
        fp.seek(max(0, size - limit), os.SEEK_SET)
        return fp.read().decode("utf-8", errors="replace")


def maybe_switch(job: Job, info: dict, args, state: dict, lines: list[str]):
    if not args.auto_switch or not job.resubmit_cmd:
        return
    if info.get("state") != "PENDING":
        return
    age = minutes_since(info.get("create_time"))
    if age is None or age < args.pending_minutes:
        return
    if state["switches"].get(job.task):
        return

    lines.append(f"[switch] {job.task} pending {age:.1f} min; stopping and resubmitting")
    lines.append(run(["taiji_client", "stop", job.task], timeout=30).strip())
    lines.append(run(job.resubmit_cmd, timeout=120).strip())
    state["switches"][job.task] = {
        "time": datetime.now().strftime(ISO),
        "fallback_task": job.fallback_task,
    }
    job.task = job.fallback_task or job.task


def snapshot(args) -> str:
    state = load_state()
    lines = [f"===== {datetime.now().strftime(ISO)} ====="]
    for job in JOBS:
        if state["switches"].get(job.task, {}).get("fallback_task"):
            job.task = state["switches"][job.task]["fallback_task"]
        text = run(["taiji_client", "il", job.task], timeout=30)
        info = parse_il(text)
        maybe_switch(job, info, args, state, lines)
        n = count_npy(job.pred_dir)
        done = job.eval_json.exists()
        state["counts"][job.key] = {
            "task": job.task,
            "state": info.get("state"),
            "count": n,
            "eval_json": str(job.eval_json),
            "eval_done": done,
            "updated": datetime.now().strftime(ISO),
        }
        age = minutes_since(info.get("time_start") or info.get("create_time"))
        age_s = "-" if age is None else f"{age:.1f}m"
        lines.append(
            f"{job.key:16s} task={job.task:34s} state={info.get('state'):16s} "
            f"age={age_s:>7s} pred={n:5d} eval={'yes' if done else 'no'}"
        )
    lines.append("-- log tails --")
    for path in LOG_FILES:
        text = tail_text(path)
        if text:
            compact = re.sub(r"\n+", "\n", text.strip())[-800:]
            lines.append(f"[{path.relative_to(REPO)}]\n{compact}")
    save_state(state)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--auto-switch", action="store_true")
    parser.add_argument("--pending-minutes", type=float, default=45.0)
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        text = snapshot(args)
        print(text, flush=True)
        (LOG_ROOT / "table2_monitor.log").open("a").write(text)
        if not args.loop:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
