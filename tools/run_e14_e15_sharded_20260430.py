#!/usr/bin/env python3
"""Launch sharded E14/E15 latest-checkpoint reruns on a debug machine.

This is intentionally tiny and operational: each process owns one GPU and one
case range, while eval_m2m_v2_all_tasks.py preserves the original sample_idx.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "work_dirs" / "e14_e15_rerun_latest_20260430"


def _launch(job: dict) -> int:
    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_ROOT / job["out"]
    log_path = log_dir / f'{job["name"]}.log'
    cmd = [
        "python3",
        "tools/eval_m2m_v2_all_tasks.py",
        "--tasks",
        job["task"],
        "--settings",
        job["setting"],
        "--models",
        job["model"],
        "--max-samples",
        str(job["end"]),
        "--start-index",
        str(job["start"]),
        "--end-index",
        str(job["end"]),
        "--num-steps",
        "50",
        "--replacement-guidance",
        "skip_last",
        "--output-dir",
        str(out_dir.relative_to(ROOT)),
        "--save-npz",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f'launched {job["name"]}: pid={proc.pid} gpu={job["gpu"]} log={log_path}')
    return proc.pid


def _e14_jobs() -> list[dict]:
    jobs: list[dict] = []
    gpu = 0
    for model in ("uncond_local", "uncond_global"):
        for setting in ("L", "M"):
            for start, end in ((0, 50), (50, 100)):
                name = f"e14_{model}_{setting}_{start:03d}_{end:03d}"
                jobs.append({
                    "name": name,
                    "gpu": gpu,
                    "task": "E14",
                    "setting": setting,
                    "model": model,
                    "start": start,
                    "end": end,
                    "out": name,
                })
                gpu += 1
    return jobs


def _e15_jobs() -> list[dict]:
    jobs: list[dict] = []
    gpu = 0
    for model in ("uncond_local", "uncond_global"):
        for start, end in ((0, 50), (50, 100), (100, 150), (150, 200)):
            name = f"e15_{model}_default_{start:03d}_{end:03d}"
            jobs.append({
                "name": name,
                "gpu": gpu,
                "task": "E15",
                "setting": "default",
                "model": model,
                "start": start,
                "end": end,
                "out": name,
            })
            gpu += 1
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", choices=["e14", "e15"], required=True)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = _e14_jobs() if args.machine == "e14" else _e15_jobs()
    pid_path = OUT_ROOT / f"{args.machine}_pids.txt"
    pids = [_launch(job) for job in jobs]
    pid_path.write_text("\n".join(map(str, pids)) + "\n")
    print(f"wrote {pid_path}")


if __name__ == "__main__":
    main()
