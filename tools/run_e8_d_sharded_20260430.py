#!/usr/bin/env python3
"""Launch sharded E8-D latest-checkpoint reruns on a debug machine."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "work_dirs" / "e8_d_rerun_latest_20260430"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["uncond_local", "uncond_global"], required=True)
    args = parser.parse_args()

    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pids: list[int] = []
    for gpu, start in enumerate(range(0, 200, 25)):
        end = start + 25
        name = f"e8_d_{args.model}_{start:03d}_{end:03d}"
        out_dir = OUT_ROOT / name
        log_path = log_dir / f"{name}.log"
        cmd = [
            "python3",
            "tools/eval_m2m_v2_all_tasks.py",
            "--tasks",
            "E8",
            "--settings",
            "D",
            "--models",
            args.model,
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
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open("wb") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pids.append(proc.pid)
        print(f"launched {name}: pid={proc.pid} gpu={gpu} log={log_path}")

    pid_path = OUT_ROOT / f"{args.model}_pids.txt"
    pid_path.write_text("\n".join(map(str, pids)) + "\n")
    print(f"wrote {pid_path}")


if __name__ == "__main__":
    main()
