#!/usr/bin/env python3
"""Run several seeds for E8-D sample 35 on one debug machine."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "work_dirs" / "e8_d_case35_seed_sweep_20260430"


def main() -> None:
    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pids: list[int] = []
    for gpu, seed_offset in enumerate(range(1000, 1008)):
        name = f"e8_d_uncond_local_sample035_seed{seed_offset}"
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
            "uncond_local",
            "--max-samples",
            "36",
            "--start-index",
            "35",
            "--end-index",
            "36",
            "--seed-offset",
            str(seed_offset),
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
        print(f"launched seed_offset={seed_offset}: pid={proc.pid} gpu={gpu} log={log_path}")
    (OUT_ROOT / "pids.txt").write_text("\n".join(map(str, pids)) + "\n")


if __name__ == "__main__":
    main()
