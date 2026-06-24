#!/usr/bin/env python3
"""Rerun all KIMODO E4 shards after fixing multiprompt transition device."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "work_dirs" / "eval_8082_refresh_20260501"
LOG_DIR = RUN_ROOT / "logs" / "kimodo"
SETTINGS = (
    "A_rhand_sparse",
    "B_ankles_sparse",
    "C_rhand_lfoot",
    "D_both_hands",
    "E_all4_sparse",
    "F_rhand_dense",
)
SHARDS = ((0, 25), (25, 50), (50, 75), (75, 100))
GPU_COUNT = 8


def _name(tag: str, setting: str, start: int, end: int) -> str:
    return f"kimodo_{tag}_E4_{setting}_{start:03d}_{end:03d}"


def _cmd(setting: str, use_caption: str, start: int, end: int, out_dir: Path) -> list[str]:
    cmd = [
        "python3",
        "tools/run_kimodo_all_tasks.py",
        "--tasks",
        "E4",
        "--settings",
        setting,
        "--max-samples",
        "100",
        "--start-idx",
        str(start),
        "--end-idx",
        str(end),
        "--use-caption",
        use_caption,
        "--output-dir",
        str(out_dir.relative_to(ROOT)),
    ]
    if use_caption == "yes":
        cmd.append("--use-rewritten")
    return cmd


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs: deque[tuple[str, str, str, int, int]] = deque()
    for setting in SETTINGS:
        for tag, use_caption in (("caption", "yes"), ("uncond", "no")):
            for start, end in SHARDS:
                jobs.append((tag, use_caption, setting, start, end))

    for tag, _use_caption, setting, start, end in jobs:
        name = _name(tag, setting, start, end)
        shutil.rmtree(RUN_ROOT / "kimodo" / name, ignore_errors=True)
        log_path = LOG_DIR / f"{name}.log"
        if log_path.exists():
            log_path.unlink()

    running: dict[int, tuple[subprocess.Popen, str, object]] = {}
    failed: list[str] = []
    done = 0

    while jobs or running:
        for gpu in range(GPU_COUNT):
            if gpu in running or not jobs:
                continue
            tag, use_caption, setting, start, end = jobs.popleft()
            name = _name(tag, setting, start, end)
            out_dir = RUN_ROOT / "kimodo" / name
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = (LOG_DIR / f"{name}.log").open("wb")
            proc = subprocess.Popen(
                _cmd(setting, use_caption, start, end, out_dir),
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (proc, name, log)
            print(f"[launch] gpu={gpu} pid={proc.pid} {name}", flush=True)

        time.sleep(10)
        for gpu, (proc, name, log) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            done += 1
            print(f"[done] {done}/48 gpu={gpu} rc={rc} {name}", flush=True)
            if rc != 0:
                failed.append(name)
            del running[gpu]

    if failed:
        raise SystemExit(f"failed jobs: {failed}")
    print("[all done] kimodo E4 device fix", flush=True)


if __name__ == "__main__":
    main()
