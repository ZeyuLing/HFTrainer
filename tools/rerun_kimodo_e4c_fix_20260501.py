#!/usr/bin/env python3
"""Rerun KIMODO E4 C_rhand_lfoot after fixing LeftFoot constraint mapping."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "work_dirs" / "eval_8082_refresh_20260501"
LOG_DIR = RUN_ROOT / "logs" / "kimodo"
TARGETS = [
    ("caption", "yes", 0, 25),
    ("caption", "yes", 25, 50),
    ("caption", "yes", 50, 75),
    ("caption", "yes", 75, 100),
    ("uncond", "no", 0, 25),
    ("uncond", "no", 25, 50),
    ("uncond", "no", 50, 75),
    ("uncond", "no", 75, 100),
]


def _name(tag: str, start: int, end: int) -> str:
    return f"kimodo_{tag}_E4_C_rhand_lfoot_{start:03d}_{end:03d}"


def _cmd(use_caption: str, start: int, end: int, out_dir: Path) -> list[str]:
    cmd = [
        "python3",
        "tools/run_kimodo_all_tasks.py",
        "--tasks",
        "E4",
        "--settings",
        "C_rhand_lfoot",
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

    for tag, _use_caption, start, end in TARGETS:
        name = _name(tag, start, end)
        shutil.rmtree(RUN_ROOT / "kimodo" / name, ignore_errors=True)
        log_path = LOG_DIR / f"{name}.log"
        if log_path.exists():
            log_path.unlink()

    running: dict[int, tuple[subprocess.Popen, str, object]] = {}
    failed: list[str] = []
    for gpu, (tag, use_caption, start, end) in enumerate(TARGETS):
        name = _name(tag, start, end)
        out_dir = RUN_ROOT / "kimodo" / name
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = (LOG_DIR / f"{name}.log").open("wb")
        proc = subprocess.Popen(
            _cmd(use_caption, start, end, out_dir),
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        running[gpu] = (proc, name, log)
        print(f"[launch] gpu={gpu} pid={proc.pid} {name}", flush=True)

    done = 0
    while running:
        time.sleep(10)
        for gpu, (proc, name, log) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            done += 1
            print(f"[done] {done}/8 gpu={gpu} rc={rc} {name}", flush=True)
            if rc != 0:
                failed.append(name)
            del running[gpu]

    if failed:
        raise SystemExit(f"failed jobs: {failed}")
    print("[all done] kimodo E4 C_rhand_lfoot fix", flush=True)


if __name__ == "__main__":
    main()
