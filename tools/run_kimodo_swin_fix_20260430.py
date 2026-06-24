#!/usr/bin/env python3
"""Shard KIMODO sliding-window reruns for active problematic 8083 settings."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "work_dirs" / "kimodo_swin_fix_20260430"

TARGETS = {
    "e3": [("E3", "every_30f", 240), ("E3", "every_60f", 240), ("E3", "adaptive", 240)],
    "e14": [("E14", "M", 100), ("E14", "L", 100)],
}


def _shards(n: int, shard_size: int) -> list[tuple[int, int]]:
    return [(s, min(s + shard_size, n)) for s in range(0, n, shard_size)]


def _cmd(task: str, setting: str, max_samples: int, start: int, end: int, out_dir: Path) -> list[str]:
    return [
        "python3",
        "tools/run_kimodo_all_tasks.py",
        "--tasks",
        task,
        "--settings",
        setting,
        "--max-samples",
        str(max_samples),
        "--start-idx",
        str(start),
        "--end-idx",
        str(end),
        "--use-caption",
        "no",
        "--output-dir",
        str(out_dir.relative_to(ROOT)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(TARGETS), required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=25)
    args = parser.parse_args()

    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, list[str]]] = []
    for task, setting, n in TARGETS[args.group]:
        for start, end in _shards(n, args.shard_size):
            name = f"kimodo_{task}_{setting}_{start:03d}_{end:03d}"
            out_dir = OUT_ROOT / name
            jobs.append((name, _cmd(task, setting, n, start, end, out_dir)))

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
    print(f"[all done] {len(jobs)} jobs for group={args.group}", flush=True)


if __name__ == "__main__":
    main()
