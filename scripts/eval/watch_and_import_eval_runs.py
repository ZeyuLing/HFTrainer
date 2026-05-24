#!/usr/bin/env python3
"""Wait for Taiji eval tasks to finish, then import eval_v2 outputs."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "motion_annot_web" / "eval_dashboard" / "eval_dashboard.db"


def task_is_running(task: str) -> bool:
    proc = subprocess.run(
        ["taiji_client", "il", task],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    line = " ".join(
        ln.strip() for ln in proc.stdout.splitlines()
        if "PENDING" in ln or "TRAINING_RUNNING" in ln or "END" in ln or "FAILED" in ln
    )
    print(f"[watch] {datetime.now():%F %T} {task}: {line[:300]}", flush=True)
    return "PENDING" in proc.stdout or "TRAINING_RUNNING" in proc.stdout


def parse_import(spec: str) -> Tuple[Path, str]:
    if "::" in spec:
        root, notes = spec.split("::", 1)
    else:
        root, notes = spec, ""
    return PROJECT_ROOT / root, notes


def import_root(root: Path, notes: str) -> None:
    if not root.exists():
        print(f"[watch] skip missing root: {root}", flush=True)
        return
    eval_count = len(list(root.rglob("eval_v2_*.json")))
    print(f"[watch] importing {root} eval_jsons={eval_count}", flush=True)
    cmd = [
        "python3",
        "scripts/eval/split_and_import_eval_v2.py",
        str(root.relative_to(PROJECT_ROOT)),
    ]
    if notes:
        cmd.extend(["--notes", notes])
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--imports", nargs="+", required=True,
                        help="Import specs as root::notes")
    parser.add_argument("--poll-sec", type=int, default=300)
    args = parser.parse_args()

    imports: List[Tuple[Path, str]] = [parse_import(x) for x in args.imports]
    print(f"[watch] start {datetime.now():%F %T}", flush=True)
    print(f"[watch] tasks={args.tasks}", flush=True)
    while True:
        running = any(task_is_running(task) for task in args.tasks)
        if not running:
            break
        time.sleep(args.poll_sec)

    backup = DB_PATH.with_name(f"{DB_PATH.name}.bak_all_latest_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(DB_PATH, backup)
    print(f"[watch] db backup {backup}", flush=True)
    for root, notes in imports:
        import_root(root, notes)
    print(f"[watch] import done {datetime.now():%F %T}", flush=True)


if __name__ == "__main__":
    main()
