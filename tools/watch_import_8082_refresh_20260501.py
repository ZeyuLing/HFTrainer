#!/usr/bin/env python3
"""Watch the 2026-05-01 8082 refresh and import finished groups.

The watcher is intentionally conservative:
  * import only after the scheduler prints "[all done]";
  * stop on scheduler failures;
  * merge validates complete sample coverage before writing import JSONs;
  * each import writes a small marker so rerunning the watcher is idempotent.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "work_dirs" / "eval_8082_refresh_20260501"
IMPORT_ROOT = RUN_ROOT / "import_jsons"
STATE_ROOT = RUN_ROOT / "watch_state"


GROUPS = {
    "hymotion": RUN_ROOT / "driver_hymotion_debug1.log",
    "kimodo": RUN_ROOT / "driver_kimodo_debug2.log",
}


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except FileNotFoundError:
        return ""


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _import_group(group: str) -> None:
    marker = STATE_ROOT / f"{group}.imported"
    if marker.exists():
        print(f"[skip] {group} already imported: {marker}", flush=True)
        return

    _run(["python3", "tools/merge_8082_refresh_shards_20260501.py", "--group", group])
    json_dir = IMPORT_ROOT / group
    paths = sorted(json_dir.glob("*.json"))
    if not paths:
        raise RuntimeError(f"No import JSONs found for {group} under {json_dir}")

    notes = f"8082 full refresh 20260501 — {group} latest production rerun"
    for path in paths:
        _run([
            "python3",
            "motion_annot_web/eval_dashboard/data_importer.py",
            "import",
            str(path),
            "--notes",
            notes,
        ])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"imported {len(paths)} files at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"[imported] {group}: {len(paths)} files", flush=True)


def _status(group: str, text: str) -> str:
    if "failed jobs:" in text:
        return "failed"
    if "[all done]" in text:
        return "done"
    if "[scheduler]" in text:
        return "running"
    return "pending"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        statuses = {}
        for group, log_path in GROUPS.items():
            text = _read(log_path)
            st = _status(group, text)
            statuses[group] = st
            print(f"[status] {group}: {st}", flush=True)
            if st == "failed":
                raise SystemExit(f"{group} scheduler failed; inspect {log_path}")
            if st == "done":
                _import_group(group)

        if all(st == "done" for st in statuses.values()):
            print("[done] all finished and imported", flush=True)
            return
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
