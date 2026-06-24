#!/usr/bin/env python3
"""Monitor generator-to-tracker replay experiments submitted to Taiji."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


DEFAULT_TASKS = [
    {
        "name": "any2track_kwfix",
        "flag": "physflow-genany2track-kwfix0617-V100-1x8-2224",
        "log": "work_dirs/physflow_genany2track_seed0617_kwfix.log",
    },
    {
        "name": "proto_copyfix",
        "flag": "physflow-genproto-copyfix0617-V100-1x1-2159",
        "log": "work_dirs/physflow_genproto_seed0617_copyfix.log",
    },
    {
        "name": "any2track_copyfix",
        "flag": "physflow-genany2track-copyfix0617-V100-1x8-2159",
        "log": "work_dirs/physflow_genany2track_seed0617_copyfix.log",
    },
    {
        "name": "proto_v100dha_rerun",
        "flag": "physflow-genproto-v100dha-rerun0617-V100-1x1-1945",
        "log": "work_dirs/physflow_genproto_seed0617_v100dha_rerun.log",
    },
    {
        "name": "any2track_v100dha_rerun",
        "flag": "physflow-genany2track-v100dha-rerun0617-V100-1x8-1948",
        "log": "work_dirs/physflow_genany2track_seed0617_v100dha_rerun.log",
    },
    {
        "name": "any2track_v100_pending",
        "flag": "physflow-genany2track-seed0617-V100-8x1-V100-1x8-1244",
        "log": "work_dirs/physflow_genany2track_seed0617.log",
    },
]


def _run(cmd: list[str], timeout: int = 60) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def _parse_state(table: str) -> str:
    for state in (
        "TRAINING_RUNNING",
        "TRAINING_INIT",
        "TRAINING_RESOURCE_WAITING",
        "TRAINING_SUCCEEDED",
        "TRAINING_FAILED",
        "PENDING",
        "RUNNING",
        "END",
        "FAILED",
        "SUCCEEDED",
    ):
        if state in table:
            return state
    m = re.search(r"\|\s*(false|true)\s*\|\s*([A-Z_]+)\s*\|", table)
    return m.group(2) if m else "UNKNOWN"


def _tail(path: Path, n: int = 24) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(errors="ignore").splitlines()
    return lines[-n:]


def _snapshot(tasks: list[dict[str, str]]) -> dict[str, object]:
    rows = []
    for task in tasks:
        flag = task["flag"]
        table = _run(["taiji_client", "il", flag], timeout=60)
        rows.append(
            {
                "name": task["name"],
                "flag": flag,
                "state": _parse_state(table),
                "log": task["log"],
                "tail": _tail(Path(task["log"])),
            }
        )
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tasks": rows}


def _write_summary(record: dict[str, object], path: Path) -> None:
    lines = [
        "# Generator -> Tracker Replay Monitor",
        "",
        f"Updated: {record['time']}",
        "",
        "| Experiment | State | Task | Log |",
        "|---|---|---|---|",
    ]
    for row in record["tasks"]:
        lines.append(f"| {row['name']} | {row['state']} | `{row['flag']}` | `{row['log']}` |")
    lines.append("")
    for row in record["tasks"]:
        lines.append(f"## {row['name']}")
        lines.append("")
        tail = row.get("tail") or []
        if tail:
            lines.append("```text")
            lines.extend(tail[-24:])
            lines.append("```")
        else:
            lines.append("_No local log yet._")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="output/generator_tracker_replay/physflow_hg1_seed_20260617/monitor_summary.md",
    )
    parser.add_argument(
        "--jsonl",
        default="output/generator_tracker_replay/physflow_hg1_seed_20260617/monitor.jsonl",
    )
    parser.add_argument("--loops", type=int, default=288)
    parser.add_argument("--sleep-sec", type=int, default=300)
    args = parser.parse_args()

    out = Path(args.out)
    jsonl = Path(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    for i in range(args.loops):
        rec = _snapshot(DEFAULT_TASKS)
        rec["loop"] = i
        with jsonl.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _write_summary(rec, out)
        if i + 1 < args.loops:
            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()
