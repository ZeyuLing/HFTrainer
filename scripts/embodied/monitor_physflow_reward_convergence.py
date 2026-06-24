#!/usr/bin/env python3
"""Monitor PhysFlow reward logs and emit a live convergence table."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

from analyze_physflow_reward_convergence import analyze_one


def latest_log(work_dir: Path) -> Path | None:
    logs = sorted(work_dir.glob("*/train.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def run_once(args: argparse.Namespace) -> Dict:
    methods: Dict[str, Tuple[Path, Path | None]] = {}
    for spec in args.method:
        if "=" not in spec:
            raise SystemExit(f"--method must be LABEL=/path/to/work_dir, got {spec}")
        label, value = spec.split("=", 1)
        work_dir = Path(value)
        methods[label] = (work_dir, latest_log(work_dir))

    report = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": {
            "window": args.window,
            "min_good": args.min_good,
            "max_reward_rel_delta": args.max_reward_rel_delta,
            "max_late_slope": args.max_late_slope,
        },
        "methods": {},
    }
    for label, (work_dir, log_path) in methods.items():
        if log_path is None:
            report["methods"][label] = {
                "status": "no_log",
                "work_dir": str(work_dir),
                "comparable": False,
                "reason": "no train.log found",
            }
            continue
        item = analyze_one(
            log_path,
            window=args.window,
            min_good=args.min_good,
            max_reward_rel_delta=args.max_reward_rel_delta,
            max_late_slope=args.max_late_slope,
        )
        item["work_dir"] = str(work_dir)
        report["methods"][label] = item
    return report


def write_outputs(report: Dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    md = out.with_suffix(".md")
    lines = [
        f"# PhysFlow Reward Convergence Live",
        "",
        f"Updated: {report['updated_at']}",
        "",
        "| Method | Step | Comparable | late n_good | late reward_best | reward rel delta | Reason |",
        "|---|---:|:---:|---:|---:|---:|---|",
    ]
    for label, item in report["methods"].items():
        late = item.get("late") or {}
        step = item.get("max_step", "-")
        comparable = "yes" if item.get("comparable") else "no"
        n_good = late.get("n_good_mean")
        reward = late.get("reward_best_mean")
        rel = item.get("late_reward_rel_delta")
        def fmt(x):
            return "-" if x is None else f"{float(x):.4f}"
        reason = str(item.get("reason", "")).replace("|", "/")
        lines.append(
            f"| {label} | {step} | {comparable} | {fmt(n_good)} | "
            f"{fmt(reward)} | {fmt(rel)} | {reason} |"
        )
    md.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", required=True, help="LABEL=/path/to/work_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--min-good", type=float, default=0.8)
    ap.add_argument("--max-reward-rel-delta", type=float, default=0.08)
    ap.add_argument("--max-late-slope", type=float, default=0.003)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=900)
    args = ap.parse_args()

    out = Path(args.out)
    while True:
        report = run_once(args)
        write_outputs(report, out)
        print(json.dumps(report, indent=2))
        if not args.loop:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
