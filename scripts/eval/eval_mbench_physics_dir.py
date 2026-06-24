#!/usr/bin/env python3
"""Evaluate MBench-style physical metrics for one motion directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hftrainer.evaluation.motion.mbench_physics import (
    evaluate_mbench_physics_dir,
    table_scaled_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute joint-based MBench physical metrics for motion_135 npz or MotionStreamer-272 files."
    )
    parser.add_argument("--src", required=True, help="Directory containing input motion files.")
    parser.add_argument("--mode", required=True, choices=["m135", "gt272"])
    parser.add_argument("--out-json", required=True, help="Where to write raw and table-scaled metrics.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_mbench_physics_dir(
        args.src,
        mode=args.mode,
        limit=args.limit,
        seed=args.seed,
        workers=args.workers,
    )
    payload = {
        "src": str(Path(args.src).resolve()),
        "mode": args.mode,
        "workers": args.workers,
        "limit": args.limit,
        "seed": args.seed,
        "raw": metrics,
        "table": table_scaled_metrics(metrics),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
