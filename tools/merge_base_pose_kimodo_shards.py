#!/usr/bin/env python3
"""Merge sharded KIMODO Base Pose Edit outputs into one variant directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


METRIC_KEYS = ["kf_mpjpe", "global_mpjpe", "src_mpjpe", "overall_smoothness", "foot_skating"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result_path in sorted(args.shards_root.glob("*/results.json")):
        data = json.loads(result_path.read_text())
        rows.extend(data.get("cases", []))
        for npz_path in sorted(result_path.parent.glob("*.npz")):
            shutil.copy2(npz_path, args.out_dir / npz_path.name)

    rows.sort(key=lambda r: r.get("case_key", ""))
    aggregate = {}
    for key in METRIC_KEYS:
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            aggregate[f"{key}_mean"] = float(np.mean(vals))

    (args.out_dir / "results.json").write_text(
        json.dumps({"aggregate": aggregate, "cases": rows}, indent=2)
    )
    print(json.dumps({"out_dir": str(args.out_dir), "num_cases": len(rows), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
