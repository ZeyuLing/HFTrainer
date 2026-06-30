#!/usr/bin/env python3
"""Aggregate SONIC unified Table-2 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = [
    "success",
    "paper_success",
    "strict_success",
    "completion",
    "root_err_mean",
    "root_err_max",
    "root_height_err_mean",
    "root_height_err_max",
    "raw_body_err_mean",
    "raw_body_err_max",
    "body_err_mean",
    "body_err_max",
    "xy_aligned_body_err_mean",
    "xy_aligned_body_err_max",
    "local_body_err_mean",
    "local_body_err_max",
    "body_vel_err_mean",
    "local_body_vel_err_mean",
    "body_acc_err_mean",
    "local_body_acc_err_mean",
    "raw_global_mpjpe_m",
    "raw_global_mpjpe_mm",
    "xy_aligned_mpjpe_m",
    "xy_aligned_mpjpe_mm",
    "mpjpe_m",
    "mpjpe_mm",
    "local_mpjpe_m",
    "local_mpjpe_mm",
    "mpjve_mps",
    "local_mpjve_mps",
    "mpjae_mps2",
    "local_mpjae_mps2",
    "joint_err_mean",
    "max_joint_err_mean",
    "max_joint_err_max",
    "min_height",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.eval_root.glob("*/metrics.json")):
        row = json.loads(path.read_text())
        row["case_dir"] = path.parent.name
        rows.append(row)
    summary = {"num_motions": len(rows)}
    for metric in METRICS:
        key = f"{metric}_rate" if metric in {"success", "paper_success", "strict_success"} else metric
        vals = [float(row.get(metric, np.nan)) for row in rows]
        summary[key] = float(np.nanmean(vals)) if vals else float("nan")
    (args.eval_root / "summary.json").write_text(json.dumps({"summary": summary, "motions": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
