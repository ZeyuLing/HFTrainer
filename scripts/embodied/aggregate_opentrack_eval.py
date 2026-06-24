#!/usr/bin/env python3
"""Aggregate sharded OpenTrack eval JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = [
    "success",
    "paper_success",
    "strict_success",
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
    for path in sorted(args.eval_root.glob("eval_shard_*.json")):
        data = json.loads(path.read_text())
        for row in data.get("motions", []):
            row = dict(row)
            row["shard"] = path.stem.replace("eval_shard_", "")
            if "strict_success" not in row:
                row["strict_success"] = bool(
                    row.get("paper_success")
                    and float(row.get("root_err_mean", float("inf"))) <= 1.0
                    and float(row.get("max_joint_err_max", float("inf"))) <= 0.7
                )
            rows.append(row)

    summary = {"num_motions": len(rows)}
    for metric in METRICS:
        if metric == "success":
            key = "success_rate"
        elif metric == "paper_success":
            key = "paper_success_rate"
        elif metric == "strict_success":
            key = "strict_success_rate"
        else:
            key = metric
        summary[key] = float(np.mean([row[metric] for row in rows])) if rows else float("nan")

    out_json = args.eval_root / "summary.json"
    out_md = args.eval_root / "summary.md"
    out_json.write_text(json.dumps({"summary": summary, "motions": rows}, indent=2) + "\n")

    lines = [
        "# Any2Track AMASS-G1 Evaluation",
        "",
        f"- num_motions: {summary['num_motions']}",
        f"- success_rate: {summary['success_rate']:.6f}",
        f"- paper_success_rate: {summary['paper_success_rate']:.6f}",
        f"- strict_success_rate: {summary['strict_success_rate']:.6f}",
        f"- raw_global_mpjpe_mm: {summary.get('raw_global_mpjpe_mm', float('nan')):.6f}",
        f"- xy_aligned_mpjpe_mm: {summary.get('xy_aligned_mpjpe_mm', float('nan')):.6f}",
        f"- mpjpe_mm: {summary['mpjpe_mm']:.6f}",
        f"- local_mpjpe_mm: {summary.get('local_mpjpe_mm', float('nan')):.6f}",
        f"- mpjve_mps: {summary['mpjve_mps']:.6f}",
        f"- local_mpjve_mps: {summary.get('local_mpjve_mps', float('nan')):.6f}",
        f"- mpjae_mps2: {summary.get('mpjae_mps2', float('nan')):.6f}",
        f"- local_mpjae_mps2: {summary.get('local_mpjae_mps2', float('nan')):.6f}",
        f"- root_height_err_mean: {summary['root_height_err_mean']:.6f}",
        f"- root_err_mean: {summary['root_err_mean']:.6f}",
        f"- body_err_mean: {summary['body_err_mean']:.6f}",
        f"- body_vel_err_mean: {summary['body_vel_err_mean']:.6f}",
        f"- local_body_err_mean: {summary.get('local_body_err_mean', float('nan')):.6f}",
        f"- local_body_vel_err_mean: {summary.get('local_body_vel_err_mean', float('nan')):.6f}",
        f"- joint_err_mean: {summary['joint_err_mean']:.6f}",
        f"- max_joint_err_max: {summary['max_joint_err_max']:.6f}",
        f"- min_height: {summary['min_height']:.6f}",
        "",
    ]
    out_md.write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
