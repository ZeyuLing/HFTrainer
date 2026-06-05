#!/usr/bin/env python3
"""Aggregate ProtoMotions inference_agent --full-eval shard logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import torch


METRIC_RE = re.compile(r"^\s+(eval/[^:]+):\s+([-+eE0-9.]+)\s*$")
SCORE_RE = re.compile(r"^\s+Overall Score:\s+([-+eE0-9.]+)\s*$")


def parse_log(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    in_block = False
    for line in path.read_text(errors="replace").splitlines():
        if "EVALUATION RESULTS" in line:
            in_block = True
            continue
        if not in_block:
            continue
        m = METRIC_RE.match(line)
        if m:
            metrics[m.group(1)] = float(m.group(2))
            continue
        m = SCORE_RE.match(line)
        if m:
            metrics["overall_score"] = float(m.group(1))
    return metrics


def shard_motion_count(motion_base: Path, shard: int, template: str) -> int:
    pt = motion_base / template.format(shard=shard)
    data = torch.load(pt, map_location="cpu", weights_only=False)
    return int(len(data["motion_lengths"]))


def aggregate(rows: Dict[int, Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    total = sum(n for n, _ in rows.values())
    if total <= 0:
        raise ValueError("No motions found across shards")

    keys = sorted({k for _, metrics in rows.values() for k in metrics.keys()})
    out: Dict[str, float] = {"num_motions": float(total), "num_shards": float(len(rows))}

    for key in keys:
        present = [(n, m[key]) for n, m in rows.values() if key in m]
        if not present:
            continue
        if key.endswith("/max"):
            out[key] = max(v for _, v in present)
        elif key.endswith("/min"):
            out[key] = min(v for _, v in present)
        else:
            denom = sum(n for n, _ in present)
            out[key] = sum(n * v for n, v in present) / denom

    if "eval/gt_error/mean" in out:
        out["eval/gt_error/mean_mm"] = out["eval/gt_error/mean"] * 1000.0
    if "eval/anchor_height_error/mean" in out:
        out["eval/anchor_height_error/mean_mm"] = (
            out["eval/anchor_height_error/mean"] * 1000.0
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True, type=Path)
    ap.add_argument("--motion-base", required=True, type=Path)
    ap.add_argument("--num-shards", required=True, type=int)
    ap.add_argument(
        "--shard-file-template",
        default="amass_g1_full_shard_{shard}.pt",
        help="MotionLib shard filename template relative to --motion-base.",
    )
    args = ap.parse_args()

    eval_root: Path = args.eval_root
    all_results = {}
    missing = {}

    for eval_dir in sorted(eval_root.glob("eval_*")):
        name = eval_dir.name[len("eval_") :]
        rows: Dict[int, Tuple[int, Dict[str, float]]] = {}
        missing_logs = []
        for shard in range(args.num_shards):
            log = eval_dir / f"shard_{shard}.log"
            if not log.exists():
                missing_logs.append(str(log))
                continue
            metrics = parse_log(log)
            if not metrics:
                missing_logs.append(str(log))
                continue
            rows[shard] = (
                shard_motion_count(args.motion_base, shard, args.shard_file_template),
                metrics,
            )
        if rows:
            all_results[name] = aggregate(rows)
        if missing_logs:
            missing[name] = missing_logs

    summary = {"results": all_results, "missing_or_incomplete_logs": missing}
    (eval_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    metric_order = [
        "num_motions",
        "eval/success_rate",
        "overall_score",
        "eval/relative_body_pos/failure_rate",
        "eval/anchor_height_error/failure_rate",
        "eval/gt_error/mean",
        "eval/gt_error/mean_mm",
        "eval/gt_error/max",
        "eval/max_joint_error/mean",
        "eval/max_joint_error/max",
        "eval/gr_error/mean",
        "eval/normalized_jerk_mean",
        "eval/action_delta_mean_rad",
    ]

    lines = []
    lines.append("| baseline | " + " | ".join(metric_order) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(metric_order)) + "|")
    for name, metrics in sorted(all_results.items()):
        vals = []
        for key in metric_order:
            val = metrics.get(key)
            vals.append("" if val is None else f"{val:.6g}")
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    if missing:
        lines.append("")
        lines.append("Incomplete logs:")
        for name, logs in sorted(missing.items()):
            lines.append(f"- {name}: {len(logs)} missing/incomplete shard logs")

    (eval_root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
