#!/usr/bin/env python3
"""Aggregate SONIC official IsaacLab eval logs for Table-2 shards."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


RATE_PATTERNS = {
    "success_rate": re.compile(r"Success Rate:\s*([0-9.]+)"),
    "progress_rate": re.compile(r"Progress Rate:\s*([0-9.]+)"),
}
ALL_PATTERN = re.compile(r"All:\s*(.*)")
METRIC_PATTERN = re.compile(r"([A-Za-z0-9_]+):\s*([-+0-9.eE]+)")


def parse_log(path: Path) -> dict[str, float]:
    text = path.read_text(errors="ignore")
    row: dict[str, float] = {}
    for key, pattern in RATE_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            row[key] = float(matches[-1])
    all_lines = ALL_PATTERN.findall(text)
    if all_lines:
        for key, value in METRIC_PATTERN.findall(all_lines[-1]):
            row[key] = float(value)
    if "success_rate" not in row or "mpjpe_l" not in row:
        raise ValueError(f"{path}: cannot parse official SONIC metrics")
    return row


def weighted_average(rows: list[dict[str, float]]) -> dict[str, float]:
    total = sum(float(row.get("num_cases", 0.0)) for row in rows)
    out: dict[str, float] = {"num_cases": total, "num_shards": float(len(rows))}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"num_cases", "num_shards", "split", "shard"} and isinstance(value, (int, float))
        }
    )
    for key in keys:
        values = []
        weights = []
        for row in rows:
            if key in row and not math.isnan(float(row[key])):
                values.append(float(row[key]))
                weights.append(float(row.get("num_cases", 0.0)))
        if values and sum(weights) > 0:
            out[key] = sum(v * w for v, w in zip(values, weights)) / sum(weights)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["lafan1_fixed600", "amass_test_fixed600", "wild_clean_fixed600"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result: dict[str, object] = {"splits": {}, "shards": [], "failed_logs": []}
    all_rows: list[dict[str, float]] = []
    for split in args.splits:
        split_rows: list[dict[str, float]] = []
        for manifest in sorted((args.root / split).glob("shard_*/manifest.json")):
            shard_dir = manifest.parent
            log_path = shard_dir / "eval.log"
            if not log_path.exists():
                continue
            case_names = json.loads(manifest.read_text())
            try:
                row = parse_log(log_path)
            except ValueError as exc:
                result["failed_logs"].append({"path": str(log_path), "error": str(exc)})  # type: ignore[index]
                continue
            row["num_cases"] = float(len(case_names))
            row["split"] = split  # type: ignore[assignment]
            row["shard"] = shard_dir.name  # type: ignore[assignment]
            split_rows.append(row)
            all_rows.append(row)
        result["splits"][split] = weighted_average(split_rows) if split_rows else {"num_cases": 0.0}
    result["overall"] = weighted_average(all_rows)
    result["shards"] = all_rows

    output = args.output or args.root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["splits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
