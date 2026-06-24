#!/usr/bin/env python3
"""Aggregate sharded MBench evaluator JSON files.

The official evaluator writes per-motion scalar values plus an aggregate.  For
sharded runs, recomputing the aggregate from the concatenated per-motion list is
less error-prone than averaging shard means.
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    return sorted(set(paths), key=lambda p: str(p))


def aggregate_metric(metric: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    for entry in entries:
        motion_id = int(entry["id"])
        if motion_id in by_id:
            duplicates.append(motion_id)
        by_id[motion_id] = entry
    if duplicates:
        dup = ", ".join(str(x) for x in sorted(set(duplicates))[:12])
        raise ValueError(f"{metric} has duplicate motion ids: {dup}")

    ordered = [by_id[k] for k in sorted(by_id)]
    values = np.asarray([float(row["value"]) for row in ordered], dtype=np.float32)
    aggregate = {
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size > 1 else 0.0,
        "num_samples": int(values.size),
    }
    return {"aggregate": aggregate, "per_motion": ordered}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files or glob patterns.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="mbench_results")
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f"No input JSONs matched: {args.inputs}")

    metric_entries: dict[str, list[dict[str, Any]]] = {}
    sources: list[str] = []
    for path in paths:
        data = load_json(path)
        sources.append(str(path))
        for metric, payload in data.items():
            if not isinstance(payload, dict):
                continue
            per_motion = payload.get("per_motion")
            if not per_motion:
                continue
            metric_entries.setdefault(metric, []).extend(per_motion)

    if not metric_entries:
        raise ValueError(f"No per_motion entries found in {len(paths)} input JSONs")

    merged = {metric: aggregate_metric(metric, rows) for metric, rows in sorted(metric_entries.items())}
    merged["_sharded_sources"] = {"files": sources, "num_files": len(sources)}

    out_dir = Path(args.output_dir)
    stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    out_path = out_dir / f"{args.name}_{stamp}_eval_results.json"
    write_json(out_path, merged)
    print(f"[aggregate-mbench] wrote {out_path}")
    for metric, payload in merged.items():
        if not isinstance(payload, dict) or "aggregate" not in payload:
            continue
        agg = payload["aggregate"]
        print(f"{metric}: mean={agg['mean']:.6f} std={agg['std']:.6f} n={agg['num_samples']}")


if __name__ == "__main__":
    main()
