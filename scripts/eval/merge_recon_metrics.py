#!/usr/bin/env python3
"""Merge sharded reconstruction metric JSON files."""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "num_samples": int(arr.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths: list[str] = []
    for item in args.inputs:
        matches = sorted(glob.glob(item))
        paths.extend(matches if matches else [item])
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise ValueError("No input metric files found")

    payloads = [load_json(path) for path in paths]
    first = payloads[0]
    per_case: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    subsets: Counter = Counter()
    selected = 0
    selected_before = 0
    skipped: Counter = Counter()

    for payload in payloads:
        per_case.extend(payload.get("per_case", []))
        failures.extend(payload.get("failures", []))
        subsets.update(payload.get("subsets", {}))
        selected += int(payload.get("selected_samples", 0))
        selected_before = max(selected_before, int(payload.get("selected_samples_before_shard", 0)))
        for key, value in payload.items():
            if key.startswith("skipped_") and isinstance(value, int):
                skipped[key] = max(skipped[key], value)

    metric_keys = sorted(
        key
        for key in {
            item_key
            for item in per_case
            for item_key in item.keys()
            if item_key.endswith("_mm") or item_key.endswith("_deg")
        }
        if key not in {"frames"}
    )
    summary = {key: summarize([float(item[key]) for item in per_case if key in item]) for key in metric_keys}
    first_summary = first.get("summary", {})
    if "cb_util_percent" in first_summary:
        summary["cb_util_percent"] = first_summary.get("cb_util_percent")
    summary["num_failures"] = len(failures)

    merged = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "merged_from": paths,
        "method": first.get("method"),
        "checkpoint": first.get("checkpoint"),
        "anno_file": first.get("anno_file"),
        "id_list": first.get("id_list"),
        "data_dir": first.get("data_dir"),
        "num_person": first.get("num_person"),
        "max_duration": first.get("max_duration"),
        "selected_samples": selected,
        "selected_samples_before_shard": selected_before,
        **dict(skipped),
        "subsets": dict(subsets),
        "summary": summary,
        "failures": failures,
        "per_case": sorted(per_case, key=lambda item: item.get("key", "")),
    }
    write_json(args.output, merged)
    print(json.dumps(merged["summary"], indent=2, ensure_ascii=False))
    print(f"[merge-recon-metrics] wrote {args.output}")


if __name__ == "__main__":
    main()
