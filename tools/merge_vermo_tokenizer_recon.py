#!/usr/bin/env python3
"""Merge sharded VerMo tokenizer reconstruction metrics."""

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
    with open(path, "r", encoding="utf-8") as f:
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
    parser.add_argument("--inputs", nargs="+", required=True, help="Metric JSON paths or glob patterns.")
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
    code_usage: list[set[int]] = []
    selected_samples = 0
    selected_before_shard = 0
    skipped_person_mismatch = 0
    skipped_duration = 0
    skipped_id_filter = 0
    codebook_size = int(first["summary"].get("codebook_size") or 0)
    max_duration = first.get("max_duration", 0.0)
    id_list = first.get("id_list", "")

    for payload in payloads:
        per_case.extend(payload.get("per_case", []))
        failures.extend(payload.get("failures", []))
        subsets.update(payload.get("subsets", {}))
        selected_samples += int(payload.get("selected_samples", 0))
        selected_before_shard = max(
            selected_before_shard,
            int(payload.get("selected_samples_before_shard", payload.get("selected_samples", 0))),
        )
        skipped_person_mismatch = max(skipped_person_mismatch, int(payload.get("skipped_person_mismatch", 0)))
        skipped_duration = max(skipped_duration, int(payload.get("skipped_duration", 0)))
        skipped_id_filter = max(skipped_id_filter, int(payload.get("skipped_id_filter", 0)))
        if payload.get("max_duration", 0.0) != max_duration:
            raise ValueError("Input max_duration values do not match")
        if payload.get("id_list", "") != id_list:
            raise ValueError("Input id_list values do not match")
        if int(payload["summary"].get("codebook_size") or 0) != codebook_size:
            raise ValueError("Input codebook sizes do not match")
        usage_values = payload.get("code_usage_values_per_quantizer", [])
        while len(code_usage) < len(usage_values):
            code_usage.append(set())
        for idx, values in enumerate(usage_values):
            code_usage[idx].update(int(x) for x in values)

    metric_keys = [
        "mpjpe_mm",
        "raw_mpjpe_mm",
        "root0_mpjpe_mm",
        "rootframe_mpjpe_mm",
        "root_mpjpe_mm",
        "pa_mpjpe_mm",
        "mpjre_deg",
    ]
    metric_values = {
        key: [float(item[key]) for item in per_case if item.get(key) is not None]
        for key in metric_keys
    }
    frame_deltas = [int(item["frame_delta"]) for item in per_case]
    util_per_quantizer = [
        (len(items) / codebook_size * 100.0) if codebook_size else None
        for items in code_usage
    ]
    cb_util = None
    if util_per_quantizer:
        cb_util = float(np.mean([x for x in util_per_quantizer if x is not None]))

    merged = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "merged_from": paths,
        "config": first.get("config"),
        "tokenizer_path": first.get("tokenizer_path"),
        "anno_file": first.get("anno_file"),
        "id_list": id_list,
        "data_dir": first.get("data_dir"),
        "num_person": first.get("num_person"),
        "rot6d_convention": first.get("rot6d_convention"),
        "max_duration": max_duration,
        "selected_samples": selected_samples,
        "selected_samples_before_shard": selected_before_shard,
        "skipped_person_mismatch": skipped_person_mismatch,
        "skipped_duration": skipped_duration,
        "skipped_id_filter": skipped_id_filter,
        "subsets": dict(subsets),
        "summary": {
            **{key: summarize(metric_values[key]) for key in metric_keys},
            "cb_util_percent": cb_util,
            "cb_util_percent_per_quantizer": util_per_quantizer,
            "codebook_size": codebook_size,
            "frame_delta_abs_mean": float(np.mean(np.abs(frame_deltas))) if frame_deltas else None,
            "frame_delta_abs_max": int(np.max(np.abs(frame_deltas))) if frame_deltas else None,
            "num_failures": len(failures),
        },
        "code_usage_values_per_quantizer": [sorted(items) for items in code_usage],
        "failures": failures,
        "per_case": sorted(per_case, key=lambda item: item["key"]),
    }
    write_json(args.output, merged)
    print(json.dumps(merged["summary"], indent=2, ensure_ascii=False))
    print(f"[merge-vermo-tokenizer-recon] wrote {args.output}")


if __name__ == "__main__":
    main()
