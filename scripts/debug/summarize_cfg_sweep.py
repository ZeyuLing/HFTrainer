#!/usr/bin/env python3
"""Summarize small CFG sweeps from eval_m2m_v2_all_tasks outputs."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any


def _metric_mean(aggregated: dict, name: str) -> float:
    value = aggregated.get(name, {})
    if isinstance(value, dict):
        return float(value.get("mean", math.nan))
    return math.nan


def _find_aggregated(obj: Any, model: str, task_key: str) -> dict | None:
    if not isinstance(obj, dict):
        return None
    if "aggregated" in obj and isinstance(obj["aggregated"], dict):
        return obj["aggregated"]
    try:
        return obj["tasks"][task_key]["models"][model]["aggregated"]
    except Exception:
        pass
    try:
        return obj[model]["tasks"][task_key]["aggregated"]
    except Exception:
        pass
    try:
        return obj[model][task_key]["aggregated"]
    except Exception:
        pass
    for value in obj.values():
        found = _find_aggregated(value, model, task_key)
        if found is not None:
            return found
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-key", default="E2_pre20")
    parser.add_argument("--label", required=True)
    parser.add_argument("--scales", nargs="+", default=["1.0", "2.0", "3.0", "5.0"])
    args = parser.parse_args()

    for scale in args.scales:
        if scale in {".", "direct"}:
            pattern = os.path.join(args.root, "eval_v2_*.json")
        else:
            pattern = os.path.join(args.root, f"scale_{scale}", "eval_v2_*.json")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"{args.label} scale={scale} MISSING")
            continue

        with open(paths[-1], "r", encoding="utf-8") as f:
            obj = json.load(f)

        aggregated = _find_aggregated(obj, args.model, args.task_key)
        if aggregated is None:
            print(
                f"{args.label} scale={scale} NO_AGGREGATED "
                f"keys={list(obj.keys())[:12]} path={paths[-1]}"
            )
            continue

        print(
            f"{args.label} scale={scale} "
            f"jitter={_metric_mean(aggregated, 'jitter_pos'):.3f} "
            f"boundary={_metric_mean(aggregated, 'boundary_accel_jump'):.3f} "
            f"mpjpe={_metric_mean(aggregated, 'mpjpe_masked'):.3f} "
            f"path={paths[-1]}"
        )


if __name__ == "__main__":
    main()
