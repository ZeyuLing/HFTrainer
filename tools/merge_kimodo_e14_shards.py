#!/usr/bin/env python3
"""Merge sharded KIMODO E14 outputs into the dashboard-visible run dirs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _aggregate(per_sample: list[dict]) -> dict:
    agg = {}
    metric_names = set()
    for sample in per_sample:
        metric_names.update(k for k in sample if not k.startswith("_"))
    for metric in sorted(metric_names):
        vals = [
            sample[metric]
            for sample in per_sample
            if isinstance(sample.get(metric), (int, float))
        ]
        if vals:
            agg[metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
    return agg


def merge_setting(shard_root: Path, final_root: Path, setting: str) -> None:
    task_key = f"E14_{setting}"
    shard_dirs = sorted(shard_root.glob(f"{task_key}_shard*/{task_key}"))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard dirs found for {task_key} under {shard_root}")

    final_dir = final_root / task_key / task_key
    final_npz = final_dir / "npz"
    final_npz.mkdir(parents=True, exist_ok=True)

    per_sample_by_idx: dict[int, dict] = {}
    template: dict | None = None
    copied = 0
    for shard_dir in shard_dirs:
        result_path = shard_dir / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing shard result: {result_path}")
        data = json.loads(result_path.read_text())
        template = template or data

        for npz_path in sorted((shard_dir / "npz").glob("*.npz")):
            shutil.copy2(npz_path, final_npz / npz_path.name)
            copied += 1

        for sample in data.get("per_sample", []):
            idx = sample.get("_sample_idx")
            if not isinstance(idx, int):
                continue
            sample = dict(sample)
            sample["_npz_path"] = str(final_npz / f"{idx:05d}.npz")
            per_sample_by_idx[idx] = sample

    per_sample = [per_sample_by_idx[i] for i in sorted(per_sample_by_idx)]
    if len(per_sample) != 100:
        missing = sorted(set(range(100)) - set(per_sample_by_idx))
        raise RuntimeError(
            f"{task_key}: expected 100 samples, got {len(per_sample)}; "
            f"missing={missing[:20]}"
        )

    result = {
        "model": template.get("model", "KIMODO_uncond") if template else "KIMODO_uncond",
        "task_id": "E14",
        "setting": setting,
        "retarget_method": "rotation_based",
        "has_caption": bool(template.get("has_caption", False)) if template else False,
        "num_prompts": len(per_sample),
        "aggregated": _aggregate(per_sample),
        "per_sample": per_sample,
    }
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"{task_key}: merged {len(per_sample)} samples, copied {copied} npz -> {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--final-root", required=True, type=Path)
    parser.add_argument("--settings", nargs="+", default=["M", "L"])
    args = parser.parse_args()

    for setting in args.settings:
        merge_setting(args.shard_root, args.final_root, setting)


if __name__ == "__main__":
    main()
