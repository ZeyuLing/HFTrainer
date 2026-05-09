#!/usr/bin/env python3
"""Merge KIMODO shard directories for one task/setting."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def aggregate(per_sample: list[dict]) -> dict:
    out: dict[str, dict] = {}
    names = set()
    for sample in per_sample:
        names.update(k for k in sample if not k.startswith("_"))
    for name in sorted(names):
        vals = [
            float(sample[name])
            for sample in per_sample
            if isinstance(sample.get(name), (int, float))
        ]
        if vals:
            out[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "count": int(len(vals)),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--final-dir", required=True, type=Path)
    parser.add_argument("--task-key", required=True, help="Example: E8_D")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--expected", type=int, default=None)
    args = parser.parse_args()

    shard_dirs = sorted(args.shard_root.glob(f"shard_*/*/{args.task_key}"))
    if not shard_dirs:
        shard_dirs = sorted(args.shard_root.glob(f"shard_*/{args.task_key}"))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard dirs for {args.task_key} under {args.shard_root}")

    final_npz = args.final_dir / "npz"
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
    if args.expected is not None and len(per_sample) != args.expected:
        missing = sorted(set(range(args.expected)) - set(per_sample_by_idx))
        raise RuntimeError(
            f"{args.task_key}: expected {args.expected}, got {len(per_sample)}; "
            f"missing={missing[:30]}"
        )

    result = {
        "model": (template or {}).get("model", "KIMODO_uncond"),
        "task_id": args.task_id,
        "setting": args.setting,
        "retarget_method": (template or {}).get("retarget_method", "rotation_based"),
        "has_caption": bool((template or {}).get("has_caption", False)),
        "num_prompts": len(per_sample),
        "aggregated": aggregate(per_sample),
        "per_sample": per_sample,
    }
    args.final_dir.mkdir(parents=True, exist_ok=True)
    (args.final_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"{args.task_key}: merged {len(per_sample)} samples, copied {copied} npz -> {args.final_dir}")


if __name__ == "__main__":
    main()
