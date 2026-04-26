#!/usr/bin/env python3
"""Precompute MoGenDIT adaptive masks for T7 eval datalist."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data" / "hymotion_data"
MOTION_ROOTS = [
    DATA_ROOT / "3D" / "20251111" / "motions",
    DATA_ROOT,
]
EVAL_DIR = PROJECT_ROOT / "data" / "eval" / "hymotion_m2m"
MASK_DIR = EVAL_DIR / "adaptive_masks_mogendit"


def resolve_motion_path(mp):
    for root in MOTION_ROOTS:
        candidate = root / mp
        if candidate.is_file():
            return str(candidate)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datalist", default="eval_repair_focused.json")
    args = parser.parse_args()

    with open(str(EVAL_DIR / args.datalist)) as f:
        data = json.load(f)

    items = data["data_list"]

    # Find missing masks
    missing = []
    for it in items:
        mp = it.get("motion_path", it.get("path", ""))
        mask_path = MASK_DIR / mp
        full_path = resolve_motion_path(mp)
        if full_path and not mask_path.is_file():
            missing.append((mp, full_path, str(mask_path)))

    print(f"Total items: {len(items)}, missing masks: {len(missing)}")
    if not missing:
        print("All masks already computed.")
        return

    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline
    pipeline = MoGenDITRepairPipeline(device=args.device)

    t0 = time.time()
    for i, (mp, full_path, mask_path) in enumerate(missing):
        try:
            result = pipeline.compute_adaptive_mask(full_path)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            np.savez_compressed(
                mask_path,
                joint_mask=result["joint_mask"],
                trans_mask=result["trans_mask"],
            )
        except Exception as e:
            print(f"[{i+1}/{len(missing)}] FAILED {mp}: {e}")
            continue

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(missing) - i - 1) / rate
            print(f"[{i+1}/{len(missing)}] {rate:.1f} samples/s, ETA {eta:.0f}s")

    print(f"Done. Computed {len(missing)} masks in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
