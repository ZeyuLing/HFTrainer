#!/usr/bin/env python3
"""Compute and save MoGenDiT adaptive masks for existing eval results.

This is a lightweight script that only runs MoGenDiT light-denoise (10 steps)
to compute adaptive masks and saves them as NPZ files. No M2M repair needed.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/compute_adaptive_masks.py \
        --eval-dir output/m2m_repair_eval_latest_ckpt \
        --data-root data/hymotion_data \
        --mogendit-steps 10
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import seaborn  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "seaborn"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args():
    p = argparse.ArgumentParser(description="Compute adaptive masks for eval results")
    p.add_argument("--eval-dir", type=str, required=True,
                    help="Path to eval output dir (e.g. output/m2m_repair_eval_latest_ckpt)")
    p.add_argument("--data-root", type=str, default="data/hymotion_data")
    p.add_argument("--mogendit-steps", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    data_root = Path(args.data_root)

    # Collect all unique motion paths from all configs' repair_stats.json
    all_paths = set()
    for stats_file in eval_dir.glob("*/repair_stats.json"):
        with open(stats_file) as f:
            stats = json.load(f)
        for detail in stats.get("details", []):
            all_paths.add(detail["path"])

    print(f"Found {len(all_paths)} unique motion paths from {eval_dir}")

    # Filter to existing files
    valid_paths = [p for p in sorted(all_paths) if (data_root / p).is_file()]
    print(f"{len(valid_paths)} files exist on disk")

    # Check which already have masks
    mask_dir = eval_dir / "adaptive_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    to_compute = [p for p in valid_paths if not (mask_dir / p).with_suffix(".npz").is_file()
                  and not (mask_dir / p).is_file()]
    # Handle case where path already ends in .npz
    to_compute2 = []
    for p in valid_paths:
        out_path = mask_dir / p
        if out_path.is_file():
            continue
        to_compute2.append(p)
    to_compute = to_compute2
    print(f"{len(to_compute)} masks to compute ({len(valid_paths) - len(to_compute)} already cached)")

    if not to_compute:
        print("Nothing to do!")
        return

    # Build MoGenDiT
    print(f"\nLoading MoGenDiT on {args.device}...")
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline
    mogendit = MoGenDITRepairPipeline(model_name='MoreDiff-0.1B', device=args.device)
    print("MoGenDiT ready.\n")

    t_start = time.time()
    success = 0
    errors = 0

    for idx, rel_path in enumerate(to_compute):
        npz_path = str(data_root / rel_path)
        try:
            result = mogendit.compute_adaptive_mask(
                npz_path,
                step=args.mogendit_steps,
                joint_threshold=0.15,
                trans_threshold=0.05,
                max_mask_ratio=0.15,
            )

            # Save
            out_path = mask_dir / rel_path
            os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
            np.savez_compressed(
                str(out_path),
                joint_mask=result["joint_mask"],
                trans_mask=result["trans_mask"],
            )
            success += 1

            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (len(to_compute) - idx - 1) / rate if rate > 0 else 0
                jm = result["joint_mask"]
                mask_ratio = jm.sum() / max(jm.size, 1)
                print(f"  [{idx+1}/{len(to_compute)}] {Path(rel_path).stem} "
                      f"mask={mask_ratio:.1%} | {rate:.1f} samples/s | ETA {eta:.0f}s")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [{idx+1}] ERROR {rel_path}: {e}")

    elapsed = time.time() - t_start
    print(f"\nDone! {success}/{len(to_compute)} masks computed in {elapsed:.1f}s")
    print(f"Saved to: {mask_dir}")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
