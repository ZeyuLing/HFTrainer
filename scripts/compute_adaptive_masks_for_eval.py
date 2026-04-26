#!/usr/bin/env python3
"""Compute MoGenDIT adaptive masks for eval_repair.json samples.

Saves hierarchical masks to eval_results/m2m/T7/adaptive_masks/<motion_path>.npz
Each mask NPZ contains:
  - joint_mask: (T, 22) bool
  - trans_mask: (T,) bool

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/compute_adaptive_masks_for_eval.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data" / "hymotion_data"
# v2 eval datalist (was: data/eval/hymotion_m2m/eval_repair.json)
EVAL_DATALIST = PROJECT_ROOT / "data" / "eval" / "m2m_v2" / "eval_e9_repair.json"
# Shared mask cache consumed by tools/eval_m2m_v2_all_tasks.py
MASK_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "hymotion_m2m" / "adaptive_masks_mogendit"

MOTION_ROOTS = [
    DATA_ROOT / "3D" / "20251111" / "motions",
    DATA_ROOT,
]


def resolve_motion_path(motion_path):
    # v2 datalist stores paths with the "data/hymotion_data/" prefix already
    # (project-root-relative), so first try PROJECT_ROOT as the anchor.
    pr_relative = str(PROJECT_ROOT / motion_path)
    if os.path.isfile(pr_relative):
        return pr_relative
    # Legacy v1 datalist stored paths without the data/hymotion_data/ prefix;
    # fall back to the hymotion_data subroots.
    for root in MOTION_ROOTS:
        full = str(root / motion_path)
        if os.path.isfile(full):
            return full
    return None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mogendit-steps", type=int, default=10)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    with open(EVAL_DATALIST) as f:
        items = json.load(f)["data_list"]
    print(f"Loaded {len(items)} samples from eval_repair.json")

    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to existing files and check cached
    to_compute = []
    for it in items:
        mp = it.get("motion_path", "")
        full = resolve_motion_path(mp)
        if not full:
            continue
        # Strip optional project-relative prefix so mask layout stays consistent
        # with the legacy v1 cache (which was indexed by hymotion_data-relative path).
        mp_for_cache = mp
        if mp_for_cache.startswith("data/hymotion_data/"):
            mp_for_cache = mp_for_cache[len("data/hymotion_data/"):]
        out_path = MASK_OUTPUT_DIR / mp_for_cache
        if out_path.is_file() and not args.force:
            continue
        to_compute.append((mp_for_cache, full))

    print(f"{len(to_compute)} masks to compute ({len(items) - len(to_compute)} cached/missing)")
    if not to_compute:
        print("Nothing to do!")
        return

    print(f"Loading MoGenDIT on {args.device}...")
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline
    mogendit = MoGenDITRepairPipeline(model_name='MoreDiff-0.1B', device=args.device)
    print("Ready.\n")

    t_start = time.time()
    success = 0
    errors = 0

    for idx, (mp, full_path) in enumerate(to_compute):
        try:
            result = mogendit.compute_adaptive_mask(
                full_path,
                step=args.mogendit_steps,
                joint_threshold=0.15,
                trans_threshold=0.05,
                max_mask_ratio=0.15,
            )

            out_path = MASK_OUTPUT_DIR / mp
            os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
            np.savez_compressed(
                str(out_path),
                joint_mask=result["joint_mask"],
                trans_mask=result["trans_mask"],
            )
            success += 1

            if (idx + 1) % 50 == 0 or idx == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                jm = result["joint_mask"]
                mask_ratio = jm.sum() / max(jm.size, 1)
                print(f"  [{idx+1}/{len(to_compute)}] mask={mask_ratio:.1%} "
                      f"| {rate:.1f}/s | ETA {(len(to_compute)-idx-1)/rate:.0f}s")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [{idx+1}] ERROR {mp}: {e}")

    elapsed = time.time() - t_start
    print(f"\nDone! {success}/{len(to_compute)} in {elapsed:.1f}s, {errors} errors")
    print(f"Saved to: {MASK_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
