#!/usr/bin/env python3
"""Build keypose evaluation dataset from PeacekeeperElite before/after MB pairs.

Creates triplets: (src_motion, keyposes, target_motion) for keyframe pose guidance evaluation.

Output: output/keypose_eval/eval_data.json + per-case NPZ files.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

BEFORE_DIR = PROJECT_ROOT / "data" / "PeacekeeperElite_MB" / "PeacekeeperElite_part4_before_MB"
AFTER_DIR = PROJECT_ROOT / "data" / "PeacekeeperElite_MB" / "PeacekeeperElite_part4_after_MB"
OUTPUT_DIR = PROJECT_ROOT / "output" / "keypose_eval"


def extract_keyposes(before_poses, after_poses, min_gap=10, frames_per_keypose=30):
    """Extract keypose indices from target motion based on per-frame diff from source.

    Returns list of keypose frame indices (sorted).
    """
    T = before_poses.shape[0]
    # Per-frame pose diff (mean across all dims)
    per_frame_diff = np.abs(before_poses - after_poses).mean(axis=1)

    # Number of keyposes: at least 1, roughly 1 per 30 frames
    K = max(1, T // frames_per_keypose)
    K = min(K, T // (min_gap + 1))  # Can't have more keyposes than fit with min_gap
    K = max(1, K)

    # Greedy selection: pick top-K frames with min_gap constraint
    sorted_indices = np.argsort(-per_frame_diff)
    selected = []
    for idx in sorted_indices:
        idx = int(idx)
        # Skip first and last frame (they'll be anchors)
        if idx == 0 or idx == T - 1:
            continue
        # Check min gap with existing selections
        if all(abs(idx - s) >= min_gap for s in selected):
            selected.append(idx)
            if len(selected) >= K:
                break

    return sorted(selected)


def build_eval_dataset():
    """Build evaluation dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    before_files = set(os.listdir(BEFORE_DIR))
    after_files = set(os.listdir(AFTER_DIR))
    common = sorted(before_files & after_files)

    print(f"Common files: {len(common)}")

    eval_cases = []
    skipped = {"diff_len": 0, "too_short": 0, "too_long": 0, "no_diff": 0, "few_diff_frames": 0}

    for fname in common:
        b = np.load(str(BEFORE_DIR / fname), allow_pickle=True)
        a = np.load(str(AFTER_DIR / fname), allow_pickle=True)

        b_poses = np.array(b["poses"], dtype=np.float32)
        a_poses = np.array(a["poses"], dtype=np.float32)

        # Must have same length
        if b_poses.shape[0] != a_poses.shape[0]:
            skipped["diff_len"] += 1
            continue

        T = b_poses.shape[0]

        # Frame count filters
        if T < 30:
            skipped["too_short"] += 1
            continue
        if T > 360:
            skipped["too_long"] += 1
            continue

        # Check actual modification exists
        per_frame_diff = np.abs(b_poses[:, :66] - a_poses[:, :66]).mean(axis=1)  # Body joints only
        max_diff = per_frame_diff.max()
        if max_diff < 0.05:
            skipped["no_diff"] += 1
            continue

        # Need at least 3 frames with significant diff
        significant_frames = (per_frame_diff > 0.02).sum()
        if significant_frames < 3:
            skipped["few_diff_frames"] += 1
            continue

        # Extract keyposes
        keypose_indices = extract_keyposes(b_poses[:, :66], a_poses[:, :66])

        if len(keypose_indices) == 0:
            skipped["no_diff"] += 1
            continue

        # Compute stats
        keypose_diffs = [float(per_frame_diff[ki]) for ki in keypose_indices]

        case = {
            "filename": fname,
            "num_frames": T,
            "keypose_indices": keypose_indices,
            "keypose_diffs": keypose_diffs,
            "max_frame_diff": float(max_diff),
            "mean_frame_diff": float(per_frame_diff.mean()),
            "significant_frames": int(significant_frames),
            "before_path": str(BEFORE_DIR / fname),
            "after_path": str(AFTER_DIR / fname),
        }
        eval_cases.append(case)

    print(f"\nSkipped: {skipped}")
    print(f"Valid eval cases: {len(eval_cases)}")

    # Sort by max diff descending (most interesting first)
    eval_cases.sort(key=lambda x: -x["max_frame_diff"])

    # Save eval dataset manifest
    manifest = {
        "total": len(eval_cases),
        "skipped": skipped,
        "cases": eval_cases,
    }
    manifest_path = OUTPUT_DIR / "eval_data.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest: {manifest_path}")

    # Print summary
    if eval_cases:
        num_keyposes = [len(c["keypose_indices"]) for c in eval_cases]
        print(f"\nKeypose stats:")
        print(f"  Total cases: {len(eval_cases)}")
        print(f"  Keyposes per case: min={min(num_keyposes)}, max={max(num_keyposes)}, mean={np.mean(num_keyposes):.1f}")
        print(f"  Frame lengths: min={min(c['num_frames'] for c in eval_cases)}, max={max(c['num_frames'] for c in eval_cases)}")

    return eval_cases


if __name__ == "__main__":
    build_eval_dataset()
