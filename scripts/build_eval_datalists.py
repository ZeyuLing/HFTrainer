#!/usr/bin/env python3
"""Build missing evaluation datalists for M2M comprehensive eval.

Generates:
  1. eval_repair.json     — 1,000 samples from low_quality.json for T7 Repair
  2. eval_trajectory.json — 500 samples derived from eval_transition.json for T8 Trajectory

Usage:
    python3 scripts/build_eval_datalists.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "eval" / "hymotion_m2m"
DATA_ROOT = PROJECT_ROOT / "data" / "hymotion_data"
LOW_QUALITY_PATH = PROJECT_ROOT / "data" / "hymotion_m2m_refine_data" / "data_quality_list" / "low_quality.json"


def build_eval_repair(seed=42, max_samples=1000):
    """Build eval_repair.json from low_quality.json."""
    print(f"Building eval_repair.json from {LOW_QUALITY_PATH}")

    with open(LOW_QUALITY_PATH) as f:
        lq = json.load(f)

    data_dir = lq.get("data_dir", "data/hymotion_data")
    items = lq["items"]
    print(f"  Total low_quality items: {len(items)}")

    # Filter: file must exist and have >= 30 frames
    valid = []
    for item in items:
        full_path = str(PROJECT_ROOT / data_dir / item["path"])
        if os.path.isfile(full_path):
            valid.append(item)

    print(f"  Files found on disk: {len(valid)}")

    # Sample
    rng = np.random.RandomState(seed)
    rng.shuffle(valid)
    sampled = valid[:max_samples]

    # Enrich with frame count
    data_list = []
    for item in sampled:
        full_path = str(PROJECT_ROOT / data_dir / item["path"])
        try:
            npz = np.load(full_path, allow_pickle=True)
            poses = npz.get("poses", npz.get("body_pose"))
            if poses is None:
                continue
            num_frames = poses.shape[0]
            if num_frames < 30:
                continue
            fps = int(npz.get("mocap_framerate", 30))
        except Exception:
            continue

        data_list.append({
            "motion_path": item["path"],
            "num_frames": num_frames,
            "fps": fps,
            "duration_sec": round(num_frames / fps, 2),
            "failed_checks": item.get("failed_checks", []),
            "reasons": item.get("reasons", []),
            "all_checks": item.get("all_checks", []),
        })

    output = {
        "meta": {
            "description": "T7 Motion Repair - low quality samples from data_quality_list",
            "total_items": len(data_list),
            "source": str(LOW_QUALITY_PATH),
            "seed": seed,
        },
        "data_list": data_list,
    }

    out_path = EVAL_DIR / "eval_repair.json"
    os.makedirs(str(EVAL_DIR), exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data_list)} items to {out_path}")
    return out_path


def build_eval_trajectory(seed=42):
    """Build eval_trajectory.json derived from eval_transition.json.

    Same samples as eval_transition, but annotated for T8 trajectory task.
    The actual root translation is extracted at eval time from the NPZ.
    """
    src_path = EVAL_DIR / "eval_transition.json"
    print(f"Building eval_trajectory.json from {src_path}")

    with open(str(src_path)) as f:
        src = json.load(f)

    data_list = src["data_list"]

    output = {
        "meta": {
            "description": "T8 Trajectory-Based Generation - derived from eval_transition.json. "
                           "Root translation extracted at eval time from NPZ files.",
            "total_items": len(data_list),
            "with_caption": sum(1 for d in data_list if d.get("has_caption")),
            "without_caption": sum(1 for d in data_list if not d.get("has_caption")),
            "source": "eval_transition.json",
        },
        "data_list": data_list,
    }

    out_path = EVAL_DIR / "eval_trajectory.json"
    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data_list)} items to {out_path}")
    return out_path


def main():
    build_eval_repair()
    build_eval_trajectory()
    print("\nDone! All datalists built.")


if __name__ == "__main__":
    main()
