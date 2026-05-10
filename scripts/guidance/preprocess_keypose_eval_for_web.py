#!/usr/bin/env python3
"""Pre-process eval_keyframe_pose results: convert 135-dim rot6d NPZ to SMPL axis-angle NPZ.

Run this on a GPU machine (with torch), output goes to output/eval_keyframe_pose/web_data/.
The web app can then serve these without needing torch.

Usage:
    python3 scripts/preprocess_keypose_eval_for_web.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    rotation_6d_to_axis_angle,
)

EVAL_ROOT = PROJECT_ROOT / "output" / "eval_keyframe_pose"
WEB_DATA = EVAL_ROOT / "web_data"


def rot6d_135_to_smpl(motion_135):
    """Convert (T, 135) rot6d to (T, 66) axis-angle + (T, 3) trans."""
    motion = np.asarray(motion_135, dtype=np.float32)
    T = motion.shape[0]
    transl = motion[:, :3]

    rot6d = motion[:, 3:135].reshape(T * 22, 6)
    rot6d_colmajor = rot6d[:, [0, 2, 4, 1, 3, 5]]
    aa = rotation_6d_to_axis_angle(rot6d_colmajor)
    aa = np.asarray(aa, dtype=np.float32).reshape(T, 66)

    return aa, transl


def process_all():
    """Convert all eval NPZ files to web-friendly format."""
    WEB_DATA.mkdir(parents=True, exist_ok=True)

    total_converted = 0
    for rot_space in ["local_rot", "global_rot"]:
        rot_dir = EVAL_ROOT / rot_space
        if not rot_dir.is_dir():
            continue

        # Copy summary
        summary_path = rot_dir / "eval_summary.json"
        if summary_path.is_file():
            import shutil
            dest = WEB_DATA / rot_space
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(summary_path), str(dest / "eval_summary.json"))

        report_path = rot_dir / "REPORT.md"
        if report_path.is_file():
            import shutil
            dest = WEB_DATA / rot_space
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(report_path), str(dest / "REPORT.md"))

        for variant_dir in sorted(rot_dir.iterdir()):
            if not variant_dir.is_dir():
                continue

            out_dir = WEB_DATA / rot_space / variant_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for npz_path in sorted(variant_dir.glob("*.npz")):
                out_path = out_dir / npz_path.name
                if out_path.exists():
                    continue

                try:
                    data = np.load(str(npz_path), allow_pickle=True)
                    output_motion = data["output_motion"]
                    gt_motion = data["gt_motion"]
                    keyframe_idx = int(data["keyframe_idx"])
                    src_mask = data["src_mask"]
                    target_pose = data["target_pose"]

                    # Convert output motion
                    out_aa, out_trans = rot6d_135_to_smpl(output_motion)

                    # Convert GT motion
                    gt_aa, gt_trans = rot6d_135_to_smpl(gt_motion)

                    # Mask summary per frame
                    mask_per_frame = src_mask.mean(axis=1).astype(np.float32)

                    np.savez_compressed(
                        str(out_path),
                        output_poses=out_aa,
                        output_trans=out_trans,
                        gt_poses=gt_aa,
                        gt_trans=gt_trans,
                        keyframe_idx=keyframe_idx,
                        mask_per_frame=mask_per_frame,
                        num_frames=output_motion.shape[0],
                    )
                    total_converted += 1

                except Exception as e:
                    print(f"Error converting {npz_path}: {e}")
                    continue

            print(f"  {rot_space}/{variant_dir.name}: converted")

    print(f"\nTotal converted: {total_converted}")
    print(f"Output: {WEB_DATA}")


if __name__ == "__main__":
    process_all()
