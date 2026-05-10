#!/usr/bin/env python3
"""Compute 198-dim stats by slicing 201-dim stats (remove pelvis position).

198-dim layout:
    [0:3]      translation (SMPL trans)
    [3:135]    22 joints x 6D rot6d (row-major)
    [135:198]  21 joints x 3D position (XZ relative to pelvis, Y absolute world)
               Joint 1 (L_Hip) through Joint 21 (R_Wrist), pelvis excluded.

The 201-dim stats include all 22 joints (including pelvis) in the position
channels (dims 135:201). Pelvis position in Scheme D is [0, pelvis_y, 0]
where XZ are always zero (relative to self) and Y is redundant with
translation. We remove dims 135:138 (pelvis position) to get 198 dims.

Usage:
    PYTHONPATH=. python3 tools/compute_198dim_stats.py

    # Verify only (no write)
    PYTHONPATH=. python3 tools/compute_198dim_stats.py --verify-only
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute 198-dim stats from 201-dim stats'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing 198-dim stats, do not write.',
    )
    return parser.parse_args()


def slice_201_to_198(arr_201: np.ndarray) -> np.ndarray:
    """Remove pelvis position (dims 135:138) from 201-dim array.

    201-dim: [trans(3), rot6d(132), pos_22joints(66)]
    198-dim: [trans(3), rot6d(132), pos_21joints(63)]

    Pelvis position is the first 3 dims of the position block (dims 135:138).
    """
    assert arr_201.shape == (201,), f"Expected (201,), got {arr_201.shape}"
    # Keep dims 0:135 (trans + rot6d) and dims 138:201 (joints 1-21 position)
    return np.concatenate([arr_201[:135], arr_201[138:]])


def process_stats(src_dir: str, dst_dir: str, label: str, verify_only: bool) -> bool:
    """Process one pair of stats directories."""
    mean_path = osp.join(src_dir, 'Mean.npy')
    std_path = osp.join(src_dir, 'Std.npy')

    if not osp.exists(mean_path) or not osp.exists(std_path):
        print(f'  ❌ {label}: source stats not found at {src_dir}')
        return False

    mean_201 = np.load(mean_path).astype(np.float32)
    std_201 = np.load(std_path).astype(np.float32)

    print(f'\n=== {label} ===')
    print(f'  Source: {src_dir}')
    print(f'  Mean 201-dim: shape={mean_201.shape}, range=[{mean_201.min():.6f}, {mean_201.max():.6f}]')
    print(f'  Std 201-dim:  shape={std_201.shape}, range=[{std_201.min():.6f}, {std_201.max():.6f}]')

    # Show pelvis position stats (dims 135:138) that will be removed
    pelvis_mean = mean_201[135:138]
    pelvis_std = std_201[135:138]
    print(f'  Pelvis position (to be removed):')
    print(f'    Mean: [{pelvis_mean[0]:.6f}, {pelvis_mean[1]:.6f}, {pelvis_mean[2]:.6f}]')
    print(f'    Std:  [{pelvis_std[0]:.6f}, {pelvis_std[1]:.6f}, {pelvis_std[2]:.6f}]')
    print(f'    (XZ should be ~0, Y should be ~pelvis height)')

    # Slice to 198-dim
    mean_198 = slice_201_to_198(mean_201)
    std_198 = slice_201_to_198(std_201)

    assert mean_198.shape == (198,), f"Expected (198,), got {mean_198.shape}"
    assert std_198.shape == (198,), f"Expected (198,), got {std_198.shape}"

    # Verify: first 135 dims unchanged
    assert np.allclose(mean_198[:135], mean_201[:135]), "Translation+rotation dims should be unchanged"
    assert np.allclose(std_198[:135], std_201[:135]), "Translation+rotation dims should be unchanged"

    # Verify: position dims 135:198 = original dims 138:201 (skip pelvis)
    assert np.allclose(mean_198[135:], mean_201[138:]), "Position dims should match (skip pelvis)"
    assert np.allclose(std_198[135:], std_201[138:]), "Position dims should match (skip pelvis)"

    print(f'  198-dim: shape={mean_198.shape}, range=[{mean_198.min():.6f}, {mean_198.max():.6f}]')

    # Channel analysis for 198-dim
    print(f'  Position channels (dims 135:198, 21 joints):')
    joint_names = [
        "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
        "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
        "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    ]
    for j in range(21):
        s = 135 + j * 3
        m = mean_198[s:s + 3]
        st = std_198[s:s + 3]
        print(f'    {joint_names[j]:12s}: mean=[{m[0]:7.4f}, {m[1]:7.4f}, {m[2]:7.4f}]  '
              f'std=[{st[0]:7.4f}, {st[1]:7.4f}, {st[2]:7.4f}]')

    if verify_only:
        # Check if existing 198-dim stats match
        dst_mean_path = osp.join(dst_dir, 'Mean.npy')
        dst_std_path = osp.join(dst_dir, 'Std.npy')
        if osp.exists(dst_mean_path) and osp.exists(dst_std_path):
            existing_mean = np.load(dst_mean_path).astype(np.float32)
            existing_std = np.load(dst_std_path).astype(np.float32)
            if np.allclose(existing_mean, mean_198) and np.allclose(existing_std, std_198):
                print(f'  ✅ Existing 198-dim stats match')
            else:
                print(f'  ❌ Existing 198-dim stats DO NOT match!')
                return False
        else:
            print(f'  ⚠️  No existing 198-dim stats to verify')
        return True

    # Save
    os.makedirs(dst_dir, exist_ok=True)
    np.save(osp.join(dst_dir, 'Mean.npy'), mean_198)
    np.save(osp.join(dst_dir, 'Std.npy'), std_198)
    print(f'  ✅ Saved to {dst_dir}/')

    return True


def main():
    args = parse_args()

    base_dir = 'data/hymotion_m2m_data'

    pairs = [
        (
            osp.join(base_dir, '_stats_201dim'),
            osp.join(base_dir, '_stats_198dim'),
            'Local rotation',
        ),
        (
            osp.join(base_dir, '_stats_201dim_global_rot'),
            osp.join(base_dir, '_stats_198dim_global_rot'),
            'Global rotation',
        ),
    ]

    all_ok = True
    for src_dir, dst_dir, label in pairs:
        ok = process_stats(src_dir, dst_dir, label, args.verify_only)
        if not ok:
            all_ok = False

    if all_ok:
        print('\n✅ All stats processed successfully.')
    else:
        print('\n❌ Some stats failed.')
        sys.exit(1)


if __name__ == '__main__':
    main()
