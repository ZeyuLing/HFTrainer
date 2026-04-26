#!/usr/bin/env python3
"""Compute Mean/Std for 201-dim motion representation (v2).

201-dim layout:
    [0:3]      translation (SMPL trans)
    [3:135]    22 joints × 6D rot6d (row-major)
    [135:201]  22 joints × 3D position (XZ relative to pelvis, Y absolute world)

Uses multiprocessing for speed (~8-16x faster than single-process).

Usage:
    # Local rotation + 201-dim position (default 16 workers)
    PYTHONPATH=. python3 tools/compute_201dim_stats.py --rotation_space local \
        --output_dir data/hymotion_m2m_data/_stats_201dim

    # Global rotation + 201-dim position
    PYTHONPATH=. python3 tools/compute_201dim_stats.py --rotation_space global \
        --output_dir data/hymotion_m2m_data/_stats_201dim_global_rot

    # Custom workers
    PYTHONPATH=. python3 tools/compute_201dim_stats.py --num_workers 32 \
        --output_dir data/hymotion_m2m_data/_stats_201dim
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import sys
import time
from functools import partial

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute Mean/Std for 201-dim motion representation'
    )
    parser.add_argument(
        '--anno',
        default='data/annotation/train_hymotion_400h_hq_20260403.json',
        help='Annotation JSON file path',
    )
    parser.add_argument(
        '--output_dir',
        default='data/hymotion_m2m_data/_stats_201dim',
        help='Output directory for Mean.npy and Std.npy',
    )
    parser.add_argument(
        '--data_dir',
        default='data/motionhub',
        help='Base data directory (for resolving relative paths)',
    )
    parser.add_argument(
        '--rotation_space',
        default='local',
        choices=['local', 'global'],
        help='Rotation space: local (SMPL frame) or global (world frame)',
    )
    parser.add_argument(
        '--bone_offsets_path',
        default='data/hymotion_m2m_data/bone_offsets_22.pt',
        help='Path to precomputed bone offsets (22, 3)',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='Number of worker processes',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _init_worker(bone_offsets_np: np.ndarray, rotation_space: str):
    """Initialize per-worker globals (called once per worker process)."""
    global _W_LOADER, _W_TO_GLOBAL, _W_BONE_OFFSETS

    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55
    from hftrainer.datasets.motion.motionhub.transforms.local_to_global import (
        LocalToGlobalRotation,
    )

    _W_LOADER = LoadSmplx55(
        key='motion',
        rot_type='rotation_6d',
        transl_type='abs',
        smpl_type='smpl_22',
        transl_aug_prob=0.0,
    )
    _W_TO_GLOBAL = LocalToGlobalRotation(key='motion') if rotation_space == 'global' else None
    _W_BONE_OFFSETS = torch.from_numpy(bone_offsets_np).float()


def _compute_position_channels(motion_135_local: np.ndarray, bone_offsets: torch.Tensor) -> np.ndarray:
    """Compute 66-dim position channels from LOCAL rotation motion via FK.

    Always uses local rotation for FK regardless of rotation_space setting,
    because FK requires local rotation to produce correct world positions.

    Returns:
        (T, 66) position channels in Scheme D (XZ rel pelvis, Y absolute world).
    """
    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

    motion_t = torch.from_numpy(motion_135_local).float()
    with torch.no_grad():
        world_pos, _, _, _ = motion135_to_fk(motion_t, bone_offsets)

    # Scheme D: XZ relative to pelvis, Y absolute world
    pelvis_world = world_pos[:, 0:1, :]  # (T, 1, 3)
    joint_pos_D = world_pos.clone()
    joint_pos_D[..., 0] -= pelvis_world[..., 0]  # X: relative to pelvis
    joint_pos_D[..., 2] -= pelvis_world[..., 2]  # Z: relative to pelvis
    # Y: keep absolute world height

    return joint_pos_D.reshape(-1, 66).numpy()


def _process_sample(args_tuple):
    """Process a single sample. Returns (n_frames, sum, sum_sq) or None on failure."""
    sample_id, smplx_path, fps, data_dir = args_tuple

    full_path = osp.join(data_dir, smplx_path)
    if not osp.exists(full_path):
        return None

    try:
        results = {
            'motion_path': full_path,
            'fps': fps,
        }

        results = _W_LOADER.transform(results)
        if results is None or 'motion' not in results:
            return None

        # Get local rotation motion for FK (position computation)
        motion_local = results['motion']
        if hasattr(motion_local, 'numpy'):
            motion_local = motion_local.numpy()

        # Optionally convert to global rotation (only affects rotation channels)
        if _W_TO_GLOBAL is not None:
            results_global = _W_TO_GLOBAL.transform(dict(results))
            motion_global = results_global['motion']
            if hasattr(motion_global, 'numpy'):
                motion_global = motion_global.numpy()
        else:
            motion_global = None

        # Collect all frames from all persons
        frames_list = []

        def _process_person(m_local, m_global):
            # FK uses LOCAL rotation to compute correct world positions
            pos_66 = _compute_position_channels(m_local, _W_BONE_OFFSETS)
            # Rotation channels come from global rotation if requested
            rot_source = m_global if m_global is not None else m_local
            return np.concatenate([rot_source, pos_66], axis=-1)

        if motion_local.ndim == 3:
            P, T, D = motion_local.shape
            for p in range(P):
                ml = motion_local[p]
                mg = motion_global[p] if motion_global is not None else None
                frames_list.append(_process_person(ml, mg))
        else:
            mg = motion_global if motion_global is not None else None
            frames_list.append(_process_person(motion_local, mg))

        # Return partial statistics (n, sum, sum_sq) for Chan's merge
        all_frames = np.concatenate(frames_list, axis=0).astype(np.float64)
        n = all_frames.shape[0]
        s = all_frames.sum(axis=0)
        sq = (all_frames ** 2).sum(axis=0)
        return (n, s, sq)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load bone offsets
    if osp.exists(args.bone_offsets_path):
        bone_offsets = torch.load(args.bone_offsets_path, map_location='cpu').float()
        bone_offsets_np = bone_offsets.numpy()
        print(f'Loaded bone offsets from {args.bone_offsets_path}')
    else:
        print(f'Bone offsets not found at {args.bone_offsets_path}')
        print('Run: PYTHONPATH=. python3 tools/precompute_bone_offsets.py first')
        sys.exit(1)

    # Load annotation
    print(f'Loading annotation from {args.anno}...')
    with open(args.anno) as f:
        anno = json.load(f)

    data_list = anno['data_list']
    num_samples = len(data_list)
    print(f'Found {num_samples} samples')
    print(f'Rotation space: {args.rotation_space}')
    print(f'Workers: {args.num_workers}')
    print(f'Output: {args.output_dir}/')

    # Prepare work items
    work_items = []
    for sample_id, sample_info in data_list.items():
        smplx_path = sample_info.get('smplx_path', '')
        fps = sample_info.get('fps', 30.0)
        work_items.append((sample_id, smplx_path, fps, args.data_dir))

    # Process with multiprocessing
    t0 = time.time()
    total_n = 0
    total_sum = np.zeros(201, dtype=np.float64)
    total_sum_sq = np.zeros(201, dtype=np.float64)
    success = 0
    failed = 0

    init_fn = partial(
        _init_worker,
        bone_offsets_np=bone_offsets_np,
        rotation_space=args.rotation_space,
    )

    # Use imap_unordered for streaming results with progress reporting
    with mp.Pool(
        processes=args.num_workers,
        initializer=init_fn,
    ) as pool:
        for idx, result in enumerate(
            pool.imap_unordered(_process_sample, work_items, chunksize=64)
        ):
            if result is None:
                failed += 1
            else:
                n, s, sq = result
                total_n += n
                total_sum += s
                total_sum_sq += sq
                success += 1

            if (idx + 1) % 10000 == 0 or idx == num_samples - 1:
                elapsed = time.time() - t0
                rate = (idx + 1) / max(elapsed, 1e-3)
                print(
                    f'  [{idx+1}/{num_samples}] success={success} failed={failed} '
                    f'frames={total_n:,} speed={rate:.1f} samples/s'
                )

    elapsed = time.time() - t0
    print(f'\nDone. success={success} failed={failed} total_frames={total_n:,} '
          f'time={elapsed:.1f}s')

    if total_n == 0:
        print('ERROR: No valid samples found. Exiting.')
        sys.exit(1)

    # Compute final mean and std from aggregated statistics
    mean = (total_sum / total_n).astype(np.float32)
    variance = (total_sum_sq / total_n) - (total_sum / total_n) ** 2
    variance = np.maximum(variance, 0.0)  # numerical safety
    std = np.sqrt(variance).astype(np.float32)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    mean_path = osp.join(args.output_dir, 'Mean.npy')
    std_path = osp.join(args.output_dir, 'Std.npy')

    np.save(mean_path, mean)
    np.save(std_path, std)

    print(f'\nSaved to {args.output_dir}/')
    print(f'  Mean: shape={mean.shape}, range=[{mean.min():.6f}, {mean.max():.6f}]')
    print(f'  Std:  shape={std.shape}, range=[{std.min():.6f}, {std.max():.6f}]')

    # Detailed channel analysis
    print(f'\n=== Channel Analysis ===')
    print(f'  Translation (dims 0:3):')
    print(f'    Mean: {mean[0:3]}')
    print(f'    Std:  {std[0:3]}')

    print(f'  Root rot6d (dims 3:9):')
    print(f'    Mean: {mean[3:9]}')
    print(f'    Std:  {std[3:9]}')

    print(f'\n  Position channels (dims 135:201):')
    joint_names = [
        "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
        "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
        "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    ]
    for j in range(22):
        s = 135 + j * 3
        m = mean[s:s+3]
        st = std[s:s+3]
        print(f'    {joint_names[j]:12s}: mean=[{m[0]:7.4f}, {m[1]:7.4f}, {m[2]:7.4f}]  '
              f'std=[{st[0]:7.4f}, {st[1]:7.4f}, {st[2]:7.4f}]')

    # Validation checks
    print(f'\n=== Validation ===')
    pelvis_xz_mean = [mean[135], mean[137]]
    print(f'  Pelvis XZ mean: [{pelvis_xz_mean[0]:.6f}, {pelvis_xz_mean[1]:.6f}] (should be ~0)')
    print(f'  Pelvis Y mean: {mean[136]:.4f} (should be ~1.0, world pelvis height)')

    l_ankle_y = mean[135 + 7*3 + 1]
    r_ankle_y = mean[135 + 8*3 + 1]
    print(f'  L_Ankle Y mean: {l_ankle_y:.4f} (should be small, near ground)')
    print(f'  R_Ankle Y mean: {r_ankle_y:.4f} (should be small, near ground)')

    l_ankle_y_std = std[135 + 7*3 + 1]
    r_ankle_y_std = std[135 + 8*3 + 1]
    print(f'  L_Ankle Y std: {l_ankle_y_std:.4f} (small = good, absolute Y is stable)')
    print(f'  R_Ankle Y std: {r_ankle_y_std:.4f}')

    # Std < 1e-3 check (will be clamped to 1.0 in bundle)
    small_std = np.where(std < 1e-3)[0]
    if len(small_std) > 0:
        print(f'\n  ⚠️  {len(small_std)} dims with std < 1e-3 (will be clamped to 1.0):')
        for d in small_std:
            print(f'    dim {d}: mean={mean[d]:.6f}, std={std[d]:.8f}')
    else:
        print(f'\n  ✅ No dims with std < 1e-3')

    # Compare with existing 135-dim stats
    ref_dir = 'data/hymotion_m2m_data/_stats'
    if args.rotation_space == 'global':
        ref_dir = 'data/hymotion_m2m_data/_stats_global_rot'
    if osp.exists(osp.join(ref_dir, 'Mean.npy')):
        ref_mean = np.load(osp.join(ref_dir, 'Mean.npy')).astype(np.float32)
        ref_std = np.load(osp.join(ref_dir, 'Std.npy')).astype(np.float32)
        print(f'\n  Comparison with existing 135-dim stats ({ref_dir}):')
        # Note: may differ due to different training data (HQ vs unfiltered)
        trans_mean_diff = np.abs(mean[:3] - ref_mean[:3]).max()
        trans_std_diff = np.abs(std[:3] - ref_std[:3]).max()
        rot_mean_diff = np.abs(mean[3:135] - ref_mean[3:135]).max()
        rot_std_diff = np.abs(std[3:135] - ref_std[3:135]).max()
        print(f'    Translation mean max_diff: {trans_mean_diff:.6f}')
        print(f'    Translation std max_diff:  {trans_std_diff:.6f}')
        print(f'    Rotation mean max_diff:    {rot_mean_diff:.6f}')
        print(f'    Rotation std max_diff:     {rot_std_diff:.6f}')
        if trans_mean_diff > 0.05 or rot_mean_diff > 0.05:
            print(f'    ⚠️  Difference > 0.05 detected — expected if using different data subset.')
        else:
            print(f'    ✅ First 135 dims consistent with existing stats.')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
