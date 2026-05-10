#!/usr/bin/env python3
"""Compute Mean/Std for global rotation representation.

Iterates over training data, applies LoadSmplx55 + LocalToGlobalRotation,
and accumulates per-dimension statistics using Welford's online algorithm.

Output: data/hymotion_m2m_data/_stats_global_rot/{Mean.npy, Std.npy}

Usage:
    python3 tools/compute_global_rot_stats.py
    python3 tools/compute_global_rot_stats.py --anno data/annotation/train_hymotion_400h.json
    python3 tools/compute_global_rot_stats.py --output_dir data/hymotion_m2m_data/_stats_global_rot
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
    LoadSmplx55,
)
from hftrainer.datasets.motion.motionhub.transforms.local_to_global import (
    LocalToGlobalRotation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute Mean/Std for global rotation motion representation'
    )
    parser.add_argument(
        '--anno',
        default='data/annotation/train_hymotion_400h.json',
        help='Annotation JSON file path',
    )
    parser.add_argument(
        '--output_dir',
        default='data/hymotion_m2m_data/_stats_global_rot',
        help='Output directory for Mean.npy and Std.npy',
    )
    parser.add_argument(
        '--data_dir',
        default='data/motionhub',
        help='Base data directory (for resolving relative paths)',
    )
    return parser.parse_args()


class WelfordAccumulator:
    """Welford online algorithm for computing mean and variance.

    Numerically stable for large datasets.
    """

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray):
        """Update with a batch of samples. x: (N, dim)."""
        for i in range(x.shape[0]):
            self.n += 1
            delta = x[i].astype(np.float64) - self.mean
            self.mean += delta / self.n
            delta2 = x[i].astype(np.float64) - self.mean
            self.M2 += delta * delta2

    def update_batch(self, x: np.ndarray):
        """Batch update using Chan's parallel algorithm for better speed."""
        batch_n = x.shape[0]
        if batch_n == 0:
            return
        x64 = x.astype(np.float64)
        batch_mean = x64.mean(axis=0)
        batch_var = x64.var(axis=0) * batch_n  # sum of squared deviations

        delta = batch_mean - self.mean
        new_n = self.n + batch_n
        self.M2 += batch_var + delta ** 2 * self.n * batch_n / max(new_n, 1)
        self.mean = (self.n * self.mean + batch_n * batch_mean) / max(new_n, 1)
        self.n = new_n

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self.M2 / self.n).astype(np.float32)

    @property
    def mean_f32(self) -> np.ndarray:
        return self.mean.astype(np.float32)


def main():
    args = parse_args()

    # Load annotation
    print(f'Loading annotation from {args.anno}...')
    with open(args.anno) as f:
        anno = json.load(f)

    data_list = anno['data_list']
    num_samples = len(data_list)
    print(f'Found {num_samples} samples')

    # Build transforms
    loader = LoadSmplx55(
        key='motion',
        rot_type='rotation_6d',
        transl_type='abs',
        smpl_type='smpl_22',
        transl_aug_prob=0.0,  # No augmentation for statistics
    )
    to_global = LocalToGlobalRotation(key='motion')

    # Accumulator for 135-dim motion
    acc = WelfordAccumulator(135)

    t0 = time.time()
    success = 0
    failed = 0

    for idx, (sample_id, sample_info) in enumerate(data_list.items()):
        smplx_path = sample_info.get('smplx_path', '')

        # Resolve relative path (same as MotionhubMultiTaskMultiAgentDataset)
        full_path = osp.join(args.data_dir, smplx_path)

        if not osp.exists(full_path):
            failed += 1
            if failed <= 5:
                print(f'  [WARN] File not found: {full_path}')
            continue

        try:
            # Build results dict as expected by transforms
            results = {
                'motion_path': full_path,
                'fps': sample_info.get('fps', 30.0),
            }

            # LoadSmplx55 expects the path in motion_path
            results = loader.transform(results)
            if results is None or 'motion' not in results:
                failed += 1
                continue

            # Convert to global rotation
            results = to_global.transform(results)

            motion = results['motion']  # (T, 135) or (P, T, 135), Tensor or ndarray
            # Ensure numpy for accumulator
            if hasattr(motion, 'numpy'):
                motion = motion.numpy()
            if motion.ndim == 3:
                # Multi-person: flatten
                motion = motion.reshape(-1, 135)

            # Accumulate
            acc.update_batch(motion)
            success += 1

        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f'  [ERROR] {sample_id}: {e}')
            continue

        if (idx + 1) % 10000 == 0 or idx == num_samples - 1:
            elapsed = time.time() - t0
            fps_rate = (idx + 1) / max(elapsed, 1e-3)
            print(
                f'  [{idx+1}/{num_samples}] success={success} failed={failed} '
                f'frames={acc.n} speed={fps_rate:.1f} samples/s'
            )

    elapsed = time.time() - t0
    print(f'\nDone. success={success} failed={failed} total_frames={acc.n} '
          f'time={elapsed:.1f}s')

    if acc.n == 0:
        print('ERROR: No valid samples found. Exiting.')
        sys.exit(1)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    mean_path = osp.join(args.output_dir, 'Mean.npy')
    std_path = osp.join(args.output_dir, 'Std.npy')

    mean = acc.mean_f32
    std = acc.std

    np.save(mean_path, mean)
    np.save(std_path, std)

    print(f'\nSaved to {args.output_dir}/')
    print(f'  Mean: shape={mean.shape}, range=[{mean.min():.4f}, {mean.max():.4f}]')
    print(f'  Std:  shape={std.shape}, range=[{std.min():.4f}, {std.max():.4f}]')

    # Compare with local stats if available
    local_stats_dir = 'data/hymotion_m2m_data/_stats'
    if osp.exists(osp.join(local_stats_dir, 'Std.npy')):
        local_std = np.load(osp.join(local_stats_dir, 'Std.npy'))
        print(f'\n  Comparison with local rotation stats:')
        # Translation dims (0:3)
        print(f'    Translation Std (local): {local_std[0:3]}')
        print(f'    Translation Std (global): {std[0:3]}')
        # Some distal joints (e.g. L_Wrist=joint20, dims 123:129)
        print(f'    L_Wrist Std (local):  {local_std[123:129].mean():.4f}')
        print(f'    L_Wrist Std (global): {std[123:129].mean():.4f}')
        print(f'    Root Std (local):  {local_std[3:9].mean():.4f}')
        print(f'    Root Std (global): {std[3:9].mean():.4f}')


if __name__ == '__main__':
    main()
