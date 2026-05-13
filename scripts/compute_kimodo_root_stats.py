"""
Compute KIMODO Root mean/std statistics for 198-dim motion.

This script:
1. Uses the same dataset pipeline as E3/E4 training configs
2. Applies LoadSmplx55 → Compute198DimPosition → SmplTransToKimodoRootOnline
3. Accumulates per-dimension statistics using Welford's online algorithm
4. Saves Mean.npy and Std.npy to _stats_198dim_kimodo_root/

Usage:
    python scripts/compute_kimodo_root_stats.py \
        --output-dir data/hymotion_m2m_data/_stats_198dim_kimodo_root \
        --max-samples 0

    # Quick test (100 samples):
    python scripts/compute_kimodo_root_stats.py --max-samples 100
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path (same as tools/train.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

# Import the actual transforms used in training
from hftrainer.datasets.motion.motionhub.transforms import (
    LoadSmplx55,
    Compute198DimPosition,
    SmplTransToKimodoRootOnline,
)


DATA_DIR = 'data/motionhub'
ANNO_FILE = 'data/annotation/train_hymotion_400h_hq_20260403.json'


def main():
    parser = argparse.ArgumentParser(description='Compute KIMODO Root statistics')
    parser.add_argument('--anno-file', type=str, default=ANNO_FILE,
                        help='Path to annotation file')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Path to motionhub data directory')
    parser.add_argument('--output-dir', type=str,
                        default='data/hymotion_m2m_data/_stats_198dim_kimodo_root',
                        help='Output directory for statistics')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Max samples to load (0=all)')
    parser.add_argument('--admm-margin-m', type=float, default=0.06,
                        help='ADMM smoothing margin in meters (default 0.06)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotation
    print(f"Loading annotation from {args.anno_file}...")
    with open(args.anno_file, 'r') as f:
        anno = json.load(f)
    data_list = anno.get('data_list', anno)  # dict or list
    if isinstance(data_list, dict):
        items = list(data_list.values())
    else:
        items = list(data_list)
    print(f"  Total entries: {len(items)}")

    if args.max_samples > 0:
        items = items[:args.max_samples]
        print(f"  Using first {args.max_samples} entries")

    # Build transforms (same as E3/E4 pipeline, minus caption/crop/conditioning)
    load_smplx = LoadSmplx55(
        key='motion',
        rot_type='rotation_6d',
        transl_type='abs',
        smpl_type='smpl_22',
    )
    compute_198 = Compute198DimPosition(key='motion')
    kimodo_transform = SmplTransToKimodoRootOnline(
        key='motion',
        admm_margin_m=args.admm_margin_m,
    )

    # Welford's online algorithm for mean/var (memory efficient, no OOM)
    count = 0
    mean_acc = np.zeros(198, dtype=np.float64)
    m2_acc = np.zeros(198, dtype=np.float64)

    success = 0
    fail = 0

    for i, item in enumerate(tqdm(items, desc="Computing KIMODO Root stats")):
        try:
            # Resolve motion path
            smplx_path = item.get('smplx_path')
            if not smplx_path:
                fail += 1
                continue
            motion_path = os.path.join(args.data_dir, smplx_path)

            # Build results dict as the dataset would
            results = {
                'motion_path': motion_path,
                'fps': item.get('fps', 30.0),
            }

            # Step 1: LoadSmplx55 — loads NPZ, converts to rot6d, outputs (T, 135)
            results = load_smplx.transform(results)

            # Step 2: Compute198DimPosition — FK to get positions, outputs (T, 198)
            results = compute_198.transform(results)

            # Step 3: SmplTransToKimodoRootOnline — ADMM smoothing, outputs (T, 198)
            results = kimodo_transform.transform(results)

            motion_kimodo = results['motion']  # (T, 198) tensor
            frames = motion_kimodo.numpy().astype(np.float64)  # (T, 198)

            # Vectorized Welford update (batch all frames at once)
            T_frames = frames.shape[0]
            for j in range(T_frames):
                count += 1
                delta = frames[j] - mean_acc
                mean_acc += delta / count
                delta2 = frames[j] - mean_acc
                m2_acc += delta * delta2

            success += 1

        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  [WARN] Failed on item {i}: {e}")

        if (i + 1) % 10000 == 0:
            print(f"  [{i+1}/{len(items)}] ok={success}, fail={fail}, frames={count}")

    if count < 2:
        print(f"ERROR: Only {count} frames accumulated. Cannot compute statistics.")
        return

    # Finalize statistics
    mean = mean_acc.astype(np.float32)
    var = (m2_acc / count).astype(np.float32)
    std = np.sqrt(var).astype(np.float32)
    std = np.maximum(std, 1e-6)  # Clamp to avoid /0

    # Save
    np.save(output_dir / 'Mean.npy', mean)
    np.save(output_dir / 'Std.npy', std)

    print(f"\n{'='*60}")
    print(f"✅ KIMODO Root statistics computed!")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}/")
    print(f"  Sequences: {success} ok, {fail} failed")
    print(f"  Total frames: {count:,}")
    print(f"  Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f"\n  Translation [0:3]:")
    print(f"    Mean: {mean[0:3]}")
    print(f"    Std:  {std[0:3]}")
    print(f"  Root rot6d [3:9]:")
    print(f"    Mean: {mean[3:9]}")
    print(f"    Std:  {std[3:9]}")
    print(f"  First position joint [135:138]:")
    print(f"    Mean: {mean[135:138]}")
    print(f"    Std:  {std[135:138]}")

    # Compare with SMPL Root stats
    smpl_stats_dir = Path('data/hymotion_m2m_data/_stats_198dim')
    if (smpl_stats_dir / 'Mean.npy').exists():
        smpl_mean = np.load(smpl_stats_dir / 'Mean.npy')
        smpl_std = np.load(smpl_stats_dir / 'Std.npy')
        print(f"\n  === Comparison with SMPL Root stats ===")
        print(f"  Translation Mean diff: {np.abs(mean[0:3] - smpl_mean[0:3])}")
        print(f"  Translation Std diff:  {np.abs(std[0:3] - smpl_std[0:3])}")
        rot_mean_diff = np.abs(mean[3:135] - smpl_mean[3:135])
        pos_mean_diff = np.abs(mean[135:198] - smpl_mean[135:198])
        print(f"  Rotation Mean diff: max={rot_mean_diff.max():.6f}, avg={rot_mean_diff.mean():.6f}")
        print(f"  Position Mean diff: max={pos_mean_diff.max():.6f}, avg={pos_mean_diff.mean():.6f}")
        print(f"  NOTE: Rotation dims [3:135] should be IDENTICAL (ADMM only changes trans+pos)")
        if rot_mean_diff.max() > 0.001:
            print(f"  ⚠️  WARNING: Rotation dims differ! This should not happen.")
        else:
            print(f"  ✅  Rotation dims match (as expected).")


if __name__ == '__main__':
    main()
