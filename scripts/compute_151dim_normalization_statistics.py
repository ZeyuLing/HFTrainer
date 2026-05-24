#!/usr/bin/env python3
"""
Compute normalization statistics (Mean.npy and Std.npy) for 151-dimensional motion.

151-dim motion = 147-dim (translation + rotation6d + end-effector positions) + 4-dim (foot contact)

This script:
1. Loads the 147-dim Mean/Std statistics
2. Loads motion samples and computes foot contact statistics
3. Combines them to create 151-dim Mean/Std

Note: Foot contact is binary (0 or 1), but we still compute mean and std for consistency
in the normalization pipeline.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def compute_151dim_statistics(
    anno_file: str,
    data_dir: str,
    output_dir: str = 'data/hymotion_m2m_data/_stats_151dim',
    motion_key: str = 'smplx',
    num_samples: int = None,
):
    """
    Compute 151-dim motion statistics from annotations.
    
    Args:
        anno_file: Path to annotation JSON file
        data_dir: Path to motion data directory
        output_dir: Output directory for Mean.npy and Std.npy
        motion_key: Motion key in annotation (e.g., 'smplx')
        num_samples: Number of samples to process (None = all)
    """
    from hftrainer.datasets.motion.motionhub.transforms import (
        LoadSmplx55, Compute147DimEndEffector, Compute151DimFootContact
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load 147-dim statistics as baseline
    stats_147_dir = 'data/hymotion_m2m_data/_stats_147dim'
    assert os.path.exists(os.path.join(stats_147_dir, 'Mean.npy')), \
        f"Missing 147-dim statistics at {stats_147_dir}. Compute them first."
    
    mean_147 = np.load(os.path.join(stats_147_dir, 'Mean.npy'))
    std_147 = np.load(os.path.join(stats_147_dir, 'Std.npy'))
    
    print(f"Loaded 147-dim statistics:")
    print(f"  Mean shape: {mean_147.shape}, Min: {mean_147.min():.4f}, Max: {mean_147.max():.4f}")
    print(f"  Std shape: {std_147.shape}, Min: {std_147.min():.4f}, Max: {std_147.max():.4f}")
    
    # Initialize accumulators for foot contact dims
    contact_sums = np.zeros(4)
    contact_sq_sums = np.zeros(4)
    contact_count = 0
    
    # Initialize transforms
    load_smplx = LoadSmplx55(
        key='motion',
        rot_type='rotation_6d',
        transl_type='abs',
        smpl_type='smpl_22',
    )
    compute_147 = Compute147DimEndEffector(key='motion')
    compute_151 = Compute151DimFootContact(
        key='motion',
        bone_offsets_dir='data/bone_offsets',
        velocity_threshold=0.002,
    )
    
    # Load annotation file
    with open(anno_file, 'r') as f:
        data = json.load(f)
    
    # Handle dict format with 'data_list' key where each value is a dict
    if isinstance(data, dict) and 'data_list' in data:
        data_list = data['data_list']
        # data_list is a dict, extract values
        annotations = list(data_list.values())
    else:
        annotations = data if isinstance(data, list) else list(data.values())
    
    print(f"\nProcessing {len(annotations)} annotations...")
    count = 0
    
    for idx, ann in enumerate(tqdm(annotations, desc='Computing statistics')):
        if num_samples is not None and count >= num_samples:
            break
        
        # Get motion path
        motion_path = ann.get('smplx_path') or ann.get(motion_key)
        if motion_path is None:
            continue
        
        motion_path = os.path.join(data_dir, motion_path)
        if not os.path.exists(motion_path):
            continue
        
        try:
            # Load and transform motion
            results = {
                'motion_path': motion_path,
                'fps': ann.get('fps', 30),
            }
            
            # Step 1: Load SMPLX data (returns 135-dim)
            results = load_smplx(results)
            if results is None:
                continue
            
            # Step 2: Compute 147-dim (add end-effector positions)
            results = compute_147(results)
            if results is None:
                continue
            
            # Step 3: Compute 151-dim (add foot contact)
            results = compute_151(results)
            if results is None:
                continue
            
            # Extract motion (151-dim)
            motion_151 = results['motion']  # (T, 151)
            
            # Accumulate statistics for foot contact dims [147:151]
            contact_data = motion_151[:, 147:151]  # (T, 4)
            
            contact_data = np.array(contact_data) if isinstance(contact_data, torch.Tensor) else contact_data
            contact_sums += contact_data.sum(axis=0)
            contact_sq_sums += (contact_data ** 2).sum(axis=0)
            contact_count += contact_data.shape[0]
            
            count += 1
            
        except Exception as e:
            continue
    
    print(f"\nProcessed {count} motion files")
    print(f"Total contact frames: {contact_count}")
    
    # Compute foot contact statistics
    if contact_count > 0:
        contact_mean = contact_sums / contact_count
        contact_var = (contact_sq_sums / contact_count) - (contact_mean ** 2)
        contact_std = np.sqrt(np.maximum(contact_var, 1e-6))  # Prevent negative variance
    else:
        print("Warning: No contact data found, using default statistics")
        contact_mean = np.array([0.5, 0.5, 0.5, 0.5])  # Default: 50% contact probability
        contact_std = np.array([0.5, 0.5, 0.5, 0.5])  # Default: high variance
    
    print(f"\nFootContact statistics:")
    print(f"  Mean: {contact_mean}")
    print(f"  Std: {contact_std}")
    
    # Concatenate to create 151-dim statistics
    mean_151 = np.concatenate([mean_147, contact_mean])
    std_151 = np.concatenate([std_147, contact_std])
    
    print(f"\n151-dim combined statistics:")
    print(f"  Mean shape: {mean_151.shape}, Min: {mean_151.min():.4f}, Max: {mean_151.max():.4f}")
    print(f"  Std shape: {std_151.shape}, Min: {std_151.min():.4f}, Max: {std_151.max():.4f}")
    
    # Save statistics
    mean_path = os.path.join(output_dir, 'Mean.npy')
    std_path = os.path.join(output_dir, 'Std.npy')
    
    np.save(mean_path, mean_151.astype(np.float32))
    np.save(std_path, std_151.astype(np.float32))
    
    print(f"\nSaved statistics:")
    print(f"  Mean: {mean_path}")
    print(f"  Std: {std_path}")
    
    return mean_151, std_151


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute 151-dim motion statistics')
    parser.add_argument(
        '--anno-file',
        default='data/annotation/train_hymotion_400h_hq_20260403.json',
        help='Path to annotation JSON file'
    )
    parser.add_argument(
        '--data-dir',
        default='data/motionhub',
        help='Path to motion data directory'
    )
    parser.add_argument(
        '--output-dir',
        default='data/hymotion_m2m_data/_stats_151dim',
        help='Output directory for statistics'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to process (None = all)'
    )
    
    args = parser.parse_args()
    
    mean, std = compute_151dim_statistics(
        anno_file=args.anno_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )
    
    print("\n✓ Done!")
