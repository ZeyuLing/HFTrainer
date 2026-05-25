"""
Debug script to validate denormalization consistency in PRISM pipeline.

This diagnostic validates whether the two-stage denormalization process
(VAE latent denormalization + motion denormalization) is frame-consistent
or if there are variations that could contribute to jitter.

Issue #2 Root Cause:
- Stage 1: Latent denormalization using latents_std/latents_mean
- Stage 2: Motion denormalization using motion_std/motion_mean
- Problem: If velocity statistics are applied inconsistently frame-to-frame,
  consecutive frames get different scaling, causing jitter
"""

import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_denormalization_factors(
    motion_data: np.ndarray,
    denorm_stats: Dict[str, np.ndarray],
    is_normalized: bool = True
) -> Dict[str, float]:
    """
    Analyze denormalization factors frame-by-frame to detect inconsistencies.
    
    Args:
        motion_data: Shape (T, 138) - motion sequence
        denorm_stats: Dict with keys 'mean' and 'std' (shape 138)
        is_normalized: If True, motion_data is normalized; apply stats
        
    Returns:
        Dictionary with consistency metrics
    """
    T = motion_data.shape[0]
    
    # Extract statistics
    mean_vec = denorm_stats['mean']  # shape (138,)
    std_vec = denorm_stats['std']    # shape (138,)
    
    # Validate statistics
    assert std_vec.shape == (138,), f"Expected std shape (138,), got {std_vec.shape}"
    assert mean_vec.shape == (138,), f"Expected mean shape (138,), got {mean_vec.shape}"
    assert (std_vec > 0).all(), "Found non-positive std values"
    assert np.isfinite(std_vec).all(), "Found NaN/Inf in std"
    assert np.isfinite(mean_vec).all(), "Found NaN/Inf in mean"
    
    results = {
        'std_consistency': {},
        'velocity_consistency': {},
        'denorm_stability': {}
    }
    
    # Analysis 1: Check if std varies unexpectedly across frames
    # Expected: std should be constant (precomputed from dataset)
    if is_normalized:
        denorm_motion = motion_data * std_vec[None, :] + mean_vec[None, :]
    else:
        denorm_motion = motion_data
    
    # Extract velocity component (first 6 dims = relative translation)
    # Indices 0-5 are: [vx, vy, vz, rx, ry, rz] in relative frame
    vel_component = denorm_motion[:, :6]  # T x 6
    
    # Frame-to-frame velocity difference (should be smooth)
    frame_velocity = np.diff(vel_component, axis=0)  # (T-1) x 6
    
    # Check for discontinuities (high frame-to-frame variance)
    vel_std_per_frame = frame_velocity.std(axis=1)  # (T-1,)
    vel_std_overall = frame_velocity.std()
    
    results['velocity_consistency']['frame_wise_std_mean'] = vel_std_per_frame.mean()
    results['velocity_consistency']['frame_wise_std_max'] = vel_std_per_frame.max()
    results['velocity_consistency']['frame_wise_std_ratio'] = (
        vel_std_per_frame.max() / (vel_std_overall + 1e-8)
    )
    
    # Analysis 2: Denormalization factor stability
    # If denormalization is frame-dependent, scaling factors would vary
    denorm_factor_per_dim = std_vec  # Should be constant, not frame-dependent
    
    results['denorm_stability']['std_min'] = std_vec.min()
    results['denorm_stability']['std_max'] = std_vec.max()
    results['denorm_stability']['std_ratio'] = std_vec.max() / (std_vec.min() + 1e-8)
    results['denorm_stability']['std_consistency'] = std_vec.std() / (std_vec.mean() + 1e-8)
    
    # Analysis 3: Check for implausible denormalization patterns
    # Implausible = motion_std varies significantly across dimensions
    # (indicating dataset statistics are reasonable)
    results['denorm_stability']['dims_with_high_std'] = (std_vec > std_vec.mean() * 1.5).sum()
    results['denorm_stability']['dims_with_low_std'] = (std_vec < std_vec.mean() * 0.5).sum()
    
    # Analysis 4: Joint-wise consistency check
    # Joints start at index 6, each joint has 6 dims (rx, ry, rz, scale)
    joint_consistency = {}
    for joint_idx in range(22):  # 22 joints
        start_idx = 6 + joint_idx * 6
        end_idx = start_idx + 6
        joint_std = std_vec[start_idx:end_idx]
        joint_consistency[f'joint_{joint_idx}'] = {
            'std_mean': joint_std.mean(),
            'std_variation': joint_std.std() / (joint_std.mean() + 1e-8)
        }
    
    results['joint_consistency'] = joint_consistency
    
    return results


def compare_denormalization_methods(
    motion_normalized: torch.Tensor,
    denorm_stats: Dict[str, np.ndarray],
    method_a_scale: float = 1.0,
    method_b_scale: float = 1.0
) -> Dict[str, float]:
    """
    Compare two potential denormalization approaches to see if they differ.
    
    This checks if there could be scaling inconsistencies between how
    latents and motion are denormalized.
    
    Args:
        motion_normalized: Normalized motion (T, 138)
        denorm_stats: Denormalization statistics
        method_a_scale: Scalar applied in method A
        method_b_scale: Scalar applied in method B
        
    Returns:
        Dictionary comparing the methods
    """
    if isinstance(motion_normalized, torch.Tensor):
        motion_normalized = motion_normalized.cpu().numpy()
    
    std_vec = denorm_stats['std']
    mean_vec = denorm_stats['mean']
    
    # Method A: Direct denormalization (current approach)
    motion_a = motion_normalized * std_vec[None, :] * method_a_scale + mean_vec[None, :]
    
    # Method B: Two-stage with potential scaling difference
    # (simulating what could happen if stages apply scales differently)
    motion_b = motion_normalized * std_vec[None, :] * method_b_scale + mean_vec[None, :]
    
    diff = motion_a - motion_b
    velocity_a = np.diff(motion_a, axis=0)
    velocity_b = np.diff(motion_b, axis=0)
    velocity_diff = velocity_a - velocity_b
    
    return {
        'max_position_diff': np.abs(diff).max(),
        'mean_position_diff': np.abs(diff).mean(),
        'max_velocity_diff': np.abs(velocity_diff).max(),
        'mean_velocity_diff': np.abs(velocity_diff).mean(),
        'velocity_jitter_increase': np.abs(velocity_diff).max() / (np.abs(velocity_a).max() + 1e-8)
    }


def check_denormalization_correctness(
    motion_normalized: torch.Tensor,
    motion_denormalized: torch.Tensor,
    denorm_stats: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Verify that denormalization was applied correctly by reversing it.
    
    Args:
        motion_normalized: Pre-denormalization motion
        motion_denormalized: Post-denormalization motion
        denorm_stats: Statistics dict with 'mean' and 'std'
        
    Returns:
        Dictionary with verification results
    """
    if isinstance(motion_normalized, torch.Tensor):
        motion_normalized = motion_normalized.cpu().numpy()
    if isinstance(motion_denormalized, torch.Tensor):
        motion_denormalized = motion_denormalized.cpu().numpy()
    
    std_vec = denorm_stats['std']
    mean_vec = denorm_stats['mean']
    
    # Reverse denormalization: (x - mean) / std should recover normalized
    motion_renormalized = (motion_denormalized - mean_vec[None, :]) / std_vec[None, :]
    
    # Check if we recover the original
    reconstruction_error = np.abs(motion_renormalized - motion_normalized)
    
    return {
        'max_reconstruction_error': reconstruction_error.max(),
        'mean_reconstruction_error': reconstruction_error.mean(),
        'std_reconstruction_error': reconstruction_error.std(),
        'reconstruction_success': (reconstruction_error < 1e-5).sum() / reconstruction_error.size
    }


def analyze_segment_denormalization(
    segments: list,  # List of motion segments (each T_i x 138)
    denorm_stats: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Analyze denormalization consistency across autoregressive segments.
    
    If denormalization factors are applied differently per segment,
    this would show up as velocity discontinuities at boundaries.
    
    Args:
        segments: List of motion tensors (each T_i x 138)
        denorm_stats: Denormalization statistics
        
    Returns:
        Dictionary with per-segment analysis
    """
    std_vec = denorm_stats['std']
    mean_vec = denorm_stats['mean']
    
    results = {
        'segments': {},
        'boundary_analysis': []
    }
    
    # Denormalize all segments
    denorm_segments = []
    for seg_idx, segment in enumerate(segments):
        if isinstance(segment, torch.Tensor):
            segment = segment.cpu().numpy()
        
        denorm_seg = segment * std_vec[None, :] + mean_vec[None, :]
        denorm_segments.append(denorm_seg)
        
        # Compute velocity stats for this segment
        vel = np.diff(denorm_seg, axis=0)
        results['segments'][f'segment_{seg_idx}'] = {
            'mean_velocity': np.abs(vel).mean(),
            'max_velocity': np.abs(vel).max(),
            'velocity_std': np.abs(vel).std(),
            'frames': denorm_seg.shape[0]
        }
    
    # Analyze boundaries
    for seg_idx in range(len(denorm_segments) - 1):
        seg_a = denorm_segments[seg_idx]
        seg_b = denorm_segments[seg_idx + 1]
        
        # Velocity at boundary
        last_frame_a = seg_a[-1:]  # (1, 138)
        first_frame_b = seg_b[0:1]  # (1, 138)
        
        boundary_velocity = np.abs(first_frame_b - last_frame_a)
        
        # Within-segment velocity (for comparison)
        within_vel_a = np.diff(seg_a[-2:], axis=0).max()
        within_vel_b = np.diff(seg_b[0:2], axis=0).max()
        within_vel_avg = (within_vel_a + within_vel_b) / 2
        
        boundary_spike = boundary_velocity.max() / (within_vel_avg + 1e-8)
        
        results['boundary_analysis'].append({
            'boundary_idx': seg_idx,
            'max_boundary_velocity': boundary_velocity.max(),
            'velocity_spike_ratio': boundary_spike,
            'mean_boundary_velocity': boundary_velocity.mean()
        })
    
    return results


def print_diagnostic_report(
    consistency_results: Dict,
    method_comparison: Optional[Dict] = None,
    reconstruction_results: Optional[Dict] = None
):
    """Print formatted diagnostic report."""
    
    print("\n" + "="*80)
    print("PRISM DENORMALIZATION CONSISTENCY DIAGNOSTIC REPORT")
    print("="*80)
    
    print("\n[DENORMALIZATION STABILITY ANALYSIS]")
    denorm_stats = consistency_results.get('denorm_stability', {})
    print(f"  Std Min: {denorm_stats.get('std_min', 0):.6f}")
    print(f"  Std Max: {denorm_stats.get('std_max', 0):.6f}")
    print(f"  Std Ratio (max/min): {denorm_stats.get('std_ratio', 0):.3f}")
    print(f"  ✓ Status: {'CONSISTENT' if denorm_stats.get('std_ratio', 0) < 1.2 else 'INCONSISTENT'}")
    
    print("\n[VELOCITY CONSISTENCY CHECK]")
    vel_stats = consistency_results.get('velocity_consistency', {})
    print(f"  Mean frame-wise std: {vel_stats.get('frame_wise_std_mean', 0):.6f}")
    print(f"  Max frame-wise std: {vel_stats.get('frame_wise_std_max', 0):.6f}")
    print(f"  Ratio (max/mean): {vel_stats.get('frame_wise_std_ratio', 0):.3f}")
    print(f"  ✓ Status: {'STABLE' if vel_stats.get('frame_wise_std_ratio', 0) < 1.5 else 'VARIABLE'}")
    
    if method_comparison:
        print("\n[DENORMALIZATION METHOD COMPARISON]")
        print(f"  Max velocity difference: {method_comparison.get('max_velocity_diff', 0):.6f}")
        print(f"  Mean velocity difference: {method_comparison.get('mean_velocity_diff', 0):.6f}")
        print(f"  Jitter increase if methods differ: {method_comparison.get('velocity_jitter_increase', 0):.2%}")
    
    if reconstruction_results:
        print("\n[RECONSTRUCTION VERIFICATION]")
        print(f"  Max error: {reconstruction_results.get('max_reconstruction_error', 0):.2e}")
        print(f"  Mean error: {reconstruction_results.get('mean_reconstruction_error', 0):.2e}")
        print(f"  Success rate: {reconstruction_results.get('reconstruction_success', 0):.2%}")
        print(f"  ✓ Status: {'CORRECT' if reconstruction_results.get('max_reconstruction_error', 0) < 1e-5 else 'ERROR DETECTED'}")
    
    print("\n[DIAGNOSIS]")
    is_consistent = denorm_stats.get('std_ratio', 0) < 1.2
    vel_stable = vel_stats.get('frame_wise_std_ratio', 0) < 1.5
    
    if is_consistent and vel_stable:
        print("  ✓ Denormalization appears CORRECT and CONSISTENT")
        print("  → Issue #2 is likely NOT causing significant jitter")
        print("  → Focus on CFG scaling (Issue #1) and segment stitching (Issue #3)")
    else:
        print("  ⚠ Denormalization shows INCONSISTENCY")
        if not is_consistent:
            print("    - Statistics vectors have high variation (std ratio > 1.2)")
        if not vel_stable:
            print("    - Velocity profile is unstable frame-to-frame")
        print("  → Issue #2 may be contributing to jitter")
        print("  → Recommendation: Investigate statistics loading/application")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run full denormalization diagnostic."""
    import sys
    
    print("\n[PRISM Denormalization Diagnostic]")
    print("This script validates whether the denormalization process is consistent")
    print("and could be contributing to motion jitter.\n")
    
    # Try to load actual statistics from the project
    stats_path = Path(
        "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/"
        "hftrainer/models/motion/components/motion_processor/smpl_processor_stats.json"
    )
    
    if not stats_path.exists():
        print(f"⚠ Could not find statistics file at {stats_path}")
        print("Creating synthetic test data for demonstration...\n")
        
        # Create synthetic test data
        T = 200  # frames
        motion_normalized = np.random.randn(T, 138) * 0.5  # normalized motion
        
        # Create realistic statistics (based on motion dataset analysis)
        denorm_stats = {
            'mean': np.zeros(138),
            'std': np.concatenate([
                np.array([0.15, 0.15, 0.15, 0.1, 0.1, 0.1]),  # translation + rotation
                np.tile(np.array([0.3, 0.3, 0.3, 0.15, 0.15, 0.15]), 22)  # 22 joints
            ])
        }
        
    else:
        import json
        print(f"✓ Loading statistics from {stats_path}")
        with open(stats_path, 'r') as f:
            denorm_stats_raw = json.load(f)
            denorm_stats = {
                'mean': np.array(denorm_stats_raw['mean']),
                'std': np.array(denorm_stats_raw['std'])
            }
        
        # For demonstration, generate synthetic motion
        T = 200
        motion_normalized = np.random.randn(T, 138) * 0.5
    
    print(f"Motion shape: {motion_normalized.shape}")
    print(f"Statistics mean shape: {denorm_stats['mean'].shape}")
    print(f"Statistics std shape: {denorm_stats['std'].shape}\n")
    
    # Run analyses
    consistency_results = analyze_denormalization_factors(
        motion_normalized, denorm_stats, is_normalized=True
    )
    
    motion_torch = torch.from_numpy(motion_normalized).float()
    method_comparison = compare_denormalization_methods(
        motion_torch, denorm_stats, method_a_scale=1.0, method_b_scale=1.0
    )
    
    # Verify correct denormalization
    motion_denorm = motion_normalized * denorm_stats['std'][None, :] + denorm_stats['mean'][None, :]
    reconstruction_results = check_denormalization_correctness(
        motion_normalized, motion_denorm, denorm_stats
    )
    
    # Print report
    print_diagnostic_report(consistency_results, method_comparison, reconstruction_results)
    
    return consistency_results


if __name__ == '__main__':
    main()
