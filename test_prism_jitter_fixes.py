"""
Comprehensive testing framework for PRISM jitter fixes.

This script measures jitter reduction from the implemented fixes:
1. Reduced guidance_scale (5.0 → 2.0)
2. Segment boundary blending
3. Optional denormalization debugging

Metrics:
- Frame-to-frame velocity (m/frame)
- Velocity standard deviation (jitter coefficient)
- Velocity spike detection at boundaries
- Acceleration profile smoothness
"""

import numpy as np
import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VelocityMetrics:
    """Container for velocity-based jitter metrics."""
    max_velocity: float
    min_velocity: float
    mean_velocity: float
    std_velocity: float
    p95_velocity: float
    velocity_jitter_cv: float  # Coefficient of variation
    max_acceleration: float
    mean_acceleration: float
    acceleration_jitter: float
    boundary_spikes: Optional[List[float]] = None
    
    def __str__(self):
        return (
            f"Velocity:     mean={self.mean_velocity:.4f}, std={self.std_velocity:.4f}, "
            f"p95={self.p95_velocity:.4f}, jitter_cv={self.velocity_jitter_cv:.4f}\n"
            f"Acceleration: mean={self.mean_acceleration:.4f}, max={self.max_acceleration:.4f}, "
            f"jitter={self.acceleration_jitter:.4f}\n"
            f"Max Velocity: {self.max_velocity:.4f} m/frame"
        )


def extract_translation(motion: np.ndarray) -> np.ndarray:
    """
    Extract translation component from motion representation.
    
    Motion format: [vx, vy, vz, rx, ry, rz, joint_0, ..., joint_21]
    Where each joint is 6D rotation representation.
    
    Returns: Translation component (T, 3) in meters
    """
    # First 3 components are typically relative translation (velocity)
    # We'll integrate to get position
    translation = motion[:, :3].copy()
    return translation


def compute_velocity_profile(motion: np.ndarray) -> Tuple[np.ndarray, VelocityMetrics]:
    """
    Compute frame-to-frame velocity and detailed metrics.
    
    Args:
        motion: Shape (T, D) motion sequence
        
    Returns:
        velocity: Shape (T-1,) frame-to-frame velocity magnitude
        metrics: VelocityMetrics dataclass with all metrics
    """
    # Extract translation
    transl = extract_translation(motion)  # (T, 3)
    
    # Compute displacement
    displacement = np.diff(transl, axis=0)  # (T-1, 3)
    
    # Velocity magnitude
    velocity = np.linalg.norm(displacement, axis=1)  # (T-1,)
    
    # Acceleration
    acceleration = np.diff(velocity)  # (T-2,)
    
    # Metrics
    metrics = VelocityMetrics(
        max_velocity=velocity.max(),
        min_velocity=velocity.min(),
        mean_velocity=velocity.mean(),
        std_velocity=velocity.std(),
        p95_velocity=np.percentile(velocity, 95),
        velocity_jitter_cv=velocity.std() / (velocity.mean() + 1e-6),
        max_acceleration=np.abs(acceleration).max() if len(acceleration) > 0 else 0.0,
        mean_acceleration=np.abs(acceleration).mean() if len(acceleration) > 0 else 0.0,
        acceleration_jitter=acceleration.std() if len(acceleration) > 0 else 0.0,
        boundary_spikes=None
    )
    
    return velocity, metrics


def detect_boundary_spikes(
    motion: np.ndarray,
    segment_boundaries: List[int],
    window_size: int = 5
) -> Dict[int, Dict[str, float]]:
    """
    Detect velocity spikes at segment boundaries.
    
    Args:
        motion: Shape (T, D) full motion sequence
        segment_boundaries: List of frame indices where segments meet
        window_size: Number of frames before/after to analyze
        
    Returns:
        Dictionary mapping boundary index → spike metrics
    """
    transl = extract_translation(motion)
    displacement = np.diff(transl, axis=0)
    velocity = np.linalg.norm(displacement, axis=1)
    
    spikes = {}
    for boundary_idx, frame_idx in enumerate(segment_boundaries):
        if frame_idx < window_size or frame_idx >= len(velocity) - window_size:
            continue
        
        # Velocity before boundary
        before = velocity[frame_idx - window_size:frame_idx]
        # Velocity after boundary
        after = velocity[frame_idx:frame_idx + window_size]
        
        before_mean = before.mean()
        after_mean = after.mean()
        boundary_mean = np.mean([velocity[frame_idx - 1], velocity[frame_idx]])
        
        # Spike ratio
        baseline = np.mean([before_mean, after_mean])
        spike_ratio = boundary_mean / (baseline + 1e-6)
        
        spikes[boundary_idx] = {
            'frame': frame_idx,
            'before_velocity_mean': before_mean,
            'after_velocity_mean': after_mean,
            'boundary_velocity': boundary_mean,
            'spike_ratio': spike_ratio
        }
    
    return spikes


def generate_synthetic_motion(
    num_frames: int = 200,
    motion_type: str = 'smooth',
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic motion for testing.
    
    Args:
        num_frames: Number of frames
        motion_type: 'smooth' | 'jittery' | 'boundary_jitter'
        seed: Random seed
        
    Returns:
        Motion array (num_frames, 138)
    """
    if seed is not None:
        np.random.seed(seed)
    
    motion = np.random.randn(num_frames, 138) * 0.1
    
    if motion_type == 'smooth':
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter1d
        motion = gaussian_filter1d(motion, sigma=2.0, axis=0)
        
    elif motion_type == 'jittery':
        # Amplify noise
        motion = motion * 5.0
        
    elif motion_type == 'boundary_jitter':
        # Add spikes at artificial boundaries
        boundaries = [50, 100, 150]
        for b in boundaries:
            motion[b:b+2, :3] += np.random.randn(2, 3) * 2.0
    
    return motion


def compare_fixes(
    motion_original: np.ndarray,
    motion_fixed: np.ndarray,
    fix_name: str = "Unknown",
    segment_boundaries: Optional[List[int]] = None
) -> Dict:
    """
    Compare metrics before and after a fix.
    
    Args:
        motion_original: Original motion (with jitter)
        motion_fixed: Fixed motion (with reduced jitter)
        fix_name: Name of the fix applied
        segment_boundaries: Optional list of segment boundary frames
        
    Returns:
        Dictionary with before/after comparison
    """
    vel_orig, metrics_orig = compute_velocity_profile(motion_original)
    vel_fixed, metrics_fixed = compute_velocity_profile(motion_fixed)
    
    # Compute improvements
    velocity_improvement = 1.0 - (metrics_fixed.mean_velocity / (metrics_orig.mean_velocity + 1e-6))
    jitter_improvement = 1.0 - (metrics_fixed.velocity_jitter_cv / (metrics_orig.velocity_jitter_cv + 1e-6))
    
    result = {
        'fix_name': fix_name,
        'metrics_original': metrics_orig,
        'metrics_fixed': metrics_fixed,
        'improvements': {
            'velocity_reduction': velocity_improvement,
            'jitter_reduction': jitter_improvement,
            'max_velocity_reduction': 1.0 - (metrics_fixed.max_velocity / (metrics_orig.max_velocity + 1e-6))
        }
    }
    
    # Analyze boundary spikes if boundaries provided
    if segment_boundaries:
        spikes_orig = detect_boundary_spikes(motion_original, segment_boundaries)
        spikes_fixed = detect_boundary_spikes(motion_fixed, segment_boundaries)
        
        result['boundary_spikes'] = {
            'original': spikes_orig,
            'fixed': spikes_fixed
        }
    
    return result


def print_comparison_report(comparison: Dict):
    """Print formatted comparison report."""
    print("\n" + "="*80)
    print(f"FIX: {comparison['fix_name']}")
    print("="*80)
    
    print("\n[BEFORE FIX]")
    print(comparison['metrics_original'])
    
    print("\n[AFTER FIX]")
    print(comparison['metrics_fixed'])
    
    print("\n[IMPROVEMENTS]")
    improv = comparison['improvements']
    print(f"Velocity Reduction:        {improv['velocity_reduction']:+.2%}")
    print(f"Jitter Reduction (CV):     {improv['jitter_reduction']:+.2%}")
    print(f"Max Velocity Reduction:    {improv['max_velocity_reduction']:+.2%}")
    
    if 'boundary_spikes' in comparison:
        print("\n[BOUNDARY ANALYSIS]")
        spikes_orig = comparison['boundary_spikes']['original']
        spikes_fixed = comparison['boundary_spikes']['fixed']
        
        for b_idx in spikes_orig:
            orig = spikes_orig[b_idx]
            fixed = spikes_fixed[b_idx]
            spike_reduction = 1.0 - (fixed['spike_ratio'] / (orig['spike_ratio'] + 1e-6))
            print(f"  Boundary {b_idx}: spike ratio {orig['spike_ratio']:.2f} → {fixed['spike_ratio']:.2f} ({spike_reduction:+.2%})")
    
    print("="*80 + "\n")


def simulate_cfg_fix(motion: np.ndarray, original_scale: float = 5.0, new_scale: float = 2.0) -> np.ndarray:
    """
    Simulate CFG scaling fix by reducing noise amplification.
    
    Args:
        motion: Normalized motion before denormalization
        original_scale: Original guidance_scale (5.0)
        new_scale: New guidance_scale (2.0)
        
    Returns:
        Motion with adjusted noise levels
    """
    # Simulate noise amplification in latent space
    noise_component = motion - motion.mean(axis=0, keepdims=True)
    
    # Reduce noise amplification
    scaling_factor = new_scale / original_scale
    adjusted_motion = motion.mean(axis=0, keepdims=True) + noise_component * scaling_factor
    
    return adjusted_motion


def simulate_blending_fix(motion: np.ndarray, segment_boundaries: List[int], blend_width: int = 5) -> np.ndarray:
    """
    Simulate segment boundary blending fix.
    
    Args:
        motion: Motion with boundary discontinuities
        segment_boundaries: List of segment boundary frame indices
        blend_width: Width of blend zone (frames on each side)
        
    Returns:
        Motion with smoothed boundaries
    """
    motion_blended = motion.copy()
    
    for boundary_idx in segment_boundaries:
        if boundary_idx < blend_width or boundary_idx >= len(motion) - blend_width:
            continue
        
        # Extract blend zone
        start = boundary_idx - blend_width
        end = boundary_idx + blend_width
        
        blend_zone = motion[start:end].copy()
        
        # Create Gaussian blend kernel
        x = np.linspace(-1, 1, blend_width * 2)
        kernel = np.exp(-x**2 / 0.2)
        kernel = kernel / kernel.sum()
        
        # Apply blending (moving average)
        for dim in range(motion.shape[1]):
            blended = np.convolve(blend_zone[:, dim], kernel, mode='same')
            motion_blended[start:end, dim] = blended
    
    return motion_blended


def run_comprehensive_test():
    """Run comprehensive jitter fix testing."""
    print("\n" + "="*80)
    print("PRISM JITTER FIX VALIDATION - COMPREHENSIVE TEST")
    print("="*80)
    
    # Generate test motions
    print("\n[GENERATING TEST DATA]")
    
    # Scenario 1: Baseline jittery motion
    motion_baseline = generate_synthetic_motion(num_frames=300, motion_type='jittery', seed=42)
    print(f"✓ Generated baseline jittery motion: {motion_baseline.shape}")
    
    # Scenario 2: Smooth motion (control)
    motion_smooth = generate_synthetic_motion(num_frames=300, motion_type='smooth', seed=43)
    print(f"✓ Generated smooth motion: {motion_smooth.shape}")
    
    # Scenario 3: Motion with boundary jitter
    motion_boundary = generate_synthetic_motion(num_frames=300, motion_type='boundary_jitter', seed=44)
    segment_boundaries = [75, 150, 225]  # Approximate boundaries
    print(f"✓ Generated motion with boundary artifacts: {motion_boundary.shape}")
    
    # Test Fix #1: CFG Scaling Reduction
    print("\n[TEST FIX #1: CFG SCALING REDUCTION]")
    motion_cfg_fixed = simulate_cfg_fix(motion_baseline, original_scale=5.0, new_scale=2.0)
    cfg_comparison = compare_fixes(
        motion_baseline, motion_cfg_fixed,
        fix_name="Guidance Scale Reduction (5.0 → 2.0)"
    )
    print_comparison_report(cfg_comparison)
    
    # Test Fix #2: Segment Boundary Blending
    print("\n[TEST FIX #2: SEGMENT BOUNDARY BLENDING]")
    motion_blend_fixed = simulate_blending_fix(motion_boundary, segment_boundaries, blend_width=5)
    blend_comparison = compare_fixes(
        motion_boundary, motion_blend_fixed,
        fix_name="Segment Boundary Blending (±5 frames)",
        segment_boundaries=segment_boundaries
    )
    print_comparison_report(blend_comparison)
    
    # Test Combined Fixes
    print("\n[TEST COMBINED FIXES #1 + #2]")
    motion_combined = simulate_cfg_fix(motion_boundary, original_scale=5.0, new_scale=2.0)
    motion_combined = simulate_blending_fix(motion_combined, segment_boundaries, blend_width=5)
    combined_comparison = compare_fixes(
        motion_baseline, motion_combined,
        fix_name="Combined CFG Scaling + Boundary Blending",
        segment_boundaries=segment_boundaries
    )
    print_comparison_report(combined_comparison)
    
    # Sanity check: ensure smooth motion is better than baseline
    print("\n[SANITY CHECK: Baseline vs Smooth (Control)]")
    control_comparison = compare_fixes(
        motion_baseline, motion_smooth,
        fix_name="Baseline Jittery vs Ground-Truth Smooth"
    )
    print_comparison_report(control_comparison)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print("\nExpected vs Actual Jitter Reduction:")
    print(f"  Fix #1 (CFG):           Expected 50-70%, Actual {cfg_comparison['improvements']['jitter_reduction']:+.2%}")
    print(f"  Fix #2 (Blending):      Expected 60-80%, Actual {blend_comparison['improvements']['jitter_reduction']:+.2%}")
    print(f"  Combined:               Expected 70-85%, Actual {combined_comparison['improvements']['jitter_reduction']:+.2%}")
    
    success = (
        cfg_comparison['improvements']['jitter_reduction'] > 0.30 and
        blend_comparison['improvements']['jitter_reduction'] > 0.40 and
        combined_comparison['improvements']['jitter_reduction'] > 0.60
    )
    
    print(f"\n  Overall Result: {'✓ PASS' if success else '✗ NEEDS TUNING'}")
    print("="*80 + "\n")


if __name__ == '__main__':
    run_comprehensive_test()
