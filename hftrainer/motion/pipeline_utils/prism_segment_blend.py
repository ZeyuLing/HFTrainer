"""Segment blending utilities for autoregressive motion generation."""

import torch
import numpy as np
from typing import Optional, Tuple, List


def blend_segment_boundary(
    segment_a: np.ndarray,
    segment_b: np.ndarray,
    blend_frames: int = 5,
    method: str = 'linear',
) -> np.ndarray:
    """
    Blend two motion segments at their boundary.
    
    Args:
        segment_a: Motion from segment A, shape (T_a, D)
        segment_b: Motion from segment B, shape (T_b, D)
        blend_frames: Number of frames to blend (default 5)
        method: 'linear', 'cubic', or 'gaussian'
        
    Returns:
        Blended motion, shape (T_a - blend_frames + T_b - blend_frames + blend_frames, D)
        i.e., (T_a + T_b - blend_frames, D)
    """
    # Extract blend zones
    blend_zone_a = segment_a[-blend_frames:]  # (blend_frames, D)
    blend_zone_b = segment_b[:blend_frames]   # (blend_frames, D)
    
    # Create blend weights
    if method == 'linear':
        # Linear interpolation: 0 -> 1
        weights_a = np.linspace(1, 0, blend_frames)[:, np.newaxis]  # (blend_frames, 1)
        weights_b = np.linspace(0, 1, blend_frames)[:, np.newaxis]  # (blend_frames, 1)
    elif method == 'cubic':
        # Cubic Hermite: smoother transitions
        t = np.linspace(0, 1, blend_frames)
        weights_b = 3 * t**2 - 2 * t**3  # Hermite basis
        weights_a = 1 - weights_b
        weights_a = weights_a[:, np.newaxis]
        weights_b = weights_b[:, np.newaxis]
    elif method == 'gaussian':
        # Gaussian envelope: weighted by distance from center
        t = np.linspace(-2, 2, blend_frames)
        gaussian = np.exp(-0.5 * t**2)
        weights_a = gaussian / (gaussian.max())
        weights_b = 1 - weights_a
        weights_a = weights_a[:, np.newaxis]
        weights_b = weights_b[:, np.newaxis]
    else:
        raise ValueError(f"Unknown blend method: {method}")
    
    # Blend zones
    blended = weights_a * blend_zone_a + weights_b * blend_zone_b  # (blend_frames, D)
    
    # Concatenate
    result = np.concatenate([
        segment_a[:-blend_frames],  # Keep start of A (without blend zone)
        blended,                     # Blended zone
        segment_b[blend_frames:],    # Keep end of B (without blend zone)
    ], axis=0)
    
    return result


def blend_motion_segments(
    segments: List[np.ndarray],
    blend_frames: int = 5,
    method: str = 'linear',
) -> np.ndarray:
    """
    Blend multiple motion segments together.
    
    Args:
        segments: List of motion arrays, each shape (T, D)
        blend_frames: Frames to blend at each boundary
        method: Blending method
        
    Returns:
        Blended motion as single array, shape (T_total - (N-1)*blend_frames, D)
    """
    if len(segments) == 1:
        return segments[0]
    
    result = segments[0]
    for seg in segments[1:]:
        result = blend_segment_boundary(
            result, seg,
            blend_frames=blend_frames,
            method=method,
        )
    
    return result


def compute_boundary_jitter(
    motion: np.ndarray,
    boundary_frames: List[int],
    window: int = 3,
) -> dict:
    """
    Analyze velocity discontinuities at segment boundaries.
    
    Args:
        motion: Full motion array, shape (T, 3) for translation or (T, D) for full motion
        boundary_frames: List of frame indices marking segment boundaries
        window: Number of frames before/after boundary to analyze
        
    Returns:
        Dictionary with jitter metrics at each boundary
    """
    metrics = {}
    
    # Use translation if available (first 3 dims), else use full motion
    motion_to_analyze = motion[:, :3] if motion.shape[1] >= 3 else motion
    
    for boundary in boundary_frames:
        if boundary - window < 0 or boundary + window >= len(motion_to_analyze):
            continue
        
        # Velocity before and after boundary
        vel_before = np.linalg.norm(motion_to_analyze[boundary] - motion_to_analyze[boundary-1])
        vel_after = np.linalg.norm(motion_to_analyze[boundary+1] - motion_to_analyze[boundary])
        
        # Jitter = change in velocity magnitude
        jitter = abs(vel_after - vel_before)
        
        metrics[boundary] = {
            'vel_before': vel_before,
            'vel_after': vel_after,
            'jitter': jitter,
        }
    
    return metrics


def compute_velocity_profile(
    motion: np.ndarray,
    window: int = 1,
) -> dict:
    """
    Compute comprehensive velocity statistics for motion.
    
    Args:
        motion: Motion array, shape (T, D)
        window: Averaging window for velocity computation
        
    Returns:
        Dictionary with velocity metrics
    """
    # Use translation (first 3 dims)
    motion_to_analyze = motion[:, :3] if motion.shape[1] >= 3 else motion
    
    # Frame-to-frame displacement
    displacement = np.diff(motion_to_analyze, axis=0)
    velocity = np.linalg.norm(displacement, axis=1)
    
    # Acceleration (second derivative)
    acceleration = np.diff(velocity)
    
    return {
        'max_velocity': float(velocity.max()),
        'min_velocity': float(velocity.min()),
        'mean_velocity': float(velocity.mean()),
        'std_velocity': float(velocity.std()),
        'p95_velocity': float(np.percentile(velocity, 95)),
        'max_acceleration': float(np.abs(acceleration).max()),
        'mean_acceleration': float(np.abs(acceleration).mean()),
        'jitter_coefficient': float(velocity.std() / (velocity.mean() + 1e-6)),
    }


if __name__ == '__main__':
    # Test the blending
    print("Testing segment blending...")
    
    # Create synthetic segments
    t = np.linspace(0, 2*np.pi, 100)
    seg_a = np.column_stack([np.sin(t), np.cos(t), t/50])  # (100, 3)
    seg_b = np.column_stack([np.sin(t+np.pi), np.cos(t+np.pi), (t+100)/50])  # (100, 3)
    
    # Blend without smoothing (direct concat)
    direct = np.concatenate([seg_a, seg_b], axis=0)
    
    # Blend with smoothing
    blended_linear = blend_segment_boundary(seg_a, seg_b, blend_frames=5, method='linear')
    blended_cubic = blend_segment_boundary(seg_a, seg_b, blend_frames=5, method='cubic')
    blended_gaussian = blend_segment_boundary(seg_a, seg_b, blend_frames=5, method='gaussian')
    
    # Compute jitters
    boundary = 100
    jitter_direct = compute_boundary_jitter(direct, [boundary])
    jitter_linear = compute_boundary_jitter(blended_linear, [95])
    jitter_cubic = compute_boundary_jitter(blended_cubic, [95])
    jitter_gaussian = compute_boundary_jitter(blended_gaussian, [95])
    
    print(f"Direct concat jitter:  {jitter_direct[boundary]['jitter']:.3f}")
    print(f"Linear blend jitter:   {jitter_linear[95]['jitter']:.3f}")
    print(f"Cubic blend jitter:    {jitter_cubic[95]['jitter']:.3f}")
    print(f"Gaussian blend jitter: {jitter_gaussian[95]['jitter']:.3f}")
    
    # Compute velocity profiles
    profile_direct = compute_velocity_profile(direct)
    profile_blended = compute_velocity_profile(blended_linear)
    
    print(f"\nDirect motion - Max velocity: {profile_direct['max_velocity']:.3f}, Jitter coeff: {profile_direct['jitter_coefficient']:.3f}")
    print(f"Blended motion - Max velocity: {profile_blended['max_velocity']:.3f}, Jitter coeff: {profile_blended['jitter_coefficient']:.3f}")
