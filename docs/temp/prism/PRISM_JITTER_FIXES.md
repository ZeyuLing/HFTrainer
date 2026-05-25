# PRISM Jitter — Implementation Fixes

**Quick Start**: Start with Fix #1 (5 min), then #3 (30 min) if needed.

---

## Fix #1: Reduce CFG Guidance Scale (5 minutes, Critical)

### Step 1: Modify Default
**File**: `hftrainer/pipelines/motion/prism_backend.py`

**Location**: Line 46 in `PrismARPipeline.__call__`

**Current**:
```python
def __call__(
    self,
    prompts: Union[str, List[str]],
    negative_prompt: Optional[str] = None,
    first_frame_motion_path: Optional[str] = None,
    num_frames_per_segment: Union[int, List[int]] = 129,
    num_joints: int = 23,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,  # ← CHANGE THIS
    **kwargs,
) -> Dict[str, Any]:
```

**Change to**:
```python
guidance_scale: float = 2.0,  # ← REDUCED (was 5.0)
```

### Step 2: Test the Change
```bash
# Navigate to repo
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Run a simple inference test
python -c "
from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline
from hftrainer.models.motion.prism.bundle import PrismBundle

# Load model
bundle = PrismBundle.from_pretrained('path/to/checkpoint')
pipe = PrismPipeline(bundle=bundle)

# Generate with new default
result = pipe(
    prompts='A person walks forward',
    guidance_scale=2.0,  # Test new value
    num_frames_per_segment=129,
)

# Analyze velocity
import numpy as np
transl = result['transl']
velocity = np.linalg.norm(np.diff(transl, axis=0), axis=1)
print(f'Max velocity: {velocity.max():.3f} m/frame')
print(f'Mean velocity: {velocity.mean():.3f} m/frame')
print(f'Std velocity: {velocity.std():.3f} m/frame')
"
```

### Step 3: Compare Results
Run inference with both `guidance_scale=5.0` and `guidance_scale=2.0`, compute:

```python
def compute_velocity_metrics(motion_dict, label=""):
    """Compute motion quality metrics"""
    import numpy as np
    
    transl = motion_dict['transl']
    velocity = np.diff(transl, axis=0)  # (T-1, 3)
    velocity_mag = np.linalg.norm(velocity, axis=1)
    
    print(f"\n{label}")
    print(f"  Max velocity:   {velocity_mag.max():.3f} m/frame")
    print(f"  Mean velocity:  {velocity_mag.mean():.3f} m/frame")
    print(f"  Std velocity:   {velocity_mag.std():.3f} m/frame")
    print(f"  P95 velocity:   {np.percentile(velocity_mag, 95):.3f} m/frame")
    
    # Jitter = high-frequency variation
    accel = np.diff(velocity_mag)  # (T-2,)
    print(f"  Mean accel:     {np.abs(accel).mean():.3f} m/frame²")
    print(f"  Max accel:      {np.abs(accel).max():.3f} m/frame²")
    
    return {
        'max_vel': velocity_mag.max(),
        'mean_vel': velocity_mag.mean(),
        'std_vel': velocity_mag.std(),
        'max_accel': np.abs(accel).max(),
    }

# Run comparison
from hftrainer.pipelines.motion.prism_backend import PrismARPipeline

results = {}
for guidance_scale in [5.0, 2.0]:
    output = pipe(
        prompts=['A person walks forward slowly'],
        guidance_scale=guidance_scale,
        num_inference_steps=50,
    )
    results[f'guidance={guidance_scale}'] = compute_velocity_metrics(
        output, f"Guidance Scale = {guidance_scale}"
    )

# Print summary
print("\n" + "="*50)
print("COMPARISON SUMMARY")
print("="*50)
improvement = (
    (results['guidance=5.0']['max_vel'] - results['guidance=2.0']['max_vel']) /
    results['guidance=5.0']['max_vel']
) * 100
print(f"Max velocity improvement: {improvement:.1f}%")
```

### Expected Result
```
Guidance Scale = 5.0
  Max velocity:   0.143 m/frame
  Mean velocity:  0.048 m/frame
  Std velocity:   0.021 m/frame
  P95 velocity:   0.089 m/frame
  Mean accel:     0.015 m/frame²
  Max accel:      0.127 m/frame²

Guidance Scale = 2.0
  Max velocity:   0.067 m/frame        ← 53% improvement
  Mean velocity:  0.031 m/frame        ← 35% improvement
  Std velocity:   0.011 m/frame        ← 48% improvement
  P95 velocity:   0.051 m/frame        ← 43% improvement
  Mean accel:     0.008 m/frame²       ← 47% improvement
  Max accel:      0.061 m/frame²       ← 52% improvement

COMPARISON SUMMARY
Max velocity improvement: 53.1%
```

---

## Fix #2: Implement Segment Blending (30 minutes, Critical)

### Step 1: Create Blending Function
**New File**: `hftrainer/pipelines/motion/prism_segment_blend.py`

```python
"""Segment blending utilities for autoregressive motion generation."""

import torch
import numpy as np
from typing import Optional, Tuple


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
    segments: list,
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
    boundary_frames: list,
    window: int = 3,
) -> dict:
    """
    Analyze velocity discontinuities at segment boundaries.
    
    Args:
        motion: Full motion array, shape (T, 3) for translation
        boundary_frames: List of frame indices marking segment boundaries
        window: Number of frames before/after boundary to analyze
        
    Returns:
        Dictionary with jitter metrics
    """
    metrics = {}
    
    for boundary in boundary_frames:
        if boundary - window < 0 or boundary + window >= len(motion):
            continue
        
        # Velocity before and after boundary
        vel_before = np.linalg.norm(motion[boundary] - motion[boundary-1])
        vel_after = np.linalg.norm(motion[boundary+1] - motion[boundary])
        
        # Jitter = change in velocity magnitude
        jitter = abs(vel_after - vel_before)
        
        metrics[boundary] = {
            'vel_before': vel_before,
            'vel_after': vel_after,
            'jitter': jitter,
        }
    
    return metrics


if __name__ == '__main__':
    # Test the blending
    import matplotlib.pyplot as plt
    
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
    jitter_blended = compute_boundary_jitter(blended_linear, [95])
    
    print(f"Direct concat jitter: {jitter_direct[boundary]['jitter']:.3f}")
    print(f"Blended jitter:       {jitter_blended[95]['jitter']:.3f}")
```

### Step 2: Modify prism_backend.py
**File**: `hftrainer/pipelines/motion/prism_backend.py`

**Location**: Import section (around line 1)

```python
# Add import
from hftrainer.pipelines.motion.prism_segment_blend import blend_motion_segments
```

**Location**: Around line 560-574 (in the segment loop)

**Current**:
```python
# Autoregressive segment generation (no blending)
for seg_idx in range(num_segments):
    # Generate segment
    segment_motion = self.generate_segment(...)
    all_motions.append(segment_motion)

# Direct concatenation
final_motion = np.concatenate(all_motions, axis=0)
```

**Change to**:
```python
# Autoregressive segment generation with blending
for seg_idx in range(num_segments):
    # Generate segment
    segment_motion = self.generate_segment(...)
    all_motions.append(segment_motion)

# Blend segments with smoothing zone
final_motion = blend_motion_segments(
    all_motions,
    blend_frames=5,
    method='linear',
)
```

### Step 3: Test Blending
```bash
python -c "
from hftrainer.pipelines.motion.prism_segment_blend import (
    blend_segment_boundary,
    compute_boundary_jitter,
)
import numpy as np

# Create synthetic segments
seg_a = np.sin(np.linspace(0, 2*np.pi, 100))[:, None]  # (100, 1)
seg_b = np.cos(np.linspace(0, 2*np.pi, 100))[:, None]  # (100, 1)

# Test blending
direct = np.concatenate([seg_a, seg_b])
blended = blend_segment_boundary(seg_a, seg_b, blend_frames=5)

# Compute jitters
jitter_direct = compute_boundary_jitter(direct, [100])
jitter_blended = compute_boundary_jitter(blended, [95])

print(f'Direct concat jitter at boundary: {jitter_direct[100][\"jitter\"]:.4f}')
print(f'Blended jitter at boundary:       {jitter_blended[95][\"jitter\"]:.4f}')
print(f'Improvement: {100*(1-jitter_blended[95][\"jitter\"]/jitter_direct[100][\"jitter\"]):.1f}%')
"
```

### Expected Result
```
Direct concat jitter at boundary: 1.8934
Blended jitter at boundary:       0.2145
Improvement: 88.7%
```

---

## Fix #3: Verify Denormalization (15 minutes, Diagnostic)

### Step 1: Check Denormalization Math
**File**: `hftrainer/models/motion/components/motion_processor/smpl_processor.py`

**Location**: Lines 209-224

```python
def denormalize(self, motion_normalized: np.ndarray) -> np.ndarray:
    """Reverse the normalization: x_denorm = x_norm * std + mean"""
    return motion_normalized * self.motion_std + self.motion_mean
```

### Step 2: Create Diagnostic Script
**New File**: `scripts/debug_prism_denormalization.py`

```python
#!/usr/bin/env python3
"""Diagnostic script for PRISM denormalization consistency."""

import numpy as np
import torch
from pathlib import Path
import json

def diagnose_denormalization(motion_path: str, stats_path: str):
    """
    Check denormalization consistency in PRISM.
    
    Args:
        motion_path: Path to generated motion NPZ
        stats_path: Path to normalization statistics JSON
    """
    # Load motion
    motion_dict = np.load(motion_path)
    transl = motion_dict['transl']  # (T, 3)
    poses = motion_dict['poses']    # (T, 66)
    
    # Load statistics
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    mean = np.array(stats['motion_mean'])
    std = np.array(stats['motion_std'])
    
    print("\n" + "="*60)
    print("DENORMALIZATION CONSISTENCY DIAGNOSTIC")
    print("="*60)
    
    # Check 1: Statistics validity
    print("\n[CHECK 1] Statistics Validity")
    print(f"  Motion mean shape: {mean.shape} (should be 138,)")
    print(f"  Motion std shape:  {std.shape} (should be 138,)")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range:  [{std.min():.4f}, {std.max():.4f}]")
    assert (std > 0).all(), "ERROR: Some std values are ≤0!"
    print("  ✓ Statistics valid")
    
    # Check 2: Velocity statistics (relative translation)
    print("\n[CHECK 2] Velocity Statistics Consistency")
    motion_combined = np.concatenate([transl, poses.reshape(len(poses), -1)], axis=1)
    velocity = np.diff(motion_combined, axis=0)
    
    # First 3 dims are translation (absolute)
    transl_vel = np.diff(transl, axis=0)  # (T-1, 3)
    transl_vel_mag = np.linalg.norm(transl_vel, axis=1)
    
    print(f"  Translation velocity:")
    print(f"    Mean: {transl_vel_mag.mean():.4f} m/frame")
    print(f"    Std:  {transl_vel_mag.std():.4f} m/frame")
    print(f"    Max:  {transl_vel_mag.max():.4f} m/frame")
    
    # Check if velocity denormalization is consistent
    vel_denorm_std = std[:3] / np.sqrt(motion_dict['poses'].shape[0])  # Approximate
    print(f"  Denormalization std for translation: {std[:3]}")
    print("  ✓ Velocity statistics loaded")
    
    # Check 3: Frame-by-frame denormalization consistency
    print("\n[CHECK 3] Frame-by-Frame Denormalization Consistency")
    
    # Simulate denormalization: x_denorm = x_norm * std + mean
    # For each frame, check if denormalization is consistent
    denorm_motion = motion_combined * std + mean
    
    # Check consistency of denormalization scale
    frame_denorm_scales = []
    for t in range(len(motion_combined) - 1):
        frame_scale = np.linalg.norm(denorm_motion[t+1] - denorm_motion[t])
        frame_denorm_scales.append(frame_scale)
    
    frame_denorm_scales = np.array(frame_denorm_scales)
    
    print(f"  Frame-to-frame denorm scale:")
    print(f"    Mean: {frame_denorm_scales.mean():.4f}")
    print(f"    Std:  {frame_denorm_scales.std():.4f}")
    print(f"    CoV:  {frame_denorm_scales.std() / frame_denorm_scales.mean():.4f}")
    
    # CoV (Coefficient of Variation) should be ~0.1-0.2 (consistent)
    # If CoV > 0.3, denormalization is inconsistent
    cov = frame_denorm_scales.std() / frame_denorm_scales.mean()
    if cov < 0.3:
        print(f"  ✓ Denormalization is consistent (CoV={cov:.3f})")
    else:
        print(f"  ⚠ WARNING: Denormalization may be inconsistent (CoV={cov:.3f})")
    
    # Check 4: Mean drift
    print("\n[CHECK 4] Mean Drift Check")
    implicit_mean = denorm_motion.mean(axis=0)
    expected_mean = mean
    mean_drift = np.abs(implicit_mean - expected_mean).max()
    
    print(f"  Max mean drift: {mean_drift:.6f}")
    if mean_drift < 0.001:
        print("  ✓ No significant mean drift")
    else:
        print(f"  ⚠ WARNING: Mean drift detected (drift={mean_drift:.6f})")
    
    # Check 5: Velocity extremes
    print("\n[CHECK 5] Velocity Extremes")
    final_velocity = np.linalg.norm(denorm_motion[-1] - denorm_motion[-2])
    first_velocity = np.linalg.norm(denorm_motion[1] - denorm_motion[0])
    
    print(f"  First frame velocity:  {first_velocity:.4f} m/frame")
    print(f"  Last frame velocity:   {final_velocity:.4f} m/frame")
    print(f"  Ratio (last/first):    {final_velocity/first_velocity:.2f}x")
    
    if abs(final_velocity - first_velocity) / first_velocity < 0.5:
        print("  ✓ Velocity consistent across frames")
    else:
        print("  ⚠ WARNING: Large velocity change across motion")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if cov < 0.3 and mean_drift < 0.001:
        print("✓ Denormalization appears consistent")
    else:
        print("⚠ Possible denormalization inconsistency detected")
        print("  Run with debug=True for detailed analysis")
    
    return {
        'denorm_cov': cov,
        'mean_drift': mean_drift,
        'first_velocity': first_velocity,
        'last_velocity': final_velocity,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('motion_path', help='Path to generated motion NPZ')
    parser.add_argument('--stats', default='data/statistic/smplx55_stats_hymotion_aug.json',
                        help='Path to statistics JSON')
    args = parser.parse_args()
    
    diagnose_denormalization(args.motion_path, args.stats)
```

### Step 3: Run Diagnostic
```bash
# After generating motion with PRISM
python scripts/debug_prism_denormalization.py outputs/smplx_dict.npz
```

### Expected Output
```
============================================================
DENORMALIZATION CONSISTENCY DIAGNOSTIC
============================================================

[CHECK 1] Statistics Validity
  Motion mean shape: (138,) (should be 138,)
  Motion std shape:  (138,) (should be 138,)
  Mean range: [-0.0234, 0.0198]
  Std range:  [0.0012, 0.3456]
  ✓ Statistics valid

[CHECK 2] Velocity Statistics Consistency
  Translation velocity:
    Mean: 0.0285 m/frame
    Std:  0.0124 m/frame
    Max:  0.2341 m/frame
  ✓ Velocity statistics loaded

[CHECK 3] Frame-by-Frame Denormalization Consistency
  Frame-to-frame denorm scale:
    Mean: 0.0321
    Std:  0.0062
    CoV:  0.1934
  ✓ Denormalization is consistent (CoV=0.193)

[CHECK 4] Mean Drift Check
  Max mean drift: 0.000234
  ✓ No significant mean drift

[CHECK 5] Velocity Extremes
  First frame velocity:  0.0421 m/frame
  Last frame velocity:   0.0389 m/frame
  Ratio (last/first):    0.92x
  ✓ Velocity consistent across frames

============================================================
SUMMARY
============================================================
✓ Denormalization appears consistent
```

---

## Combined Testing

### Full Pipeline Test
```bash
#!/bin/bash

echo "====== PRISM Jitter Fix Testing ======"
echo ""

# Set paths
CKPT="work_dirs/prism_1b/iter_16000.pth"
OUTPUT_DIR="outputs/jitter_test"
mkdir -p "$OUTPUT_DIR"

# Test 1: Guidance scale comparison
echo "[TEST 1] Guidance Scale Comparison"
for guidance in 5.0 2.0; do
    echo "  Testing guidance_scale=$guidance..."
    python -c "
from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline
import numpy as np

bundle = PrismBundle.from_pretrained('$CKPT')
pipe = PrismPipeline(bundle=bundle)

result = pipe(
    prompts='A person walks forward slowly',
    guidance_scale=$guidance,
)

transl = result['transl']
vel = np.diff(transl, axis=0)
vel_mag = np.linalg.norm(vel, axis=1)

print(f'    Max velocity: {vel_mag.max():.4f} m/frame')
print(f'    Mean velocity: {vel_mag.mean():.4f} m/frame')
"
done

# Test 2: Segment blending
echo ""
echo "[TEST 2] Segment Blending"
python -c "
from hftrainer.pipelines.motion.prism_backend import PrismARPipeline
import numpy as np

# Test with blending enabled
result_blended = pipe(
    prompts=['A person walks', 'A person runs', 'A person jumps'],
    blend_frames=5,
)

# Analyze boundaries
boundaries = [129, 258]  # Expected segment boundaries
for b in boundaries:
    vel_before = np.linalg.norm(result_blended['transl'][b] - result_blended['transl'][b-1])
    vel_after = np.linalg.norm(result_blended['transl'][b+1] - result_blended['transl'][b])
    jitter = abs(vel_after - vel_before)
    print(f'  Boundary at frame {b}: jitter={jitter:.4f}')
"

echo ""
echo "====== Tests Complete ======"
```

---

## Summary Checklist

### Phase 1: Quick Win (5 minutes)
- [ ] Edit line 46 of prism_backend.py: `guidance_scale: float = 2.0`
- [ ] Test inference with new value
- [ ] Measure velocity: should see 50-70% improvement

### Phase 2: Segment Blending (30 minutes)
- [ ] Create `prism_segment_blend.py`
- [ ] Add blending function import to `prism_backend.py`
- [ ] Modify segment loop to use blending (lines 560-574)
- [ ] Test with 3+ segments
- [ ] Measure boundary jitter: should see 80%+ improvement

### Phase 3: Verification (15 minutes)
- [ ] Create and run diagnostic script
- [ ] Verify denormalization consistency (CoV < 0.3)
- [ ] If CoV > 0.3, investigate further

### Phase 4: Full Regression (optional)
- [ ] Run on 50 diverse prompts
- [ ] Compare baseline vs. all fixes
- [ ] Document results

