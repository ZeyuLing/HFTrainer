# Trembling/Instability Issue: Quick Reference Guide

## The Problem
Retargeted robot motions show **trembling/instability beyond what exists in source SMPL data**. This manifests as:
- Jittery pelvis movement (especially in Z)
- Rapid joint angle oscillations
- Unnatural acceleration spikes
- Visible shaking in reference renders (FK-only)

## Root Causes (Priority Order)

| Issue | Location | Severity | Symptom | Quick Test |
|-------|----------|----------|---------|-----------|
| **Frame-by-frame FK ground correction** | `gmr_to_protomotions.py:155-229` | 🔴 CRITICAL | Pelvis Z jitter | Disable `--no-fk-ground-correction` |
| **Finite differences velocity** | `gmr_to_protomotions.py:345-384` | 🔴 CRITICAL | Velocity spikes | Check `dof_vel` for jumps |
| **Joint limit hard clamping** | `gmr_retarget_headless.py:85-109` | 🟠 IMPORTANT | Trembling near limits | Check if joints hit limits |
| Per-frame foot grounding (GMR) | `gmr_retarget_headless.py:112-140` | 🟡 SECONDARY | Foot position discontinuities | Compare with v0 GMR |

## Quick Diagnostic Checklist

### ✅ Step 1: Is it in the cache?
```bash
# Render reference mode (pure FK, no ONNX)
python scripts/embodied/render_tracker_headless.py \
    --motion output/embodied_t2m_v4/data/caches/pipeline_000.pt \
    --mode reference \
    --output-dir /tmp/ref_test
```
- **Trembling visible** → Problem in cache (Issues #3.3, #3.5)
- **Smooth** → Problem in ONNX tracking or physics simulation

### ✅ Step 2: FK correction to blame?
```bash
# Retarget WITHOUT FK ground correction
python scripts/embodied/pipeline_motion_to_robot.py \
    --input work_dirs/motion_135/00000.npz \
    --output /tmp/no_fk.pt \
    --no-fk-ground-correction

# Compare cache files
python << 'PYEOF'
import torch
cache_with = torch.load('motion_cache_with_fk.pt', weights_only=False)
cache_without = torch.load('/tmp/no_fk.pt', weights_only=False)

# Check root Z
z_with = cache_with['body_pos'][:, 0, 2]
z_without = cache_without['body_pos'][:, 0, 2]

print(f"Root Z with FK correction:    std={z_with.std():.6f}, range={z_with.max()-z_with.min():.6f}")
print(f"Root Z without FK correction: std={z_without.std():.6f}, range={z_without.max()-z_without.min():.6f}")

# If std is MUCH lower without correction, FK correction is the issue
PYEOF
```

### ✅ Step 3: Velocity computation to blame?
```python
import torch
import numpy as np

cache = torch.load('pipeline_000.pt', weights_only=False)
dof_vel = cache['dof_vel'].numpy()  # (T, 29)

# Check for velocity discontinuities
vel_jumps = np.abs(np.diff(dof_vel, axis=0))
max_jump = np.max(vel_jumps)
mean_jump = np.mean(vel_jumps)

print(f"Max velocity jump: {max_jump:.6f}")
print(f"Mean velocity jump: {mean_jump:.6f}")
print(f"Std of jumps: {np.std(vel_jumps):.6f}")

# If mean >> std, there are specific large jumps (discontinuities)
# If std is large relative to mean, velocity is noisy
if max_jump > 0.1:
    print("⚠️  Large velocity jumps detected - likely finite difference artifacts")
```

### ✅ Step 4: Joint limits to blame?
```python
import torch
import numpy as np

cache = torch.load('pipeline_000.pt', weights_only=False)
dof_pos = cache['dof_pos'].numpy()  # (T, 29)

# G1 joint limits (from gmr_retarget_headless.py:52-82)
limits = {
    # Check a few key joints
    'l_knee': (-0.087267, 2.8798),      # index 3
    'r_knee': (-0.087267, 2.8798),      # index 9
    'l_ankle_pitch': (-0.87267, 0.5236), # index 4
}

for name, (lo, hi) in limits.items():
    idx_map = {'l_knee': 3, 'r_knee': 9, 'l_ankle_pitch': 4}
    idx = idx_map[name]
    near_lo = np.sum(dof_pos[:, idx] < lo + 0.05)
    near_hi = np.sum(dof_pos[:, idx] > hi - 0.05)
    print(f"{name}: near_lo={near_lo}, near_hi={near_hi}")
```

## Quick Fixes (Try in Order)

### Fix #1: Disable FK Ground Correction (Fastest Test)
```bash
python scripts/embodied/pipeline_motion_to_robot.py \
    --input work_dirs/motion_135/00000.npz \
    --output /tmp/test_no_fk.pt \
    --no-fk-ground-correction
```
**Expected**: If trembling reduces significantly, FK correction is the issue.

### Fix #2: Replace Finite Differences with Smoothing (Code Change)
In `gmr_to_protomotions.py`, replace the `compute_velocities()` function with `compute_velocities_smooth()` from `RETARGETING_ANALYSIS.md` using Savitzky-Golay filtering.

### Fix #3: Make FK Correction Smooth (Code Change)
In `gmr_to_protomotions.py`, replace `fk_ground_correction()` with `fk_ground_correction_smooth()` from `RETARGETING_ANALYSIS.md`.

## File Map

| File | Purpose | Issue |
|------|---------|-------|
| `motion135_to_smplx.py` | Rot6D → SMPL-X | #1.2 (small) |
| `gmr_retarget_headless.py` | SMPL-X → GMR | #2.1, #2.2, #2.3 |
| `gmr_to_protomotions.py` | GMR → ProtoMotions | **#3.3, #3.5** ⚠️ MAIN |
| `render_tracker_headless.py` | Render reference | Shows cache quality |
| `run_tracker_export.py` | ONNX simulation | Can amplify cache issues |
| `convert_cache_to_json.py` | Cache → JSON | No issues |

## Key Parameters

```python
# In gmr_to_protomotions.py:

# Line 155: FK ground correction function
fk_ground_correction(
    mjcf_path,
    root_pos,           # Will be modified in-place
    root_rot_xyzw,
    dof_pos,
    foot_body_indices=[7, 13],  # left/right ankle roll
    ground_clearance=0.0
)

# Line 296: Resampling
resample_motion(times_src, dof_pos, body_pos, body_rot_xyzw, control_dt=0.02)
# control_dt = 0.02 → 50 Hz

# Line 345: Velocity computation
compute_velocities(dof_pos, body_pos, body_rot_xyzw, dt=0.02)
# Uses simple finite differences, no smoothing
```

## Severity Ranking

1. 🔴 **CRITICAL**: FK ground correction + Finite diff velocities
   - Directly visible as trembling in reference render
   - Affects all motions
   
2. 🟠 **IMPORTANT**: Joint limit clamping
   - Only affects motions where joints reach limits
   - Visible as sudden jerks

3. 🟡 **SECONDARY**: Other coordinate transforms, resampling
   - Subtle effects
   - May accumulate

## Next Steps

1. **Read** `RETARGETING_ANALYSIS.md` for detailed technical analysis
2. **Test** the diagnostic steps above
3. **Implement** Fix #1 (disable FK correction) first to isolate the issue
4. **Implement** Fixes #2-#3 based on diagnostic results
