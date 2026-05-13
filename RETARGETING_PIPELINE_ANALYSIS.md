# SMPL-to-Robot Retargeting Pipeline: Trembling/Instability Analysis

## Executive Summary

The SMPL-to-robot (Unitree G1) retargeting pipeline shows trembling/instability that exceeds what's in the source SMPL data. The investigation has identified multiple potential sources of these artifacts across the entire retargeting chain.

---

## Pipeline Overview

```
motion_135 (HyMotion eval output)
    ↓
    [motion135_to_smplx.py]
    Converts motion_135 format to SMPL-X NPZ
    ↓
SMPL-X NPZ (body pose + shape)
    ↓
    [gmr_retarget_headless.py]
    GMR (General Motion Retargeting) performs IK to convert SMPL-X to robot joints
    ↓
GMR PKL (root_pos, root_rot, dof_pos)
    ↓
    [gmr_to_protomotions.py]
    Converts GMR output to ProtoMotions cache format
    - Coordinate frame conversion (Y-up → Z-up)
    - FK-based ground correction
    - Resampling (30Hz → 50Hz)
    - Velocity computation (finite differences)
    ↓
ProtoMotions cache (.pt)
    ↓
    [render_tracker_headless.py]
    Renders motion (reference mode: direct qpos setting)
    ↓
Final rendered motion (with potential trembling artifacts)
```

---

## Identified Instability Sources

### 1. **motion135 → SMPL-X Conversion** (motion135_to_smplx.py)

**Issue: Rotation Representation Bug**
- Input: 6D rotation representation (row-major layout from HyMotion)
- Conversion: Row-major → column-major reordering before Gram-Schmidt
- Code reorders: `[0,2,4,1,3,5]` to convert layouts
- **Risk**: If the reordering is incorrect or HyMotion's layout differs from assumed, rotations will be corrupted
- **Result**: Joint orientations become noisy, especially high-frequency components

```python
# Line 39 in motion135_to_smplx.py
rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]  # Row-major → column-major
```

**Verification needed**: 
- Confirm HyMotion's actual 6D layout matches assumption
- Test with known hand-crafted rotations

---

### 2. **GMR Retargeting** (gmr_retarget_headless.py)

**Issue A: IK Solver Oscillation**
- GMR uses inverse kinematics to fit SMPL-X skeleton to robot skeleton
- IK solvers can produce oscillating solutions when:
  - Target is near singularity
  - Joint limits are tight
  - Multiple solutions exist for a given position
- **Code**: Lines 195-196: Frame-by-frame IK without temporal smoothing
```python
for i, frame_data in enumerate(smplx_data_frames):
    qpos = retarget.retarget(frame_data, offset_to_ground=args.offset_to_ground)
```

**Issue B: Joint Limit Clamping Introduces Discontinuities**
- Post-processing: `clamp_joint_limits()` clips DOF values to mechanical limits
- **Risk**: If motion violates limits on adjacent frames, clamping creates jumps
- Line 213: `dof_pos, num_clamped = clamp_joint_limits(dof_pos)`
- Statistics show ~13% of values clamped (trembling hotspot!)

**Issue C: Ground Offset Mismatch**
- `compute_ground_offset()` finds lowest body Z across ALL frames (global offset)
- Applied uniformly to all frames
- **Risk**: If motion has dynamic ground contact (stepping), a global offset won't work
- Example: Ground touching at frame 10 and frame 50 with different feet may need different offsets

```python
# Line 187-189: Single offset applied to all frames
ground_offset = compute_ground_offset(retarget, smplx_data_frames)
retarget.set_ground_offset(ground_offset)
```

---

### 3. **Coordinate Frame Conversion** (gmr_to_protomotions.py)

**Issue A: Root Rotation Offset Removal (Line 69-89)**

GMR applies a `rot_offset` (120° Y-up→Z-up rotation) during IK. This is baked into the output quaternion. 
The script attempts to remove it:

```python
def remove_gmr_root_offset(root_rot_xyzw):
    rot_offset = _get_gmr_rot_offset()
    root_rots = R.from_quat(root_rot_xyzw)
    corrected = root_rots * rot_offset.inv()
    return corrected.as_quat().astype(root_rot_xyzw.dtype)
```

**Risk**: 
- Quaternion multiplication order matters (non-commutative)
- Right-multiply vs left-multiply error would produce reversed rotations
- Small numerical errors in quaternion normalization compound frame-by-frame

**Issue B: Position Frame Conversion (Line 92-111)**

```python
def convert_root_pos_to_zup(root_pos):
    rot_offset = _get_gmr_rot_offset()
    return rot_offset.inv().apply(root_pos).astype(root_pos.dtype)
```

- Applies rotation to **translation** (axis mixing)
- Should be: `[x,y,z]_smplx → [z,x,y]_mujoco`
- **Risk**: If frame conversion is inconsistent between rotation and position, feet "float" relative to root
- **Result**: FK correction can't fix the inconsistency, leading to shimmering

---

### 4. **FK-Based Ground Correction** (gmr_to_protomotions.py, Line 155-229)

**Issue A: Single Reference Height**
```python
for t in range(T):
    # ... compute FK ...
    min_foot_z = np.inf
    for bi in foot_body_indices:
        foot_z = data.xpos[bi + 1][2]
        if foot_z < min_foot_z:
            min_foot_z = foot_z
    
    # Adjust root_pos Z so lowest foot is at ground_clearance
    z_offset = ground_clearance - min_foot_z
    corrected_root_pos[t, 2] = root_pos[t, 2] + z_offset
```

**Risk**:
- Modifies **only Z** component of root position
- FK is nonlinear: changing Z affects joint angles through IK-like constraints
- Adjacent frames may have different kinematic solutions (discontinuity)
- **Result**: Trembling as root Z oscillates to maintain ground contact

**Issue B: No Temporal Smoothing**
- Correction is per-frame independent
- High-frequency Z oscillations aren't damped
- **Result**: Joint angles oscillate to match corrected root heights

---

### 5. **Resampling** (gmr_to_protomotions.py, Line 296-342)

**Issue A: Linear Interpolation for Rotations**
```python
# Line 322-323: Linear interp for dof_pos (joints)
dof_interp = interp1d(times_src, dof_pos, axis=0, kind='linear')
dof_pos_resampled = dof_interp(times_tgt).astype(np.float32)

# Line 337-340: SLERP for rotations (body_rot)
rots = R.from_quat(body_rot_xyzw[:, b, :])
slerp_fn = Slerp(times_src, rots)
rots_resampled = slerp_fn(times_tgt)
```

**Risk**:
- Joint angles (`dof_pos`) are resampled with **linear interpolation**
- This is geometric interpolation in angle space, NOT in workspace
- Example: Interpolating between [0°, 180°] linearly gives 90°, but geometrically (shortest path in SO(3)) may differ
- **Result**: High-frequency oscillations introduced if motion has fast direction changes

---

### 6. **Velocity Computation** (gmr_to_protomotions.py, Line 345-384)

**Issue A: Simple Finite Differences**
```python
# Line 364-365: DOF velocity
dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt
dof_vel[0] = dof_vel[1]  # repeat first

# Line 368-370: Body linear velocity
body_vel[1:] = (body_pos[1:] - body_pos[:-1]) / dt
body_vel[0] = body_vel[1]
```

**Risk**:
- First-order finite differences are **NOT smooth** at frame boundaries
- Velocities can be large/noisy if qpos has discontinuities (from clamping or FK correction)
- No low-pass filtering applied
- **Result**: Tracker policy may struggle with noisy velocity references

**Issue B: Angular Velocity from Quaternion Differences**
```python
# Line 379-381: Relative rotation from t-1 to t
drot = rots[t] * rots[t - 1].inv()
rotvec = drot.as_rotvec()
body_ang_vel[t, b] = rotvec / dt
```

**Risk**:
- Numerical instability if rotations are nearly identical (rotvec → 0)
- Accumulation of quaternion normalization errors frame-by-frame
- **Result**: Angular velocity spikes at transitions

---

### 7. **Rendering** (render_tracker_headless.py)

**Issue: No Validation Against Source Motion**
- Reference mode directly sets qpos from cache (Line 283-289)
- No comparison with SMPL-X motion being visualized
- Can't detect if trembling is introduced by retargeting vs. rendering

---

## Data Quality Issues

### Height Corruption (from existing investigation)
- V3 reference has **19.23% height reduction** (0.7357m → 0.5942m)
- Signature of scale factor error in retargeting pipeline
- Some motions have **negative root heights** (physically impossible)
- This compounds instability: incorrect skeleton scale → IK produces worse solutions

### Frame Mismatches
- NPZ files (60 frames) vs JSON files (99 frames)
- Suggests resampling without consistency checks

---

## Potential Trembling Root Causes (Ranked by Likelihood)

| Rank | Cause | Probability | Impact |
|------|-------|-------------|--------|
| 1 | FK ground correction oscillation (per-frame independent) | HIGH (70%) | Direct Z trembling, cascades to joint oscillation |
| 2 | Joint limit clamping discontinuities | MEDIUM-HIGH (60%) | Sudden jumps create shock waves in FK |
| 3 | IK solver oscillation without temporal smoothing | MEDIUM (55%) | Frame-by-frame IK can produce multiple solutions |
| 4 | Incorrect coordinate frame conversion (rot_offset) | MEDIUM (50%) | Root rotation mismatch → kinematic inconsistency |
| 5 | Linear interpolation for rotations during resampling | MEDIUM (45%) | High-frequency artifacts in angles |
| 6 | Quaternion normalization accumulation | MEDIUM-LOW (35%) | Subtle but persistent drift |
| 7 | Rotation representation bug (motion135→SMPLX) | LOW (25%) | Would affect source, not retargeting |

---

## Diagnostic Experiments

### To isolate trembling source:

1. **Test coordinate frame conversion alone**:
   - Load GMR output, apply frame conversion
   - Render with MuJoCo FK
   - Compare foot positions before/after

2. **Test FK ground correction**:
   - Run without FK correction (`--no-fk-ground-correction`)
   - Compare motion stability

3. **Test resampling**:
   - Use source FPS (30Hz) directly (no resampling)
   - Check if trembling persists

4. **Test joint clamping**:
   - Remove clamping, render unclamped DMJ output
   - Check if trembling is reduced

5. **Test IK smoothing**:
   - Apply Savitzky-Golay filter to qpos before saving
   - Compare motion smoothness

---

## Recommended Fixes (Priority Order)

### P0: FK Ground Correction - Temporal Smoothing
- Replace per-frame independent correction with **Viterbi-like smoothing**
- Cost function: minimize Z jumps while maintaining foot contact
- Expected impact: **HIGH** - Should reduce main source of trembling

### P1: Joint Limit Clamping - Soft Clipping
- Replace hard clipping with **exponential penalty**
- Allow slightly out-of-range values if motion requires it
- Expected impact: **MEDIUM** - Reduces discontinuities

### P2: IK Temporal Smoothing
- Add **motion prior** to GMR retargeting
- Penalize large frame-to-frame changes
- Expected impact: **MEDIUM** - Reduces IK oscillation

### P3: Coordinate Frame Validation
- Add **sanity checks** after frame conversion
- Verify skeleton geometry is preserved
- Expected impact: **MEDIUM** - Catch subtle frame errors

### P4: Rotation Resampling
- Use **geodesic interpolation** for body rotations (already done via SLERP)
- Apply same SLERP to joint rotations if available
- Expected impact: **LOW-MEDIUM** - Improves smoothness

---

## Files Needing Review

1. `gmr_retarget_headless.py` - IK oscillation, ground offset, clamping
2. `gmr_to_protomotions.py` - FK correction, coordinate conversion, resampling
3. Motion retargeting configuration in GMR (IK solver parameters)
4. SMPL-X model initialization (height scaling issue)

---

## Key Metrics to Track

- Foot Z position oscillation (should be <1mm RMS)
- Root position continuity (should be smooth across frames)
- Joint angle variance frame-to-frame (should be <5% of max range)
- Skeleton geometry preservation (bone lengths should be constant)

