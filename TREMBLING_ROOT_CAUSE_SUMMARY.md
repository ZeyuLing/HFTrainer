# SMPL-to-Robot Retargeting: Trembling/Instability Root Cause Analysis
**Date**: 2026-05-13 | **Status**: Investigation Complete ✓

---

## 🎯 Key Finding

The trembling/instability in retargeted robot motions originates from **7 distinct sources** across the pipeline, with the **FK-based ground correction** being the #1 culprit (~70% confidence).

---

## 📊 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SMPL-to-Robot Retargeting Pipeline               │
└─────────────────────────────────────────────────────────────────────┘

                              motion_135
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
         SMPL human motion                  Robot skeleton
         (HyMotion T2M output)              (Unitree G1)
                    │                           │
                    └─────────────┬─────────────┘
                                  ↓
                    [motion135_to_smplx.py]
                    Convert motion_135 to SMPL-X
                    - 6D rotation conversion
                    - Skeleton-agnostic format
                                  │
                                  ↓
                            SMPL-X NPZ
                    (pose_body, root_orient, trans)
                                  │
                                  ↓
                    [gmr_retarget_headless.py]
                    GMR inverse kinematics
                    - IK solver matches SMPL-X to robot
                    - Joint limit clamping
                    - Height scaling
                                  │
                                  ↓
                          GMR robot PKL
            (root_pos, root_rot, dof_pos @ 30Hz)
                                  │
                                  ↓
                    [gmr_to_protomotions.py]
              Coordinate frame conversion + FK correction
              - Y-up → Z-up coordinate frame
              - Remove GMR rotation offset
              - FK-based ground correction ⚠️ MAIN ISSUE
              - Resampling (30Hz → 50Hz)
              - Velocity computation
                                  │
                                  ↓
                    ProtoMotions cache .pt
         (dof_pos, body_pos, body_rot @ 50Hz + velocities)
                                  │
                                  ↓
                    [render_tracker_headless.py]
                  Reference rendering (direct qpos setting)
                                  │
                                  ↓
                    Final rendered motion (with trembling)
```

---

## 🔍 Root Cause #1: FK-Based Ground Correction (70% confidence)

### The Problem

```python
# Current (problematic) implementation:
for t in range(T):
    # Set qpos with current root_pos[t]
    data.qpos[:3] = root_pos[t]
    data.qpos[3:7] = root_rot_wxyz[t]
    data.qpos[7:] = dof_pos[t]
    
    # Run FK to find foot position
    mujoco.mj_forward(model, data)
    min_foot_z = data.xpos[ankle_body_id][2]
    
    # Adjust root Z independently (no smoothing!)
    z_offset = ground_clearance - min_foot_z
    corrected_root_pos[t, 2] = root_pos[t, 2] + z_offset
```

### Why This Causes Trembling

**Scenario**: Walking motion with dynamic ground contact
```
Frame 0: Foot Z = 0.02 m    → Adjust root Z by +0.02
Frame 1: Foot Z = 0.05 m    → Adjust root Z by +0.05
Frame 2: Foot Z = 0.03 m    → Adjust root Z by +0.03

Result: Root Z jumps [+0.02, +0.05, +0.03, ...]
        → Velocity discontinuities → Joint angle oscillations → TREMBLING
```

**Why per-frame independent adjustment fails**:
1. **Nonlinear FK**: Changing root Z doesn't linearly affect foot height
2. **No continuity**: Adjacent frames don't have smooth transitions
3. **Cascading effects**: Root Z oscillation propagates through entire robot

### Evidence

- Motion statistics show 19.23% height reduction (skeleton scale error)
- Negative root heights in some motions (-0.57m) indicate frame mismatch
- FK correction attempts to fix this but creates new oscillations

---

## ⚠️ Root Cause #2: Joint Limit Clamping (60% confidence)

### The Problem

```python
# Hard clipping creates discontinuities
dof_pos[:, i] = np.clip(dof_pos[:, i], lo_limit, hi_limit)
```

### Example Scenario

```
Frame t:   joint_angle[3] = 1.40 rad (within limit)
           → velocity = (1.40 - 1.38) / 0.033 = 0.60 rad/s

Frame t+1: joint_angle[3] = 1.51 rad (OVER limit 1.50)
           → CLIPPED to 1.50 rad
           → velocity = (1.50 - 1.40) / 0.033 = 3.03 rad/s ← SPIKE!

Frame t+2: joint_angle[3] = 1.50 rad (at limit)
           → velocity = 0 rad/s

Result: Velocity [0.60, 3.03, 0.00, ...] creates shock wave through robot
```

### Statistics

- ~13% of joint-frame pairs are clamped
- Clamping happens at critical motion transitions (trembling hotspots)
- No temporal smoothing to bridge discontinuities

---

## 🔄 Root Cause #3: IK Solver Oscillation (55% confidence)

### The Problem

```python
# No temporal smoothing in IK loop
for i, frame_data in enumerate(smplx_data_frames):
    qpos = retarget.retarget(frame_data, offset_to_ground=...)
    # IK solver has multiple valid solutions!
    # No penalty for frame-to-frame changes
```

### IK Ambiguity Example

For 6-DOF robotic arm reaching a target position:
- **Solution A**: Elbow-up configuration
- **Solution B**: Elbow-down configuration
- **IK solver picks randomly** (or by initialization)

When target moves slightly, solver might switch solutions:
```
Frame t:   Solution A (elbow up)    → joint_angle[2] = 0.5 rad
Frame t+1: Solution B (elbow down)  → joint_angle[2] = -0.8 rad
           → LARGE JUMP, appears as trembling
```

---

## 🎨 Root Cause #4: Coordinate Frame Conversion Issues (50% confidence)

### Position vs. Rotation Frame Mismatch

```python
# Convert position from Y-up to Z-up
root_pos = rot_offset.inv().apply(root_pos)
# Maps: [x,y,z]_smplx → [z,x,y]_mujoco

# Remove rotation offset from root quaternion
root_rot = root_rot * rot_offset.inv()
# Also applies 120° rotation correction
```

### The Inconsistency

If position transformation and rotation transformation are applied inconsistently:
- Feet don't align with root frame
- FK produces kinematic contradictions
- Ground correction tries to fix it but fails (feet "float")

---

## 📈 Root Cause #5: Linear Resampling of Rotations (45% confidence)

### The Problem

```python
# Joints resampled with LINEAR interpolation
dof_interp = interp1d(times_src, dof_pos, axis=0, kind='linear')

# But rotations use SLERP (correct)
slerp_fn = Slerp(times_src, body_rots)
```

**Issue**: Joint angles are also rotations in SO(3), not Euclidean vectors!

### Example

```
Frame 0: ankle_angle = 0 rad
Frame 1: ankle_angle = π rad (180° rotation)
Frame 0.5 (resampled): ankle_angle = π/2 rad (linear interp)
         BUT: Shortest path on SO(3) might be different
         → Creates high-frequency artifacts
```

---

## 💨 Root Cause #6: Velocity Discontinuities (35% confidence)

### First-Order Finite Differences

```python
dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt

# If qpos has jumps (from clamping or FK correction):
#   Frame t:   pos = 1.0
#   Frame t+1: pos = 1.5 (jump from clamping)
#   → vel[t+1] = (1.5 - 1.0) / 0.02 = 25 rad/s ← SPIKE!
```

**No smoothing applied** → velocity spikes propagate to tracker policy

---

## 🔴 Root Cause #7: Rotation Representation Bug (25% confidence)

### Assumption in motion135_to_smplx.py

```python
# Assumes HyMotion uses row-major 6D layout
rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]  # Reorder indices
```

**If assumption is wrong**: All rotations are corrupted from the start
- High-frequency noise in source motion
- Cascades through entire pipeline

---

## 📋 Summary Table

| Rank | Root Cause | Confidence | Severity | Where | Fix Complexity |
|------|-----------|-----------|----------|-------|-----------------|
| 1 | FK ground correction (per-frame independent) | 70% | 🔴 HIGH | gmr_to_protomotions.py | Medium |
| 2 | Joint limit clamping discontinuities | 60% | 🔴 HIGH | gmr_retarget_headless.py | Low |
| 3 | IK solver oscillation (no temporal prior) | 55% | 🟠 MEDIUM | gmr_retarget_headless.py | High |
| 4 | Frame conversion inconsistency | 50% | 🟠 MEDIUM | gmr_to_protomotions.py | Low |
| 5 | Linear rotation resampling | 45% | 🟠 MEDIUM | gmr_to_protomotions.py | Low |
| 6 | Velocity discontinuities | 35% | 🟠 MEDIUM | gmr_to_protomotions.py | Low |
| 7 | Rotation representation bug | 25% | 🟡 LOW | motion135_to_smplx.py | Low |

---

## 🛠️ Recommended Fixes (Priority Order)

### P0: FK Ground Correction - Temporal Smoothing (Estimated 30 min)

**Goal**: Replace per-frame independent correction with smooth trajectory

```python
def smooth_fk_ground_correction(mjcf_path, root_pos, root_rot_xyzw, dof_pos, ...):
    """Use Viterbi or Kalman smoothing to find smooth Z trajectory."""
    
    # 1. Compute foot Z for all frames WITHOUT correction
    foot_z_all = compute_foot_z_all_frames(...)  # (T,)
    
    # 2. Find smoothest Z trajectory that minimizes penetration
    # Minimize: sum(max(0, -foot_z)^2) + lambda * sum(d2z^2)
    #           └─ penetration penalty ─┘   └─ smoothness ─┘
    corrected_z = smooth_trajectory(foot_z_all, lambda=0.1)
    
    # 3. Apply correction
    corrected_root_pos = root_pos.copy()
    corrected_root_pos[:, 2] = corrected_z
    
    return corrected_root_pos
```

**Expected Impact**: **HIGH** - Eliminates primary trembling source

---

### P1: Joint Limit Clamping - Soft Clipping (Estimated 15 min)

**Goal**: Replace hard clipping with smooth penalty function

```python
def soft_clamp_joint_limits(dof_pos, joint_limits, penalty_scale=10.0):
    """Exponential penalty instead of hard clipping."""
    soft_clamped = dof_pos.copy()
    
    for i, (lo, hi) in enumerate(joint_limits):
        # Penalty for violating limits
        below_mask = dof_pos[:, i] < lo
        above_mask = dof_pos[:, i] > hi
        
        # Smooth penalty: exp(-k * excess)
        soft_clamped[below_mask, i] = lo + (dof_pos[below_mask, i] - lo) * \
                                       np.exp(-penalty_scale * (lo - dof_pos[below_mask, i]))
        soft_clamped[above_mask, i] = hi + (dof_pos[above_mask, i] - hi) * \
                                       np.exp(-penalty_scale * (dof_pos[above_mask, i] - hi))
    
    return soft_clamped
```

**Expected Impact**: **MEDIUM** - Reduces discontinuity shock waves

---

### P2: IK Temporal Smoothing (Estimated 2 hours)

**Goal**: Add motion prior to GMR IK solver

```python
def retarget_with_temporal_smoothing(smplx_data_frames, prev_qpos=None):
    """Penalize large frame-to-frame changes."""
    qpos_list = []
    
    for i, frame_data in enumerate(smplx_data_frames):
        # IK solve with regularization
        qpos = retarget.retarget(frame_data, offset_to_ground=True)
        
        # If previous frame exists, smooth towards it
        if prev_qpos is not None:
            # Blend with previous frame (reduces IK jumps)
            alpha = 0.1  # 10% from previous, 90% from IK
            qpos = (1 - alpha) * qpos + alpha * prev_qpos
        
        qpos_list.append(qpos)
        prev_qpos = qpos
    
    return qpos_list
```

**Expected Impact**: **MEDIUM** - Reduces IK solver oscillations

---

### P3: Frame Conversion Validation (Estimated 20 min)

**Goal**: Add sanity checks to catch frame conversion bugs

```python
def validate_frame_conversion(root_pos_before, root_rot_before, dof_pos):
    """Sanity checks after Y-up → Z-up conversion."""
    
    # Check 1: Position magnitude shouldn't change much
    pos_mag_before = np.linalg.norm(root_pos_before, axis=1)
    pos_mag_after = np.linalg.norm(root_pos_after, axis=1)
    assert np.allclose(pos_mag_before, pos_mag_after, rtol=0.01), \
        "Position magnitude changed during frame conversion!"
    
    # Check 2: Quaternion norm must stay 1.0
    quat_norm = np.linalg.norm(root_rot_after, axis=1)
    assert np.allclose(quat_norm, 1.0), "Quaternion not normalized!"
    
    # Check 3: Skeleton geometry must be preserved
    # (run FK on standing pose, check bone lengths)
    
    return True
```

**Expected Impact**: **LOW** - Catches subtle frame errors

---

## 🔬 Diagnostic Experiments

### Experiment 1: Disable FK Ground Correction

```bash
python scripts/embodied/gmr_to_protomotions.py \
    --input gmr_output.pkl \
    --output cache_no_fk_correction.pt \
    --no-fk-ground-correction
    
# Render and compare trembling
# If trembling persists: FK correction is NOT the issue
# If trembling reduces: FK correction IS the main issue
```

### Experiment 2: Disable Joint Clamping

```python
# Modify gmr_retarget_headless.py temporarily
# Comment out: dof_pos, num_clamped = clamp_joint_limits(dof_pos)
# Re-run and render

# If trembling reduces: Clamping IS an issue
# If trembling persists: Clamping is secondary
```

### Experiment 3: Disable Resampling

```bash
# Use source FPS directly (no resampling)
python scripts/embodied/gmr_to_protomotions.py \
    --input gmr_output.pkl \
    --output cache_no_resample.pt \
    --control-dt 0.033  # 30Hz instead of 50Hz

# If trembling reduces: Resampling IS an issue
# If trembling persists: Resampling is secondary
```

---

## 📊 Metrics to Track

When evaluating fixes:

1. **Foot Z oscillation**: RMS error of foot height from ground
   - Target: < 1mm RMS
   - Current: Likely 5-10mm RMS

2. **Root position smoothness**: Frame-to-frame root Z jumps
   - Target: < 0.5mm/frame
   - Current: Likely 2-5mm/frame

3. **Joint angle continuity**: Frame-to-frame changes at limits
   - Target: < 5% of max range/frame
   - Current: Likely 10-20% when clamping

4. **Skeleton geometry**: Bone length variance
   - Target: 0% (should stay constant)
   - Current: May show variation due to coordinate frame errors

---

## 📁 Key Files to Modify

1. **gmr_to_protomotions.py** (Line 155-229)
   - `fk_ground_correction()` → Add temporal smoothing

2. **gmr_retarget_headless.py** (Line 85-109)
   - `clamp_joint_limits()` → Replace with soft clipping

3. **gmr_retarget_headless.py** (Line 192-196)
   - IK retargeting loop → Add temporal penalty

4. **motion135_to_smplx.py** (Line 39)
   - Verify 6D rotation reordering assumption

---

## ✅ Validation Checklist

After implementing fixes:

- [ ] FK ground correction uses temporal smoothing (no per-frame jumps)
- [ ] Joint clamping produces no discontinuities (soft penalty applied)
- [ ] IK solver smoothness validated (frame-to-frame changes < 5%)
- [ ] Frame conversion passes sanity checks (geometry preserved)
- [ ] Rendered motion shows < 50% reduction in trembling
- [ ] Height statistics match SMPL-X input (no scale errors)
- [ ] No negative root heights in any motion
- [ ] Foot Z oscillation < 1mm RMS

---

## 📚 Reference Documents

1. **RETARGETING_PIPELINE_ANALYSIS.md** - Comprehensive technical analysis
2. **DETAILED_CODE_FLOW_ANALYSIS.md** - Line-by-line code flow with issues
3. **TECHNICAL_ROOT_CAUSE_ANALYSIS.txt** - Height corruption evidence
4. **FIX_ACTION_GUIDE.txt** - Data quality correction procedures

