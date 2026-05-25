# Standard Benchmark Evaluation Settings for Motion Completion/Control Papers

**Analysis Date**: May 22, 2026  
**Source**: HyMotion M2M v2 evaluation framework + KIMODO/UMO/MotionLab/SOAR/StableMotion reference implementations  
**Framework Used**: M2M v2 (HunyuanMotion MMDiT + VACE conditioning, 135-dim representation)

---

## Executive Summary

This document consolidates the **standard evaluation settings** used by state-of-the-art motion completion/control papers. The information is extracted from:

1. **HyMotion M2M v2 (Internal)**: Tasks E1-E16 with settings A-F for each
2. **KIMODO (NVIDIA, 2024)**: Two-stage imputation-based completion
3. **UMO (Brown/MIT/Meta, 2024)**: Frame-level meta-operation (P/G/E) approach  
4. **MotionLab (SUTD/Lightspeed, 2025)**: Task instruction modulation + unified gen/edit
5. **SOAR (NUS/Alibaba/MSR, 2025)**: Diffusion post-training via on-policy rollout
6. **StableMotion (SFU, 2025)**: Quality-indicator-driven cleanup

---

## Part 1: Temporal Completion Tasks (In-Betweening, Prediction, Keyframe Infilling)

### Task Definition
**Goal**: Given start/end frames or sparse keyframes, generate smooth in-between motions that transition naturally.

### 1.1 Motion In-Betweening (MIB)

#### Standard Protocol (E2 in M2M v2)

| Setting | Name | Description | Keep-Start | Keep-End | Mask Pattern |
|---------|------|-------------|-----------|----------|--------------|
| **A** | start_1f | Keep first 1 frame only | 1 frame | 0 frames | `[0, 1, 1, ..., 1]` |
| **B** | end_1f | Keep last 1 frame only | 0 frames | 1 frame | `[1, 1, ..., 1, 0]` |
| **C** | both_1f | Keep first + last 1 frame | 1 frame | 1 frame | `[0, 1, ..., 1, 0]` |
| **D** | pre20 | Keep first 20% → predict rest | ceil(0.20×T) frames | 0 frames | First 20% known |
| **E** | post20 | Keep last 20% → predict rest | 0 frames | ceil(0.20×T) frames | Last 20% known |
| **F** | mid60 | Keep first + last 20% → predict middle | ceil(0.20×T) | ceil(0.20×T) | Front/back 20% known |

#### Key Metrics
```
Primary:
  - MPJPE (masked): Position error on generated frames only
    Calculation: √(1/N_gen Σ ||pred_pos[i] - gt_pos[i]||²) where i ∈ generated frames
    Unit: meters
  - MPJPE (unmasked): Position error on entire sequence
  - Boundary acceleration jump: ||accel[keep_end_frame] - accel[gen_start_frame]||
    Measures discontinuity at mask transition
  
Secondary:
  - Jitter (m/s³): Mean jerk = mean(||d³x/dt³||) across all joints
    Computed as: diff3 = x[t+3] - 3x[t+2] + 3x[t+1] - x[t], jerk = diff3/dt³
  - Foot skating ratio: % of frames where foot velocity > 0.5 cm/frame while foot contact > 0.8
```

#### Data Statistics (HyMotion M2M v2)
- **Dataset**: 220 motions from Private held-out pool
- **Stratification**: By action category + pelvis-speed bucket  
- **Captions**: Rewritten by Qwen3-30B-A3B-GRPO (12-20 words, "A person..." format)
- **Motion duration**: Variable (T ∈ [10, 240] frames @ 30 fps)
- **Test set**: Fixed at eval time (not random split per run)

#### KIMODO Settings (for comparison)
```
Imputation-based ("Phase 2"):
- Sample random position constraints from {ankle, wrist, spine, pelvis}
- Set contact_frames at geometric(p=0.1) rate
- Impute position + set binary mask
- Inference: hard-replace noisy_x[t, dims] every denoise step
- Metrics: Position error (mm), rotation error (deg), blend smoothness
```

#### UMO Settings (for comparison)
```
Frame-level meta-op approach:
- Source motion: [first_frame, 0, 0, ..., 0, last_frame]  
- Meta-ops: [P, G, G, ..., G, P] (Preserve/Generate)
- Inference: element-wise add source + meta-op embedding to input embedding
- No hard replacement — soft constraint via context fusion
- Metrics: Position error, trajectory smoothness, temporal consistency
```

---

### 1.2 Keyframe Interpolation / Sparse Keyframe In-filling

#### Standard Protocol (E3 in M2M v2)

| Setting | Name | Interval | Anchors @ 30fps | Density |
|---------|------|----------|-----------------|---------|
| **A** | every_5f | 5 frames | 6 fps | Very dense |
| **B** | every_10f | 10 frames | 3 fps | Dense |
| **C** | every_15f | 15 frames | 2 fps | Medium |
| **D** | every_30f | 30 frames (1s) | 1 fps | Sparse (standard) |
| **E** | every_60f | 60 frames (2s) | 0.5 fps | Very sparse |
| **F** | adaptive | Peak detection on acceleration | ~1 per second | Adaptive |

**Adaptive Keyframe Selection Algorithm**:
```python
# Peak detection on joint acceleration
accel = np.diff(positions, n=2)  # T-2 accel values per joint
joint_accel_norm = np.linalg.norm(accel, axis=-1)  # (T-2, 22)
per_frame_accel = joint_accel_norm.max(axis=-1)  # (T-2,) max across joints

# Find peaks with 30-frame minimum gap
peaks = find_peaks(per_frame_accel, distance=30)
keyframe_times = sorted(peaks[0].tolist() + [0, T-1])  # Always include first + last
```

#### Key Metrics (E3)
```
Same as MIB:
  - MPJPE (masked): Error on interpolated frames
  - MPJPE (unmasked): Overall error
  - Jitter: High-frequency noise
  - Foot skating: Ground contact violation
  
Specific to keyframe:
  - Per-anchor consistency: MPJPE at anchor frames should ≈ 0
```

#### Data Statistics
- **Dataset**: Same 220-motion pool as E2 (eval_e2_inbetween_v2_rewritten.json)
- **Anchor strategy**: Every_30f is the standard for publications
- **Typical motion length**: 100-240 frames → 3-8 keyframes per motion

---

### 1.3 Motion Prediction / Tail Prediction

#### Standard Protocol (E16 in M2M v2)

| Setting | Name | Keep-End | Generate | Mask Pattern |
|---------|------|----------|----------|--------------|
| **A** | 1f_anchor | Last 1 frame | Rest | `[1, 1, ..., 1, 0]` |
| **B** | 5f_anchor | Last 5 frames | Rest | Last 5 known |
| **C** | 10f_anchor | Last 10 frames | Rest | Last 10 known |

**Key difference from MIB**: In prediction, we keep the **END** frame (tail anchor) and predict **backwards** to fill the sequence. In MIB, we keep both ends.

#### Data Statistics
- **Dataset**: eval_e16_tail_prediction.json (different from E2/E7)
- **Typical setup**: Condition on last 5-10 frames, generate the prefix

---

## Part 2: Spatial Editing Tasks

### 2.1 End-Effector Position Constraint (E4)

#### Standard Protocol

| Setting | EE Joints | Frame Mode | Frame Count | Mask Type |
|---------|-----------|-----------|------------|-----------|
| **A** | 1 random | Random | ~10 frames | Single EE, sparse |
| **B** | 1 random | Random | 10% of T | Single EE, medium |
| **C** | 2 random | Random | ~10 frames | Dual EE, sparse |
| **D** | 2 random | Random | 10% of T | Dual EE, medium |
| **E** | All 4 (l_ankle, r_ankle, l_wrist, r_wrist) | Random | ~10 frames | All limb EE, sparse |
| **F** | All 4 | Random | 20% of T | All limb EE, dense |

#### Constraint Implementation (M2M v2)

**Mask Layer**:
- Dimension: 198-dim (135 rotation + 63 position channels)
- Position channels: dims [135:198] = 21 joints × 3 dims (XYZ relative to pelvis)
- Pelvis excluded from position channels (only 3D absolute translation at dims [0:3])

**Per-joint constraint**:
```python
for constraint_frame in constraint_frames:
    for joint_name in joint_names:
        j = JOINT_NAME_TO_IDX[joint_name]  # 0-21, 0=pelvis
        if j == 0:
            continue  # Pelvis has no position channel (redundant with translation)
        # Lock the position channel for this joint
        pos_start = 135 + (j - 1) * 3
        mask[constraint_frame, pos_start:pos_start+3] = 0  # Keep (condition)
```

#### Key Metrics (E4)

```
Primary:
  - EE error (mean, median, p95, max): Distance between predicted EE position and GT
    Calculation: ||FK(pred_rot)[EE_joint] - GT_position[EE_joint]||
    Unit: meters
  - EE hit rate @5cm: % of constraint frames where EE error < 0.05m
  - EE hit rate @10cm: % of constraint frames where EE error < 0.10m

Secondary:
  - Jitter: High-frequency noise in EE trajectory
  - MPJPE (masked): Error on constrained frames
```

#### KIMODO Settings (for comparison)
```
Direct position imputation:
- Impute world-space wrist position at random frames
- Impute ankle contact status (binary)
- Position constraint applied via hard replacement each denoise step
- Accuracy: position-exact, rotation inferred via FK consistency
```

#### MotionLab Settings (for comparison)
```
Hint-modality trajectory control:
- Trajectory sequence as additional modality input
- 5 modality path: (source, target, text, trajectory, style)
- Trajectory error metric: L2 distance on root XZ + root Y separately
```

---

### 2.2 Part-Level / Joint-Level Editing (E4-D)

#### Standard Protocol

| Setting | Joints | Op | Description |
|---------|--------|----|----|
| **A** | Upper body (spine + arms) | Replace | Replace upper body with random pose from same motion |
| **B** | Lower body (legs) | Replace | Replace lower body with random pose |
| **C** | Left arm | Replace | Isolated left arm replacement |
| **D** | Right arm | Replace | Isolated right arm replacement |

**Mask construction**:
```python
# Joint groups (from universal_mask)
UPPER_BODY = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]  # spine, head, arms
LOWER_BODY = [6, 7, 8, 9, 10, 11]  # pelvis, legs
LEFT_ARM = [12, 13, 14]
RIGHT_ARM = [15, 16, 17]

# For each group, create binary mask (0=keep from source, 1=generate)
grid = np.ones((T, 23))  # 23-group space
grid[:, joint_groups] = 0  # Mark these as condition
mask = expand_grid_to_mask(grid)  # Expand to 135-dim
```

#### Key Metrics
- MPJPE (edited): Position error on edited joints only
- Boundary smoothness: Acceleration continuity at joint-group boundaries
- Visual naturalness: Per-group consistency with source motion dynamics

---

### 2.3 Pose Guidance / First-Frame Continuation

#### Standard Protocol (E7 in M2M v2)

| Setting | Description | Mask | Metrics |
|---------|-------------|------|---------|
| **A** | Keep frame 0 + text, generate rest | `[0, 1, 1, ..., 1]` | MPJPE (unmasked), Jitter, Foot skating |

**Data**: eval_e7_first_frame.json (requires caption)

#### UMO Equivalent: Frame-level Keypose

```python
# UMO uses [preserve] on frame 0 only
source_motion = [frame_0_pose, 0, 0, ..., 0]
meta_ops = [P, G, G, ..., G]
```

---

## Part 3: Trajectory Control Tasks

### 3.1 Root Trajectory Following (E5)

#### Standard Protocol (M2M v2)

| Setting | Axes | Mode | Interval | Description |
|---------|------|------|----------|-------------|
| **A** | XZ only | Dense | Every frame | Full dense trajectory (planar) |
| **B** | XZ only | Sparse | Every 30 frames | Sparse waypoints (planar) |
| **C** | XZ + heading | Dense+heading | Every frame | Dense XZ + pelvis yaw constraint |
| **D** | XYZ | Dense | Every frame | Full 3D trajectory (with height/Y) |
| **E** | XYZ | Sparse | Every 30 frames | Sparse 3D waypoints |
| **F** | XYZ + heading | Dense+heading | Every frame | Full 3D + pelvis yaw |

#### Constraint Implementation

**XZ-only (training-aligned, recommended)**:
```python
# Constrain only X and Z translation, leave Y (height) free for natural dynamics
mask = np.ones((T, 135))
# Dims 0=X, 1=Y, 2=Z
for t in frames:  # frames = range(T) for dense, range(0,T,30)+[T-1] for sparse
    mask[t, 0] = 0  # X keep
    mask[t, 2] = 0  # Z keep
    # mask[t, 1] = 1  # Y generate (height free)
```

**XYZ (explicit height control)**:
```python
# All three translation dims constrained
for t in frames:
    mask[t, 0:3] = 0  # X, Y, Z all keep
```

**Heading only**:
```python
# Constrain pelvis rotation (6D at dims 3:9)
for t in frames:
    mask[t, 3:9] = 0
```

#### Key Metrics (E5)

```
Primary:
  - Trajectory ADE (Average Displacement Error):
    ADE = (1/T) Σ_t ||pred_rootXZ[t] - gt_rootXZ[t]||
    Unit: meters
  - Trajectory FDE (Final Displacement Error):
    FDE = ||pred_rootXZ[T-1] - gt_rootXZ[T-1]||
    Unit: meters

Secondary:
  - Foot skating ratio: Same as MIB
  - Jitter: Same as MIB
```

#### KIMODO Settings
```
Trajectory constraints:
- Impute root position at waypoint frames
- Maintain smooth root via post-process filtering
- Implicit heading from surrounding context
- Metrics: ADE, FDE in mm
```

#### MotionLab Settings
```
Trajectory as dedicated modality:
- Trajectory sequence (T, 3) as hint input
- Aligned 1D RoPE ensures time correspondence
- End-to-end learning with trajectory loss
- Trajectory error @ root: 0.0286m (state-of-art)
```

---

### 3.2 Foot Ground Contact / Sticky Foot (E6)

#### Standard Protocol

| Setting | Constraint | Axes | Description |
|---------|-----------|------|-------------|
| **A** | Position (198-dim) | XYZ | Lock ankle position at GT contact frames |

**Contact detection**:
```python
# Use GT motion to find contact frames
# Method 1: Foot Y velocity near-zero (< 0.1 m/s)
# Method 2: Foot-ground distance < 0.05m
# Method 3: Explicit contact label from motion capture

contact_frames = np.where(gt_foot_velocity < 0.1)[0]
```

**Mask construction**:
```python
l_ankle_pos_start = 135 + 6 * 3  # dim 153
r_ankle_pos_start = 135 + 7 * 3  # dim 156

mask = np.ones((T, 198))
for t in contact_frames:
    # Lock ankle XYZ position
    mask[t, l_ankle_pos_start:l_ankle_pos_start+3] = 0
    mask[t, r_ankle_pos_start:r_ankle_pos_start+3] = 0
    # Allow rotation to adjust freely (dims 3:135 remain 1)
```

#### Key Metrics

```
Primary:
  - Foot penetration: % of frames where foot Y < 0 (below ground)
  - Foot float: Average distance from ground when contact frame has foot Y > 0.02m
  - Foot skating: Already defined (velocity while contacting)

Calculation:
  - Penetration = mean(ReLU(-foot_min_y))  # Average distance below ground
  - Float = mean(foot_y[contact_frames]) where foot_y > 0.02m
  - Skating: Velocity-based (see E2/E5 definitions)
```

---

## Part 4: Loop / Continuation Tasks

### 4.1 Loop Animation (E8)

#### Standard Protocol

| Setting | Mode | End constraint | Anchor | Duration | Model type |
|---------|------|-----------------|--------|----------|-----------|
| **A** | Pure loop | Frame 0 == Frame T-1 | First pose | Natural T | Caption-aware |
| **D** | Loop completion | Full GT + N_append | Return to Frame 0 | GT + transition | Uncond only |

**Setting A (Pure Loop)**:
```python
# Classic loop: first and last frames must match
src_motion = np.zeros((T, 135))
src_mask = np.zeros((T, 135))

# Set first and last frames to the same pose
src_motion[0, :] = gt_motion[0, :]
src_motion[-1, :] = gt_motion[0, :]  # Last frame = first frame pose
src_mask[0, :] = 0  # Known
src_mask[-1, :] = 0  # Known
src_mask[1:-1, :] = 1  # Generate middle

# T_loop = sample.num_frames (adaptive, no fixed length)
```

**Setting D (Loop Completion)**:
```python
# Adaptive transition: given full GT motion, append N_append frames to return to GT[0]
# N_append computed by:
#   1. Root displacement: ||motion[-1, 0:3] - motion[0, 0:3]||
#   2. Joint position change: mean(||FK(rot[-1]) - FK(rot[0])||)
#   3. Joint angle change: mean(||rot[-1] - rot[0]||)
# N_append = compute_transition_length(metric1, metric2, metric3)
# Clamped: 30 <= N_append <= 150 frames

# Mask:
T_total = T_gt + N_append
src_mask = np.zeros((T_total, 135))
src_mask[T_gt:T_gt+N_append-1] = 1  # Generate appended frames
src_mask[-1] = 0  # Last frame = first frame (loop target)
```

#### Key Metrics (E8)

```
Primary:
  - Loop position error: MPJPE between frame[0] and frame[-1] predicted poses
  - Loop velocity error: Velocity difference at frame 0 and frame -1 boundary
    Calculation: ||vel[-1] - vel[0]|| where vel[t] = (pos[t] - pos[t-1]) / dt

Secondary:
  - Jitter: Same definition
  - Boundary acceleration jump: Same as MIB
```

---

## Part 5: Quality/Repair Tasks

### 5.1 Motion Repair / Detection + Inpainting (E9)

#### Standard Protocol

| Setting | Mask Source | Repair Mode | Threshold | Description |
|---------|-------------|------------|-----------|-------------|
| **E** | Union | Inpaint + skip_last | — | Union of adaptive + QC invalid masks |
| **D_ada_t005** | Adaptive (MoGenDIT) | Ada-denoise (2-stage) | 0.05 (abs) | Conservative, minimal repair |
| **D_ada_t020** | Adaptive | Ada-denoise | 0.20 (abs) | Aggressive, more repair |
| **D_strict_d2_b3_bsmooth** | Adaptive (strict) | Single-shot inpaint | — | Recommended baseline |

#### E9 Mask Generation

**Union Mask (E_union_mask)**:
```python
# Combine two mask sources:
# 1. MoGenDIT adaptive mask (change-based detection)
# 2. QC checker invalid_mask (anatomical defects)

mask_adaptive = mogendit_detect(motion_lq)  # Change > threshold
mask_qc = qc_checker.invalid_mask(motion_lq)  # Defects
mask_union = mask_adaptive | mask_qc  # Element-wise OR

# Apply spatial/temporal dilation on QC half
mask_union = temporal_dilate(mask_union, d=2)  # Expand in time
mask_union = spatial_dilate_neighbors(mask_union)  # Kinematic neighbor propagate
```

**Ada-Denoise (D_ada_denoise_t005)**:
```python
# Stage 1: Full regeneration via SDEdit from LQ
# τ=0.5, clean_motion=LQ → LQ manifold projection
x_stage1 = sdedit_from_lq(motion_lq, tau=0.5, T_steps=50)

# Stage 2: Change detection
change = ||normalize(motion_lq) - normalize(x_stage1)||  per joint
keep_mask = change <= 0.05  (absolute threshold)

# Stage 3: Inpaint with computed keep_mask
mask_stage3 = ~keep_mask  (invert: 0=keep, 1=generate)
x_final = inpaint(motion_lq, mask_stage3, sdedit_tau=0.0)
```

#### Key Metrics (E9)

```
Primary:
  - QC pass rate: % of frames passing quality checks after repair
    Checks: Penetration, contact, skeleton constraint, jitter, skating, etc.
  - Jitter (m/s³): Post-repair high-frequency noise

Secondary:
  - Boundary smoothness: Repair region boundary acceleration
  - FK consistency: |diff between rotation FK and position channels|
```

#### StableMotion Settings (for comparison)
```
Quality-indicator-driven cleanup:
- Quality label as additional channel (0=defect, 1=clean)
- Two-mode training: detect (predict label) + inpaint (given label, fix body)
- Ensemble best-of-N + model self-scoring
- SITS (Soft Inpaint Time Schedule): per-frame adaptive t_start = ceil(sin((label+0.5)·π/2)·T)
```

---

## Part 6: Common Metrics & Computation Details

### 6.1 Position-Based Metrics (FK Required)

#### MPJPE (Mean Per-Joint Position Error)

```python
def compute_mpjpe(pred_motion, gt_motion, mask=None):
    """
    Args:
        pred_motion: (T, 135) predicted motion
        gt_motion: (T, 135) ground truth motion
        mask: (T, 135) optional mask (1=generated, 0=known)
    
    Returns:
        mpjpe_mean: float (meters)
        mpjpe_per_joint: [22] floats
    """
    # Forward kinematics: motion135 → (T, 22, 3) world positions
    pred_pos = FK(pred_motion, bone_offsets)   # (T, 22, 3)
    gt_pos = FK(gt_motion, bone_offsets)
    
    # Frame selection based on mask
    if mask is not None:
        frame_mask = mask.max(axis=-1) > 0.5  # (T,) frames with any generation
        N = frame_mask.sum()
    else:
        frame_mask = np.ones(T, dtype=bool)
        N = T
    
    # Per-joint error
    error = np.linalg.norm(pred_pos[frame_mask] - gt_pos[frame_mask], axis=-1)  # (N, 22)
    mpjpe_mean = error.mean()
    mpjpe_per_joint = error.mean(axis=0)  # (22,)
    
    return {
        'mpjpe_mean': float(mpjpe_mean),
        'mpjpe_per_joint': [float(x) for x in mpjpe_per_joint]
    }
```

**Mask variants**:
- `mpjpe_masked`: Error on mask=1 frames only (generated regions)
- `mpjpe_unmasked`: Error on entire sequence

---

### 6.2 Jitter (Temporal Smoothness)

#### Jitter via 3rd-order Finite Difference (Jerk)

```python
def compute_jitter(motion_or_positions, fps=30.0):
    """
    Jitter = mean jerk = mean(||d³x/dt³||) across all joints and time
    
    Args:
        motion_or_positions: (T, 135) motion or (T, 22, 3) positions
        fps: frame rate (default 30)
    
    Returns:
        jitter: float (m/s³ for positions, unitless for motion)
    """
    dt = 1.0 / fps
    if motion_or_positions.ndim == 2:  # (T, D) motion
        if motion_or_positions.shape[1] == 135:  # 135-dim
            # 3rd finite diff on 135-dim directly
            diff3 = (motion[3:] - 3*motion[2:-1] + 3*motion[1:-2] - motion[:-3])
            jerk = diff3 / (dt**3)
            jitter = np.abs(jerk).mean()
        else:  # (T, D) generic
            # Vectorize over D
            diff3 = (motion[3:] - 3*motion[2:-1] + 3*motion[1:-2] - motion[:-3])
            jitter = np.abs(diff3).mean()
    else:  # (T, 22, 3) positions
        # d³x/dt³ at each joint
        diff3 = (pos[3:] - 3*pos[2:-1] + 3*pos[1:-2] - pos[:-3])  # (T-3, 22, 3)
        jerk_norm = np.linalg.norm(diff3.reshape(diff3.shape[0], -1), axis=-1)  # (T-3,)
        jitter = jerk_norm.mean()
    
    return float(jitter)
```

**Units**:
- Motion 135-dim: unitless (on representation scale)
- Positions: m/s³
- Typical good motion: jitter < 1000 (normalized) or < 0.01 m/s³

---

### 6.3 Foot Skating Ratio

```python
def compute_foot_skating(motion, contact_threshold=0.8, velocity_threshold=0.005):
    """
    Foot skating: % of frames where foot has high contact but non-zero velocity.
    
    Args:
        motion: (T, 135) motion
        contact_threshold: min contact value to count as "in contact"
        velocity_threshold: max velocity to count as "stationary" (m/frame @ 30fps)
    
    Returns:
        skating_ratio: float [0, 1]
    """
    positions = FK(motion, bone_offsets)  # (T, 22, 3)
    
    # Detect contact frames (foot Y near ground + low velocity)
    foot_l_y = positions[:, 7, 1]  # Left ankle Y
    foot_r_y = positions[:, 8, 1]  # Right ankle Y
    
    in_contact = (foot_l_y < 0.05) | (foot_r_y < 0.05)  # Near ground
    
    # Velocity
    vel = np.linalg.norm(np.diff(positions, axis=0), axis=-1)  # (T-1, 22)
    foot_vel = np.concatenate([
        vel[:, 7:8],    # Left ankle
        vel[:, 8:9]     # Right ankle
    ], axis=-1)
    foot_moving = foot_vel.max(axis=-1) > velocity_threshold
    
    # Skating = (in_contact AND moving)
    skating_frames = in_contact[:-1] & foot_moving
    skating_ratio = skating_frames.mean()
    
    return float(skating_ratio)
```

**Threshold tuning**:
- Contact: typically Y < 0.05m (5cm above ground)
- Velocity: typically > 0.5 cm/frame (~0.15 m/s @ 30fps)
- Acceptable: skating_ratio < 5% (most mocap cleanups reach 10-20%)

---

### 6.4 Trajectory Metrics (ADE / FDE)

```python
def compute_trajectory_metrics(pred_motion, gt_motion, mask=None):
    """
    Trajectory = root XZ plane position (planar)
    
    Args:
        pred_motion: (T, 135)
        gt_motion: (T, 135)
        mask: (T, 135) optional
    
    Returns:
        trajectory_ade: float (meters)
        trajectory_fde: float (meters)
    """
    # Extract root XZ (dims 0, 2 of absolute translation)
    pred_traj = pred_motion[:, [0, 2]]  # (T, 2)
    gt_traj = gt_motion[:, [0, 2]]
    
    # Frame mask
    if mask is not None:
        frame_mask = mask.max(axis=-1) > 0.5
    else:
        frame_mask = np.ones(T, dtype=bool)
    
    if not frame_mask.any():
        return {'trajectory_ade': 0.0, 'trajectory_fde': 0.0}
    
    pred_sel = pred_traj[frame_mask]  # (N, 2)
    gt_sel = gt_traj[frame_mask]
    
    # ADE = average L2 distance
    distances = np.linalg.norm(pred_sel - gt_sel, axis=-1)  # (N,)
    ade = distances.mean()
    
    # FDE = final frame distance
    fde = distances[-1] if len(distances) > 0 else 0.0
    
    return {
        'trajectory_ade': float(ade),
        'trajectory_fde': float(fde)
    }
```

---

### 6.5 Loop Metrics

```python
def compute_loop_metrics(pred_motion, gt_motion):
    """
    Loop errors: position/velocity discontinuity between frame[-1] and frame[0].
    
    For a perfect loop, pred[-1] should smoothly transition to pred[0].
    """
    pred_pos = FK(pred_motion, bone_offsets)  # (T, 22, 3)
    gt_pos = FK(gt_motion, bone_offsets)
    
    # Position error: frame[-1] vs frame[0]
    pos_error = np.linalg.norm(pred_pos[-1] - pred_pos[0])
    
    # Velocity error: v[-1] vs v[0]
    vel = np.diff(pred_pos, axis=0)  # (T-1, 22, 3)
    vel_error = np.linalg.norm(vel[-1] - vel[0])
    
    return {
        'loop_position_error': float(pos_error),
        'loop_velocity_error': float(vel_error)
    }
```

---

### 6.6 End-Effector Metrics

```python
def compute_ee_metrics(pred_motion, gt_motion, constraint_frames, constraint_joints):
    """
    End-effector accuracy: distance between predicted and GT end-effector positions.
    
    Args:
        constraint_frames: (N,) frame indices where EE is constrained
        constraint_joints: (N,) joint indices (0-21)
    """
    pred_pos = FK(pred_motion, bone_offsets)  # (T, 22, 3)
    gt_pos = FK(gt_motion, bone_offsets)
    
    # Collect EE positions at constraint frames
    errors = []
    for frame in constraint_frames:
        for joint in constraint_joints:
            error = np.linalg.norm(pred_pos[frame, joint] - gt_pos[frame, joint])
            errors.append(error)
    
    errors = np.array(errors)
    
    return {
        'ee_error_mean': float(errors.mean()),
        'ee_error_p50': float(np.percentile(errors, 50)),
        'ee_error_p95': float(np.percentile(errors, 95)),
        'ee_error_max': float(errors.max()),
        'ee_hit_rate_5cm': float((errors < 0.05).mean()),
        'ee_hit_rate_10cm': float((errors < 0.10).mean()),
    }
```

---

## Part 7: Dataset Configurations

### 7.1 Evaluation Data Files (M2M v2)

| Task | File | Size | Source | Duration Range | Caption |
|------|------|------|--------|-----------------|---------|
| E1 (T2M) | eval_e1_t2m.json | — | MotionHub HQ | Diverse | Yes, rewritten |
| E2 (MIB) | eval_e2_inbetween_v2_rewritten.json | 220 motions | Private (held-out) | 50-240 frames | Yes, rewritten |
| E3 (Keyframe) | eval_e3_keyframe_v2_rewritten.json | 240 motions | Private (held-out) | Stratified by action + speed | Yes, rewritten |
| E4 (EE) | eval_e4_end_effector.json | — | MotionHub | Diverse | Optional |
| E5 (Trajectory) | eval_e2_inbetween_v2_rewritten.json | 220 motions | Same as E2 | 50-240 frames | Optional |
| E6 (Foot) | eval_e6_foot_ground.json | — | Mocap (clean) | Diverse | No |
| E7 (First-frame) | eval_e7_first_frame.json | — | MotionHub | Diverse | Yes |
| E8 (Loop) | eval_e8_loop_v2.json | — | MotionHub | 100-300 frames | Yes (Setting A only) |
| E9 (Repair) | eval_e9_repair_v2.json | 389 motions | Quality-checked defects | Diverse | No |
| E14 (Transition) | eval_e14_transition.json | — | MotionHub pairs | Various transition | No |
| E15 (Both-frame) | eval_e15_both_frame.json | — | MotionHub | Diverse | Yes |
| E16 (Tail) | eval_e16_tail_prediction.json | — | MotionHub | Diverse | Yes |

**Caption Processing**:
- Rewriter: Qwen3-30B-A3B-GRPO (deployed @ 11.216.46.236:8080)
- Output format: 12-20 words, "A person..." format
- Applied to: E1, E2, E3, E5, E7, E8-A, E15

---

### 7.2 Motion Representation Canonical Form

All motions should be canonicalized before evaluation:

```python
def canonicalize_motion_135d(motion, bone_offsets, fps=30):
    """
    Enforce input distribution: frame[0] (tx, tz) = 0, y_min = 0.
    
    Training data audit (200 clips):
    - frame[0] tx, tz = exactly 0 (offline canonicalized)
    - y_min centered at 0 (mean=-0.003m, std=0.038m, 80% within ±5cm)
    - Yaw uniformly distributed (NOT canonical)
    
    Test set OOD: y_min ≈ +0.146m (floats 14.6cm, ~4σ above training)
    
    Canonical form:
    1. Shift XZ so frame[0] (tx, tz) = (0, 0)
    2. Shift Y so min joint Y across all frames = 0 (ground contact)
    3. Leave yaw untouched
    """
    out = motion.astype(np.float32, copy=True)
    
    # Zero XZ translation at frame 0
    out[:, 0] -= out[0, 0]
    out[:, 2] -= out[0, 2]
    
    # Ground shift (Y)
    positions = FK(out, bone_offsets)  # (T, 22, 3)
    y_min = positions[:, :, 1].min()
    out[:, 1] -= y_min
    
    return out
```

---

## Part 8: Baseline Paper Configurations

### 8.1 KIMODO (NVIDIA, arXiv 2024-03-16)

**Paper**: Global Motion Representation for Motion Editing and Prediction

| Aspect | Setting |
|--------|---------|
| **Backbone** | Custom Transformer: 16L×8H×1024, 282M params |
| **Motion Repr** | Global 6D rotation (world frame) + smooth root, 333-dim (27 joints) |
| **Sampling** | DDPM: 1000 train steps, DDIM 100 infer steps |
| **Conditioning** | Imputation (hard replace) + binary mask concat |
| **Training** | Phase 1 (500k): T2M only; Phase 2 (500k): T2M + completion |
| **Data** | 700h optical mocap (Bones Rigplay, ~145k motions) |
| **T2M Metrics** | FID: ~0.65 (on HumanML3D test w/ custom eval) |
| **Completion** | Supports keyframe, EE, trajectory, contact via imputation |
| **Post-process** | Foot lock filtering (reduce skating) |

**Unique capabilities**:
- ✅ Direct XYZ position imputation (global rotation repr includes world pos)
- ✅ Explicit contact state modeling
- ✅ Two-stage curriculum (T2M safety valve)
- ✅ High-quality mocap data

**Weaknesses**:
- ❌ No editing semantics (only generation/completion)
- ❌ Rotation↔position consistency only via FK loss (soft)
- ❌ No reaction/multi-person support

---

### 8.2 UMO (Brown/MIT/Meta/MPI/HKU, arXiv 2024-03-16)

**Paper**: Universal Motion Generation with Meta-Operators

| Aspect | Setting |
|--------|---------|
| **Backbone** | HY-Motion-Lite MMDiT (460M), frozen during training |
| **Motion Repr** | 201-dim (3 abs_transl + 6 root + 21×6 rot + 21×3 pos), HumanML3D |
| **Sampling** | Flow Matching (rectified flow, 50-step Euler) |
| **Conditioning** | Temporal Fusion (element-wise add, 0.207M params only) |
| **Training** | Multi-task joint from start (MIB, prediction, editing, reaction) |
| **Data** | HumanML3D + MotionFix + internet video + animation (mixed quality) |
| **Meta-operators** | [Preserve], [Generate], [Edit] — frame-level labels |
| **Metrics** | FID, diversity, EE error (frame-level tasks) |

**Unique capabilities**:
- ✅ Three explicit meta-operations (P/G/E)
- ✅ Minimal architecture invasion (0.207M params)
- ✅ Multi-task + reaction + multi-person
- ✅ Instruction editing via text (e.g., "use opposite leg")

**Limitations**:
- ❌ Frame-level only (no part-level control)
- ❌ Position channels in repr but NOT directly controllable (soft constraint only)
- ❌ P-frame outputs have ~0.95mm error (not exact)
- ❌ Code not released (paper: "will release")

---

### 8.3 MotionLab (SUTD/Lightspeed, ICCV 2025, 2026-05 archive)

**Paper**: Unified Generation and Editing of Motion via Motion-Condition-Motion with Task Instruction**

| Aspect | Setting |
|--------|---------|
| **Backbone** | MM-DiT (multi-modality), 5 modality paths |
| **Motion Repr** | HumanML3D 263-dim (3 transl + 21×6 rot + 21×3 pos + 22 contact + velocity) |
| **Sampling** | Flow Matching (Euler / DPM-Solver) |
| **Conditioning** | 5 independent modality inputs + Aligned 1D RoPE |
| **Training** | 7-stage curriculum + motion-aware FID resampling |
| **Data** | HumanML3D (10k) + HyMotion MotionFix + paired editing corpus |
| **Unified Framework** | (source, target, condition) → (source, target, text, trajectory, style) |
| **Task Instruction** | CLIP-encoded task text (e.g., "edit source by given text") |

**Key metrics** (ICCV 2025):
- **T2M**: FID 0.238 (vs MoMask 0.238, MotionGPT 0.258)
- **Inpainting**: MPJPE 0.101m (masked), 0.098m (unmasked)
- **Keyframe**: 0.083m (masked, every_30f)
- **Trajectory**: 0.0286m ADE (state-of-art for trajectory control)
- **Style transfer**: SRA 69.21 (novel contribution)
- **Editing**: All 6 tasks > specialist models

**Unique contributions**:
- ✅ Trajectory as dedicated modality + Aligned RoPE
- ✅ Task Instruction Modulation (language-driven routing)
- ✅ Curriculum Learning (7 stages, FID-weighted sampling)
- ✅ Unified gen+edit in single model (specialist > unified on avg was a myth)

**Curriculum schedule** (most important for reproduction):
```
Epoch 0-1000: Masked pre-training on 50% masked (all tasks invisible)
Epoch 1000-1200: T2M only (task token: "generate motion from text")
Epoch 1200-1400: T2M + Inpainting (add inpaint token)
Epoch 1400-1600: + Keyframe (add keyframe token)
Epoch 1600-1800: + Trajectory (add trajectory token)
Epoch 1800-2000: + Inst-Editing (add instruction-edit token)
Epoch 2000-2200: + Style transfer (add style token)
Post-epoch 2200: All tasks with FID-weighted resampling (tasks with high FID get more batches)
```

**Key findings**:
- Ablation: removing curriculum → FID 11.7× worse (most critical)
- Unified training: all tasks improve vs curriculum-less (multi-task acts as regularization)
- Trajectory ADE improves from 0.041m (naive) → 0.0286m (with Aligned RoPE)

---

### 8.4 SOAR (NUS/Alibaba/MSR, arXiv 2025-04, CVPR 2026 submitted)

**Paper**: Diffusion Models Learn Exposure Bias during Training (+ SOAR: Exposure Bias Correction via On-Policy Rollout)

| Aspect | Setting |
|--------|---------|
| **Problem** | Exposure bias in diffusion: train on GT x_t but infer from model x_t (off-trajectory) |
| **Solution** | On-policy rollout + re-noise + dense per-timestep correction |
| **Stage** | Post-training (applies to any flow matching model) |
| **Method** | Perform 1-step ODE rollout of current model → re-noise → supervise toward clean target |
| **Data** | None (self-supervised, no extra labels) |
| **Computation** | ~2× inference cost (50-step ODE × 2 = 100 steps effective) |

**Application to M2M**:
- Direct applicability (M2M uses rectified flow, same as SOAR target)
- Post-training on frozen M2M (no retraining)
- Orthogonal to _man editing mode (can combine)

**Performance (SD3.5-Medium)**: 
- GenEval: 0.70 → 0.78 (outperforms SD3.5-Large 0.71)
- Motion expectation: similar 10-15% improvement for motion diffusion

---

### 8.5 StableMotion (SFU/Lightspeed/NRC, SIGGRAPH Asia 2025, 2026-04 archive)

**Paper**: Motion Cleanup via Quality Indicator Learning on Unpaired Corrupted Data**

| Aspect | Setting |
|--------|---------|
| **Problem** | Motion datasets have defects (penetration, foot skating, jerks) — hard to remove without clean reference |
| **Solution** | Quality-indicator channel + unpaired corrupted data + detect+fix training |
| **Motion Repr** | SMPL RIFKE global 6D, **+ 1 binary quality dimension** (defect flag) |
| **Training** | Two modes: (1) Detect mode: given body, predict quality; (2) Inpaint mode: given quality, repair body |
| **Detection** | Monte-Carlo: run 5 stochastic inferences, pick consensus |
| **Inference** | Detect → dilate → inpaint (detect every frame, label quality 0/1) |
| **Adaptive schedule** | SITS (Soft Inpaint Time Schedule): t_start = ceil(sin((label+0.5)·π/2)·T) |

**Key metrics**:
- Clean frame restoration: ~95% (when defect clearly marked)
- False positive rate: <5% (rarely marks clean as defect)
- Works on completely unpaired data (no clean references needed)

**Reference value for M2M**:
- Quality channel (1-dim) can be integrated into 135-dim representation
- Unpaired training paradigm aligns with M2M's goal of iterative quality improvement
- `_man` + StableMotion quality channel = two-axis improvement (known region fidelity + defect removal)

---

## Part 9: Experiment Design Template

### 9.1 Standard M2M Evaluation Template

```python
# Load model
model = load_model('work_dirs/m2m_v2_caption_local/checkpoint.ckpt')
model.eval()

# Iterate over tasks and settings
TASKS = ['E2', 'E3', 'E4', 'E5', 'E6', 'E7']
SETTINGS_BY_TASK = {
    'E2': ['start_1f', 'end_1f', 'both_1f', 'pre20', 'post20', 'mid60'],
    'E3': ['every_5f', 'every_10f', 'every_15f', 'every_30f', 'every_60f', 'adaptive'],
    'E4': ['single_sparse', 'single_medium', 'two_sparse', 'two_medium', 'all4_sparse', 'all4_dense'],
    'E5': ['A_xz_dense', 'B_xz_sparse', 'C_xz_heading', 'D_xyz_dense', 'E_xyz_sparse', 'F_xyz_heading'],
    'E6': ['pos_contact'],
    'E7': ['default'],
}

metrics_all = {}

for task_id in TASKS:
    for setting_name in SETTINGS_BY_TASK[task_id]:
        # Load eval task
        task = get_eval_task(task_id)
        setting = task.settings[setting_name]
        
        # Load data
        data = load_json(task.data_file)
        samples = data['data'][:100]  # Limit to 100 for quick eval
        
        # Run evaluation
        results = []
        for sample in samples:
            motion_gt = load_motion(sample['motion_path'])  # (T, 135)
            caption = sample['caption'] if task.needs_caption else ""
            
            # Build mask
            T = motion_gt.shape[0]
            mask = task.build_mask(T, D=135, setting_name=setting_name)
            
            # Prepare condition
            src_motion = motion_gt * mask  # Masked-out regions = 0
            src_mask = mask
            
            # Run inference
            with torch.no_grad():
                pred_motion = model.generate(
                    src_motion=src_motion,
                    src_mask=src_mask,
                    caption=caption,
                    guidance_scale=7.5 if task.needs_caption else 1.0,
                    num_steps=50,
                )  # (T, 135)
            
            # Canonicalize
            pred_motion = canonicalize_motion_135d(pred_motion, bone_offsets)
            motion_gt = canonicalize_motion_135d(motion_gt, bone_offsets)
            
            # Compute metrics
            metrics_sample = {}
            for metric_name in task.default_metrics:
                if metric_name == 'mpjpe_masked':
                    result = compute_mpjpe(pred_motion, motion_gt, mask)
                    metrics_sample[metric_name] = result['mpjpe_mean']
                elif metric_name == 'jitter_pos':
                    metrics_sample[metric_name] = compute_jitter(pred_motion)
                elif metric_name == 'foot_skating_ratio':
                    metrics_sample[metric_name] = compute_foot_skating(pred_motion)
                elif metric_name == 'trajectory_ade':
                    result = compute_trajectory_metrics(pred_motion, motion_gt, mask)
                    metrics_sample[metric_name] = result['trajectory_ade']
                # ... etc
            
            results.append(metrics_sample)
        
        # Aggregate
        metrics_all[f"{task_id}_{setting_name}"] = {
            metric: np.mean([r[metric] for r in results])
            for metric in task.default_metrics
        }

# Output results
import json
with open('eval_results.json', 'w') as f:
    json.dump(metrics_all, f, indent=2)
```

---

## Part 10: Quick Reference Table

### Metric Quick-Look

| Metric | Task | Unit | Good Value | Computation |
|--------|------|------|-----------|-------------|
| **FID** | T2M | — | < 0.30 | Inception feature distance |
| **MPJPE (masked)** | All completion | m | < 0.10 | Position error on generated frames |
| **MPJPE (unmasked)** | All completion | m | < 0.15 | Position error on all frames |
| **Jitter** | All | m/s³ or unitless | < 1000 (135-dim) | Mean ||d³x/dt³|| |
| **Foot skating** | All | ratio | < 5% | % frames in contact + moving |
| **Trajectory ADE** | E5 | m | < 0.05 | Mean root XZ distance |
| **Trajectory FDE** | E5 | m | < 0.10 | Final root XZ distance |
| **Loop error** | E8 | m | < 0.10 | ||pos[-1] - pos[0]|| |
| **EE error mean** | E4 | m | < 0.05 | Mean distance to constrained EE |
| **EE hit rate 5cm** | E4 | % | > 80% | % of constraints within 5cm |
| **Boundary accel** | E2, E8 | m/s² | < 2.0 | Acceleration jump at mask edge |

---

## Conclusion

This document provides the **complete benchmark settings** for motion completion/control tasks as standardized by:
- NVIDIA KIMODO (imputation-based, global rotation)
- Brown/MIT/Meta UMO (frame-level meta-ops, temporal fusion)
- SUTD MotionLab (task instruction + curriculum)
- NUS/Alibaba SOAR (exposure bias post-training)
- SFU StableMotion (quality-indicator repair)
- HyMotion M2M v2 (dimension-level VACE conditioning)

**Key takeaways for your experiments**:
1. **Temporal completion**: Use E2 (MIB with pre20/post20/mid60 settings) and E3 (keyframe every_30f)
2. **Trajectory control**: Use E5 A_xz_dense or B_xz_sparse (training-aligned, planar)
3. **End-effector**: Use E4 all4_dense (all 4 limbs, 20% random frames)
4. **Standard data**: 220-motion held-out pool for E2/E3/E5 (consistent evaluation)
5. **Metrics**: Always report (MPJPE, jitter, foot_skating) — these are universally used
6. **Canonicalization**: Always apply ground-shift + XZ-zero before evaluation

---

**Generated**: 2026-05-22  
**Format**: Markdown (plain-text friendly)  
**Status**: Ready for team alignment on benchmark standards

