# Detailed Code Flow Analysis: Retargeting Pipeline

## Step 1: motion135 → SMPL-X (motion135_to_smplx.py)

### Data Flow
```
Input motion_135 NPZ:
├── motion_135: (T=60, 135) = [transl(3) + 22*rot6d(132)]
├── Optional: positions, translation

↓ [Convert rot6d to rotmat to axis-angle]

Output SMPL-X NPZ:
├── pose_body: (T=60, 63) axis-angle (21 joints)
├── root_orient: (T=60, 3) axis-angle (pelvis)
├── trans: (T=60, 3) translation
├── betas: (10,) shape params (zeros)
└── mocap_frame_rate: 30
```

### Rotation Conversion Pipeline
```
HyMotion rot6d (row-major)
    [R00, R01, R10, R11, R20, R21]
        ↓ [0,2,4,1,3,5] reorder
Column-major
    [R00, R10, R20, R01, R11, R21]
        ↓ Gram-Schmidt orthogonalization
Rotation matrix (3×3)
        ↓ scipy.spatial.transform.Rotation
Axis-angle (3,)
```

**CRITICAL CODE - Line 39:**
```python
rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]  # <-- ASSUMPTION: HyMotion uses row-major
```
**ACTION**: Verify this reordering by comparing against known rotations

---

## Step 2: SMPL-X → GMR Robot (gmr_retarget_headless.py)

### Data Flow
```
Input SMPL-X NPZ
├── pose_body: (T, 63) axis-angle
├── root_orient: (T, 3) axis-angle
├── trans: (T, 3) SMPL-X Y-up
└── betas: (10,) shape

↓ [Load GMR, initialize retargeter]

GMR Processing (per-frame):
├── Frame 0: frame_data = dict(pose_body, root_orient, trans, ...)
├── retarget(frame_data, offset_to_ground=True/False)
│   ├── Internal: Scale SMPL-X by human height
│   ├── Internal: IK solver finds robot joint angles
│   ├── Internal: Apply joint limit clamping
│   └── Output: qpos = [root_pos(3), root_rot_wxyz(4), dof_pos(29)]
└── Repeat for all frames

↓ [Post-processing: clamp joint limits]

Output GMR PKL:
├── fps: 30
├── root_pos: (T, 3) SMPL-X Y-up frame
├── root_rot: (T, 4) xyzw quaternion
└── dof_pos: (T, 29) joint angles
```

### Critical Code Sections

#### Ground Offset Computation (Line 112-139)
```python
def compute_ground_offset(retarget, smplx_data_frames):
    offset = np.inf
    for frame_data in smplx_data_frames:
        # 1. Convert SMPL-X to numpy (format conversion)
        human_data = retarget.to_numpy(frame_data)
        
        # 2. Scale by body model (height adjustment)
        human_data = retarget.scale_human_data(
            human_data, retarget.human_root_name, retarget.human_scale_table
        )
        
        # 3. Apply offsets from skeleton mapping
        human_data = retarget.offset_human_data(
            human_data, retarget.pos_offsets1, retarget.rot_offsets1
        )
        
        # 4. Find minimum Z across all body parts
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if pos[2] < offset:
                offset = pos[2]
    return offset
```

**Issue**: This computes ground offset BEFORE IK retargeting!
- The offset from SMPL-X skeleton may not be relevant to robot skeleton
- Robot skeleton could be taller/shorter/different proportions
- **Result**: Offset might be wrong for the actual robot

**Recommendation**: Compute offset AFTER GMR IK, from FK body positions

#### IK Retargeting Loop (Line 192-196)
```python
for i, frame_data in enumerate(smplx_data_frames):
    qpos = retarget.retarget(frame_data, offset_to_ground=args.offset_to_ground)
    qpos_list.append(qpos)
```

**Issue**: No temporal smoothing!
- Each frame solved independently
- IK can produce different solutions (elbow-up vs elbow-down for same end-effector)
- No penalty for large frame-to-frame changes
- **Result**: Joint angles can oscillate unpredictably

#### Joint Limit Clamping (Line 85-109)
```python
def clamp_joint_limits(dof_pos, joint_order=G1_JOINT_ORDER, joint_limits=G1_JOINT_LIMITS):
    clamped = dof_pos.copy()
    num_clamped = 0
    for i, joint_name in enumerate(joint_order):
        if joint_name in joint_limits:
            lo, hi = joint_limits[joint_name]
            clamped[:, i] = np.clip(clamped[:, i], lo, hi)
            num_clamped += np.sum((dof_pos[:, i] < lo) | (dof_pos[:, i] > hi))
    return clamped, int(num_clamped)
```

**Issue**: Hard clipping creates discontinuities!
- Frame t: joint angle = 1.5 rad (just at limit)
- Frame t+1: joint angle = 1.51 rad (over limit) → clipped to 1.5 rad
- Result: Large velocity spike (1.51 - 1.5) / dt = 0.01/0.033 ≈ 0.3 rad/s
- Accumulated across frames: joint angle trajectory becomes jagged
- **Result**: Trembling when robot joint hits limits on adjacent frames

---

## Step 3: GMR → ProtoMotions (gmr_to_protomotions.py)

### Data Flow
```
Input GMR PKL
├── fps: 30
├── root_pos: (T=60, 3) Y-up
├── root_rot: (T=60, 4) xyzw
└── dof_pos: (T=60, 29)

↓ [Coordinate frame conversion]

├── Convert root_pos from Y-up to Z-up
└── Remove rot_offset from root_rot quaternion

↓ [FK-based ground correction] (optional)

├── For each frame t:
│   ├── Set MuJoCo qpos = [root_pos, root_rot, dof_pos]
│   ├── Run forward kinematics
│   ├── Find min foot Z from FK
│   └── Adjust root_pos[t, 2] to make foot touch ground
└── Output: corrected_root_pos

↓ [MuJoCo FK]

For each frame:
├── Set qpos = [root_pos, root_rot, dof_pos]
├── Run mujoco.mj_forward()
├── Extract body positions and rotations
└── Collect: body_pos (T, 33, 3), body_rot (T, 33, 4)

↓ [Resampling]

├── Source: 30 Hz (60 frames)
├── Target: 50 Hz (100 frames)
├── Method: linear interp for dof_pos, SLERP for rotations
└── Output: resampled data at 50 Hz

↓ [Velocity computation]

├── DOF velocity: finite difference
├── Body velocity: finite difference
└── Body angular velocity: from quaternion differences

Output ProtoMotions cache .pt:
├── dof_pos: (T'=100, 29)
├── dof_vel: (T'=100, 29)
├── body_pos: (T'=100, 33, 3)
├── body_rot: (T'=100, 33, 4) xyzw
├── body_vel: (T'=100, 33, 3)
├── body_ang_vel: (T'=100, 33, 3)
├── control_dt: 0.02
└── num_frames: 100
```

### Critical Code Sections

#### Coordinate Frame Conversion (Line 92-111)
```python
def convert_root_pos_to_zup(root_pos):
    """Apply rotation to translation vector (axis remapping)."""
    rot_offset = _get_gmr_rot_offset()  # 120° rotation matrix
    return rot_offset.inv().apply(root_pos).astype(root_pos.dtype)

# rot_offset = [0.5, -0.5, -0.5, -0.5] (wxyz) [120° rotation]
# Maps: [x,y,z]_smplx → [z,x,y]_mujoco
#       X → Z (forward becomes up)
#       Y → X (left becomes forward)
#       Z → Y (up becomes left)
```

**Issue**: The rotation is applied correctly, BUT:
- Position and rotation transformations might not be consistent
- If root_rot offset removal is wrong, feet don't align with root
- **Test**: Load a simple standing pose, check foot positions relative to root

#### Root Rotation Offset Removal (Line 69-89)
```python
def remove_gmr_root_offset(root_rot_xyzw):
    """Remove the 120° frame conversion rotation baked into GMR output."""
    rot_offset = _get_gmr_rot_offset()  # Rotation object: 120° Y-up→Z-up
    root_rots = R.from_quat(root_rot_xyzw)  # scipy Rotation objects
    corrected = root_rots * rot_offset.inv()  # RIGHT multiply
    return corrected.as_quat().astype(root_rot_xyzw.dtype)

# Quaternion algebra (Hamilton convention):
# q1 * q2 means: apply q2 first, then q1
# So: root_rots * rot_offset.inv()
#     = apply root_rots, then undo the rot_offset
#     = This should be correct!
```

**Question**: Is scipy.spatial.transform.Rotation using Hamilton or JPL convention?
- Hamilton: q * v is "apply q to v"
- JPL: q * v means "undo q then apply to v"
- **Verify**: Check scipy documentation + test with known quaternions

#### FK-Based Ground Correction (Line 155-229)
```python
def fk_ground_correction(mjcf_path, root_pos, root_rot_xyzw, dof_pos, 
                         foot_body_indices=None, ground_clearance=0.0):
    """Adjust root_pos Z so feet touch ground after FK."""
    
    for t in range(T):
        # 1. Set qpos for this frame
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = quat_xyzw_to_wxyz(root_rot_xyzw[t])
        data.qpos[7:] = dof_pos[t]
        data.qvel[:] = 0.0
        
        # 2. Run forward kinematics
        mujoco.mj_forward(model, data)
        
        # 3. Find minimum foot Z from body positions
        min_foot_z = np.inf
        for bi in foot_body_indices:  # e.g., [7, 13] = left/right ankle
            foot_z = data.xpos[bi + 1][2]  # body index +1 (world body offset)
            min_foot_z = min(min_foot_z, foot_z)
        
        # 4. Adjust root Z to make lowest foot touch ground
        z_offset = ground_clearance - min_foot_z
        corrected_root_pos[t, 2] = root_pos[t, 2] + z_offset
```

**Issues**:
1. **Per-frame independence**: Each frame's root Z is adjusted independently
   - Frame t: foot_z = 0.05, adjusted Z += 0.05
   - Frame t+1: foot_z = 0.10, adjusted Z += 0.10
   - Result: Root Z jumps by 0.05 between frames (trembling!)

2. **Nonlinear FK**: Changing root Z doesn't scale feet linearly
   - Root is free-floating: changing Z doesn't directly change dof_pos
   - But dof_pos describes robot joints in root frame
   - Result: Foot positions depend on complex FK chain

3. **No temporal smoothing**: Corrections are isolated
   - No continuity constraint between frames
   - High-frequency oscillations preserved/amplified

**Fix Strategy**: 
   - Compute all foot Z values without correction
   - Find smoothest Z trajectory that minimizes foot penetration
   - Use Viterbi or Kalman smoothing

#### Resampling (Line 296-342)
```python
# Create time vectors
times_src = np.arange(T_src) * src_dt  # [0, 0.033, 0.067, ...]
times_tgt = np.arange(T_tgt) * control_dt  # [0, 0.02, 0.04, ...]

# Resample dof_pos with LINEAR interpolation
dof_interp = interp1d(times_src, dof_pos, axis=0, kind='linear')
dof_pos_resampled = dof_interp(times_tgt)

# Resample body_rot with SLERP (Spherical Linear Interpolation)
for b in range(num_bodies):
    rots = R.from_quat(body_rot_xyzw[:, b, :])  # Convert to Rotation objects
    slerp_fn = Slerp(times_src, rots)  # Create SLERP interpolator
    rots_resampled = slerp_fn(times_tgt)  # Evaluate at target times
    body_rot_resampled[:, b, :] = rots_resampled.as_quat()
```

**Issue**: Joint angles use LINEAR interpolation!
- Joints are angles in SO(3), but treated as Euclidean vectors
- Linear interpolation in angle space ≠ geodesic interpolation
- Example: 
  - Frame 0: ankle angle = 0 rad
  - Frame 1: ankle angle = π rad (180°)
  - Linear interp at t=0.5: angle = π/2 rad (90°)
  - But shortest path might be different depending on joint limits!

**Fix**: Apply SLERP-like interpolation to joint rotations
- Convert joint angles to quaternions (or rotation matrices)
- Apply SLERP
- Convert back to angles

#### Velocity Computation (Line 345-384)
```python
# DOF velocity from position finite diff
dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt
dof_vel[0] = dof_vel[1]

# Body linear velocity from position finite diff
body_vel[1:] = (body_pos[1:] - body_pos[:-1]) / dt
body_vel[0] = body_vel[1]

# Body angular velocity from quaternion difference
for b in range(num_bodies):
    rots = R.from_quat(body_rot_xyzw[:, b, :])
    for t in range(1, T):
        drot = rots[t] * rots[t-1].inv()  # Relative rotation
        rotvec = drot.as_rotvec()  # Convert to axis-angle
        body_ang_vel[t, b] = rotvec / dt
```

**Issues**:
1. **First-order finite differences**: Not smooth, especially at discontinuities
   - If qpos has jumps (from clamping), velocity has spikes
   - No low-pass filtering applied

2. **Angular velocity instability**:
   - For small rotations, rotvec → 0 (numerical instability)
   - Large rotations might have singularities (e.g., ±π)
   - Accumulation of quaternion normalization errors

**Fix**:
   - Apply Savitzky-Golay filter to velocities
   - Use higher-order finite difference schemes
   - Smooth angular velocity with low-pass filter

---

## Step 4: Rendering (render_tracker_headless.py)

### Reference Mode (Line 220-323)
```python
def render_reference_mode(cache, model, data, output_dir, ...):
    """Directly set qpos from cache (no simulation)."""
    
    for frame_idx in range(num_frames, skip_frames):
        # Extract state from cache
        root_pos = body_pos[frame_idx, 0, :]  # pelvis position
        root_rot_xyzw = body_rot[frame_idx, 0, :]  # pelvis rotation
        dof = dof_pos[frame_idx, :]  # joint angles
        
        # Construct qpos
        root_rot_wxyz = [root_rot_xyzw[3], root_rot_xyzw[0:3]]  # xyzw → wxyz
        qpos = np.concatenate([root_pos, root_rot_wxyz, dof])
        
        # Set in MuJoCo (NO simulation step)
        data.qpos[:len(qpos)] = qpos
        data.qvel[:] = 0.0  # Zero velocities
        
        # Render without stepping
        mujoco.mj_forward(model, data)  # FK only
        renderer.render()
```

**Note**: This is a PURE kinematics render (no dynamics)
- No validation against source SMPL-X motion
- Trembling visible here is purely from retargeting pipeline
- Good for isolating retargeting issues

---

## Summary: Where Trembling Can Be Introduced

| Stage | Component | Effect |
|-------|-----------|--------|
| 1 | rot6d reordering | High-frequency rotation noise |
| 2 | IK solver (no temporal smoothing) | Oscillating joint solutions |
| 2 | Joint limit clamping | Discontinuities at limits |
| 2 | Ground offset (global, not per-frame) | Kinematic inconsistency |
| 3 | Frame conversion (position/rotation mismatch) | Foot float relative to root |
| 3 | FK ground correction (per-frame independent) | Root Z oscillation → joint trembling |
| 3 | Linear resampling of joint angles | High-frequency angle artifacts |
| 3 | Simple finite difference velocities | Noisy velocity references |

**Most Likely Culprits**:
1. FK ground correction (per-frame adjustment without smoothing)
2. Joint limit clamping (creates discontinuities)
3. IK solver oscillation (no temporal prior)

