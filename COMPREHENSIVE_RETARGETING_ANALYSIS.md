# COMPREHENSIVE SMPL-X → UNITREE G1 RETARGETING PIPELINE ANALYSIS

## Executive Summary

The retargeting pipeline has **MULTIPLE CRITICAL ERRORS** that compound to produce fundamentally incorrect robot motion. The center of gravity, joint positions, and coordinate frames are all misaligned. Here's what's wrong:

---

## 1. COMPLETE PIPELINE ARCHITECTURE

```
motion_135 NPZ (HyMotion eval output)
    ├─ Shape: (T, 135) = [transl(3) + 22×rot6d(132)]
    └─ Coordinate frame: ??? (undocumented, assumed HyMotion M2M internal)
    
motion135_to_smplx.py
    ├─ Converts rot6d (row-major) → rotation matrix (Gram-Schmidt) → axis-angle
    ├─ Output: SMPL-X NPZ with pose_body(T,63), root_orient(T,3), trans(T,3)
    └─ Coordinate frame: ASSUMED Y-up (standard SMPL)
    
gmr_retarget_headless.py (uses ref_repo/GMR)
    ├─ IK Solver: mink (MuJoCo-based inverse kinematics)
    ├─ Joint mapping: 22 SMPL-X joints → 29 G1 DOFs via ik_match_table1/2
    ├─ Coordinate frame conversions: 
    │   └─ SMPL-X Y-up → MuJoCo Z-up via rot_offset [0.5, -0.5, -0.5, -0.5] (wxyz)
    ├─ Output: GMR PKL {fps, root_pos(T,3), root_rot(T,4 xyzw), dof_pos(T,29)}
    └─ Coordinate frame: MuJoCo Z-up CLAIMED, but contains issues
    
gmr_to_protomotions.py
    ├─ Converts GMR PKL → ProtoMotions .pt cache
    ├─ Steps:
    │   1. Convert root_pos: Y-up → Z-up (rot_offset.inv().apply())
    │   2. Remove GMR's rot_offset from root_rot
    │   3. FK ground correction (per-frame or smoothed)
    │   4. MuJoCo FK for all 33 bodies
    │   5. Resample 30Hz → 50Hz
    │   6. Compute velocities via Savitzky-Goyal or finite diff
    ├─ Output: .pt cache {dof_pos, dof_vel, body_rot, body_pos, body_vel, body_ang_vel}
    └─ Coordinate frame: MuJoCo Z-up
```

---

## 2. COORDINATE FRAME MISALIGNMENT (CRITICAL ERROR #1)

### 2.1 SMPL-X Frame (Standard)
- **X-axis**: Right (positive to subject's right)
- **Y-axis**: UP (positive upward)
- **Z-axis**: Forward/Out of page (positive forward)
- **Height**: Measured along Y

### 2.2 MuJoCo G1 Frame (Z-up)
- **X-axis**: Forward
- **Y-axis**: Left
- **Z-axis**: UP (positive upward)
- **Height**: Measured along Z

### 2.3 The "Frame Conversion" in GMR

**In smplx_to_g1.json - pelvis rot_offset:**
```json
"rot_offset": [0.5, -0.5, -0.5, -0.5]  // wxyz
```

This is claimed to convert Y-up → Z-up, mapping:
- X_smplx → Z_mujoco
- Y_smplx → X_mujoco
- Z_smplx → Y_mujoco

**PROBLEM #1: The frame conversion is INCOMPLETE**

1. **Root position pass-through WITHOUT frame conversion in GMR**
   - GMR's IK solver takes SMPL-X positions as-is
   - NO frame conversion applied to target body positions
   - Result: IK targets are in wrong coordinate frame
   - G1's solver tries to match Y-up positions in Z-up space

2. **Frame conversion happens AFTER IK in gmr_to_protomotions.py**
   ```python
   # Line 505: convert_root_pos_to_zup()
   root_pos = rot_offset.inv().apply(root_pos)
   # This converts: [x, y, z]_smplx → [z, x, y]_mujoco
   ```
   
   **CRITICAL ERROR**: Root position was computed by IK solver in MIXED frame:
   - Input targets (body positions): SMPL-X Y-up
   - Robot model (MuJoCo): Z-up with forward kinematics in Z-up
   - Result: Root position is **wrong** before conversion
   - Post-hoc frame conversion doesn't fix the underlying error

---

## 3. JOINT MAPPING & IK CONFIGURATION ERRORS (CRITICAL ERROR #2)

### 3.1 SMPL-X Body Structure (22 joints)
```
Index 0: Pelvis (root)
Indices 1-3: Left leg (hip, knee, ankle)
Indices 4-6: Right leg (hip, knee, ankle)
Indices 7-9: Spine (spine1, spine2, spine3)
Indices 10-12: Left arm (shoulder, elbow, wrist)
Indices 13-15: Right arm (shoulder, elbow, wrist)
Indices 16-21: Head, jaw, eyes (ignored in animation)
```

### 3.2 Unitree G1 Robot Structure (29 DOF)
```
Fixed-base to floating-base:
  0-2:   left leg (hip_pitch, hip_roll, hip_yaw)
  3-5:   left knee, ankle_pitch, ankle_roll
  6-8:   right leg (hip_pitch, hip_roll, hip_yaw)
  9-11:  right knee, ankle_pitch, ankle_roll
  12-14: waist (yaw, roll, pitch)
  15-17: left shoulder (pitch, roll, yaw)
  18:    left elbow
  19-21: left wrist (roll, pitch, yaw)
  22-24: right shoulder (pitch, roll, yaw)
  25:    right elbow
  26-28: right wrist (roll, pitch, yaw)
```

### 3.3 Mapping Configuration (smplx_to_g1.json analysis)

**IK Match Table 1 (Primary):**
```
Robot frame → SMPL body  | Position weight | Rotation weight | Position offset | Rotation offset
pelvis → pelvis          | 100              | 10              | [0, 0, 0]      | [0.5, -0.5, -0.5, -0.5]
left_hip_roll_link → left_hip        | 0 | 10 | [0,0,0] | [0.426, -0.564, -0.564, -0.427]
left_knee_link → left_knee           | 0 | 10 | [0,0,0] | [0.5, -0.5, -0.5, -0.5]
left_toe_link → left_foot            | 100 | 10 | [0, 0.02, 0] | [0.5, -0.5, -0.5, -0.5]
... (symmetric for right side)
torso_link → spine3                  | 0 | 10 | [0,0,0] | [0.5, -0.5, -0.5, -0.5]
left_shoulder_yaw_link → left_shoulder | 0 | 10 | [0,0,0] | [0.707, 0, -0.707, 0]
left_elbow_link → left_elbow         | 0 | 10 | [0,0,0] | [1, 0, 0, 0]
... etc
```

**PROBLEM #2.1: Missing hip pitch/yaw frames**

Notice in the mapping above:
- `left_hip_pitch_link` - **NOT MAPPED** in ik_match_table1
- `left_hip_yaw_link` - **NOT MAPPED** in ik_match_table1
- Only `left_hip_roll_link` is mapped

But SMPL-X has a single `left_hip` joint for all three DOFs (pitch, roll, yaw).

**What happens?**
1. SMPL-X provides single left_hip orientation
2. IK only matches it to `left_hip_roll_link` (zero position weight!)
3. The hip pitch and yaw **are not explicitly constrained**
4. IK solver must infer them from leg kinematics
5. **Result: Hip orientation highly ambiguous, IK may use bad solutions**

**PROBLEM #2.2: Position weight inconsistency**

Notice:
```
left_hip_roll_link: position_weight = 0  (don't match position)
left_toe_link: position_weight = 100     (match position strongly)
```

But SMPL-X `left_hip` has:
- Position: center of hip joint
- Cannot be positioned accurately without matching hip frame position

**Result: Hip position floats, foot position locked**
**Consequence: Leg link lengths become inconsistent with SMPL-X**

---

## 4. IK SOLVER CONFIGURATION (CRITICAL ERROR #3)

### 4.1 The mink IK Solver

From `motion_retarget.py` lines 13-22:

```python
def __init__(
    self,
    src_human: str,
    tgt_robot: str,
    actual_human_height: float = None,
    solver: str="daqp",      # Changed from "quadprog"
    damping: float=5e-1,     # 0.5, changed from 1e-2 (0.01)
    verbose: bool=True,
    use_velocity_limit: bool=False,
)
```

**Key parameters:**
- **Solver**: DAQP (Distributed Algorithm for Quadratic Programming)
- **Damping**: 0.5 (HIGH - means heavy regularization)
- **Max iterations**: 10
- **Configuration limits**: Only joint range limits (not velocity limits)

### 4.2 IK Loop Logic (motion_retarget.py lines 173-216)

```python
def retarget(self, human_data, offset_to_ground=False):
    # Update task targets
    self.update_targets(human_data, offset_to_ground)
    
    if self.use_ik_match_table1:
        curr_error = self.error1()
        for iter in range(self.max_iter):
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, solver, damping, ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            if curr_error - next_error <= 0.001:
                break
            curr_error = next_error
    
    if self.use_ik_match_table2:
        # ... repeat with tasks2 ...
```

**PROBLEM #3.1: High damping (0.5) kills performance**
- Damping = regularization strength
- High damping → prefer joint velocity close to zero
- Effect: IK produces stiff, slow joint movements
- May not reach targets even with multiple iterations

**PROBLEM #3.2: Sequential IK (table1 THEN table2)**
- First: Solve with table1 (rotation-heavy constraints)
- Then: Re-solve with table2 (position-heavy constraints on feet)
- **Issue**: Position constraints on feet may conflict with table1 targets
- **Result**: Foot position accurate but upper body rotations wrong (or vice versa)

**PROBLEM #3.3: Only 10 max iterations**
- Complex IK problems (like full-body retargeting) need 20+ iterations
- 10 iterations insufficient for non-trivial poses
- Tasks may not fully converge

### 4.3 Task Error Calculation

```python
def error1(self):
    return np.linalg.norm(np.concatenate(
        [task.compute_error(self.configuration) for task in self.tasks1]
    ))
```

**PROBLEM #3.4: Position errors in different scales**
- Pelvis position error: ~meters
- Foot position error: ~meters
- Shoulder rotation error: ~quaternion distance (unitless)
- **Not normalized by body scale**
- Feet can have 0.1m error while shoulders perfectly matched
- Or vice versa

---

## 5. SMPL-X DATA LOADING AND SCALING (ERROR #4)

### 5.1 Scale Table (smplx_to_g1.json)

```json
"human_scale_table": {
    "pelvis": 0.9,
    "spine3": 0.9,
    "left_hip": 0.9,        "right_hip": 0.9,
    "left_knee": 0.9,       "right_knee": 0.9,
    "left_foot": 0.9,       "right_foot": 0.9,
    "left_shoulder": 0.8,   "right_shoulder": 0.8,
    "left_elbow": 0.8,      "right_elbow": 0.8,
    "left_wrist": 0.8,      "right_wrist": 0.8
}
```

**PROBLEM #4.1: Uniform scaling assumption**
- All bodies in left leg scaled by 0.9
- But G1 has different segment lengths:
  - Hip pitch → hip roll → hip yaw → knee → ankle_pitch → ankle_roll
  - Each has different geometry
- **SMPL-X has only 3 leg joints** (hip, knee, ankle)
- **Linear mapping** between SMPL-X joint positions and G1 DOFs breaks down

**PROBLEM #4.2: Scale applied to positions, not angles**
```python
# Line 243-266 in motion_retarget.py
scaled_root_pos = human_scale_table[human_root_name] * root_pos  # Correct
for body_name in human_data.keys():
    human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
```

- Scales position vectors
- **BUT**: Rotation/orientation NOT scaled (correct)
- **Issue**: Scaled position assumes uniform robot body size
- G1's actual body proportions differ from scaled SMPL-X

**PROBLEM #4.3: Auto height detection in gmr_retarget_headless.py**

```python
auto_human_height = 1.66 + 0.1 * betas[0]  # Default SMPL-X height formula
```

- Assumes SMPL-X beta[0] encodes height
- But motion_135_to_smplx.py creates betas=zeros(10)
- **Result: Always uses 1.66m even for actual human data of different heights**
- Can be overridden via --actual-human-height flag

---

## 6. COORDINATE FRAME CONVERSIONS IN gmr_to_protomotions.py (ERROR #5)

### 6.1 The _get_gmr_rot_offset() Function

```python
def _get_gmr_rot_offset():
    rot_offset_xyzw = np.array([-0.5, -0.5, -0.5, 0.5])  # xyzw
    return R.from_quat(rot_offset_xyzw)
```

This is the **inverse** of GMR's pelvis rot_offset:
- GMR pelvis rot_offset: [0.5, -0.5, -0.5, -0.5] (wxyz) = [-0.5, -0.5, -0.5, 0.5] (xyzw)
- Inverse: ???

**PROBLEM #5.1: Quaternion inverse vs frame rotation inverse**
```python
corrected = root_rots * rot_offset.inv()  # Quaternion multiplication
```

- Multiplies quaternion `root_rots * rot_offset^(-1)`
- Expects: frame conversion removal
- **But**: Quaternion multiplication order matters!
- **Issue**: Is this right-multiply or left-multiply semantics?
- **In scipy**: `q1 * q2` means "apply q1, then apply q2"
- **Result**: May be inverting direction of frame conversion

### 6.2 Root Position Frame Conversion

```python
def convert_root_pos_to_zup(root_pos):
    rot_offset = _get_gmr_rot_offset()
    return rot_offset.inv().apply(root_pos)
```

Maps:
```
[x, y, z]_smplx → rot_offset.inv().apply([x, y, z])
```

If rot_offset encodes: X→Z, Y→X, Z→Y, then:
- rot_offset.inv() should map: Z→X, X→Y, Y→Z
- But code says: applies rot_offset.inv()

**PROBLEM #5.2: Inconsistency in order**
- Root ROT removal: `root_rots * rot_offset.inv()` (right multiply)
- Root POS conversion: `rot_offset.inv().apply(root_pos)` (left apply)
- **These are NOT equivalent!**
- Right multiply of quaternion ≠ left apply of rotation matrix
- **Result: Root position and rotation in different frames**

### 6.3 FK Ground Correction (Lines 156-256)

This is where **TREMBLING happens**. Three modes:

1. **"global" mode** (DEFAULT):
   ```python
   global_offset = np.median(per_frame_offsets)
   corrected_root_pos[:, 2] = root_pos[:, 2] + global_offset
   ```
   - Single offset applied to all frames
   - Smooth but may leave some frames with feet off ground

2. **"smooth" mode**:
   ```python
   smooth_offsets = savgol_filter(per_frame_offsets, window_length=31, polyorder=3)
   corrected_root_pos[:, 2] = root_pos[:, 2] + smooth_offsets
   ```
   - Per-frame offset smoothed with Savitzky-Golay
   - Smoother than perframe but still changes per frame

3. **"perframe" mode**:
   ```python
   corrected_root_pos[:, 2] = root_pos[:, 2] + per_frame_offsets
   ```
   - Each frame independently adjusted
   - **Causes trembling**

**PROBLEM #5.3: FK assumes qpos already correct**
```python
for t in range(T):
    data.qpos[:3] = root_pos[t]
    data.qpos[3:7] = root_rot_wxyz
    data.qpos[7:] = dof_pos[t]
    mujoco.mj_forward(model, data)
    min_foot_z = data.xpos[foot_body_indices][2]
    foot_min_z_before[t] = min_foot_z
```

- Sets ROOT POSITION, then computes FK
- But root position already came from (flawed) IK
- **Result: Adjusting bad root position by measured foot error**
- **Like trying to correct a wrong answer by averaging guesses**

---

## 7. VELOCITY COMPUTATION (ERROR #6)

### 7.1 Savitzky-Golay Smoothing (from gmr_to_protomotions.py lines 371-457)

**FOR DOF VELOCITIES:**
```python
dof_vel = savgol_filter(dof_pos, window_length=win_len, polyorder=3, deriv=1, delta=dt, axis=0)
```

- Window length: 9-11 frames (0.18-0.22s at 50Hz)
- Polyorder: 3 (cubic fit)
- Delta: 0.02s (50Hz)
- **Issue #6.1**: Window too small for smooth derivatives
- **Issue #6.2**: Polyorder=3 may oscillate between fit points

**FOR BODY ANGULAR VELOCITY:**
```python
for t in range(1, T):
    drot = rots[t] * rots[t - 1].inv()
    body_ang_vel[t, b] = drot.as_rotvec() / dt
body_ang_vel[0, b] = body_ang_vel[1, b]  # Copy second frame
```

- Simple difference in quaternion space
- Converted to axis-angle via `as_rotvec()`
- **Issue #6.3**: First frame velocity is a copy (discontinuity)
- **Issue #6.4**: Angular velocity not smoothed after computation
- **Issue #6.5**: No normalization for quaternion scaling effects

### 7.2 Velocity Ramp at Boundaries (lines 439-455)

```python
ramp_frames = min(5, T // 4)
ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_frames) / ramp_frames))
dof_vel[:ramp_frames] *= ramp[:, None]
dof_vel[-ramp_frames:] *= ramp[::-1, None]
```

- Applies cosine ramp to first/last 5 frames
- Attenuates velocities at boundaries
- **Issue #6.6**: Artificially suppresses motion start/end
- **Result**: Can't represent actual startup/shutdown dynamics

---

## 8. JOINT LIMIT CLAMPING (ERROR #7)

From gmr_retarget_headless.py lines 85-120:

```python
def clamp_joint_limits(dof_pos, joint_order, joint_limits, soft=True):
    clamped = dof_pos.copy()
    for i, joint_name in enumerate(joint_order):
        lo, hi = joint_limits[joint_name]
        if soft:
            mid = (lo + hi) / 2.0
            half_range = (hi - lo) / 2.0
            scale = 0.9
            clamped[:, i] = mid + half_range * np.tanh((clamped[:, i] - mid) / (half_range * scale))
        else:
            clamped[:, i] = np.clip(clamped[:, i], lo, hi)
    return clamped, int(num_clamped)
```

**PROBLEM #7.1: "Soft" clamping via tanh is non-monotonic**
- tanh is smooth but asymptotic
- At limits, motion slows down dramatically (derivative → 0)
- **Result**: Artificial deceleration before hitting limits
- **Unrealistic for robot behavior**

**PROBLEM #7.2: Hard clipping used originally**
- Current code uses "soft=True" (tanh)
- But the comment mentions "hard np.clip" as original behavior
- **Hard clipping without smoothing = discontinuities in velocity**
- **Soft clipping = artificial motion slowdown**

**PROBLEM #7.3: Limits may be wrong**
```python
'left_hip_pitch_joint': (-2.5307, 2.8798),
'left_hip_roll_joint': (-0.5236, 2.9671),
...
```

- These should be verified against actual G1 URDF/MJCF
- Asymmetric limits (e.g., hip_pitch, hip_roll) may not match actual robot
- If limits are wrong, clamping ruins motion

---

## 9. ROOT POSITION COMPUTATION IN GMR (ERROR #8)

### 9.1 Ground Offset in motion_retarget.py

```python
def apply_ground_offset(self, human_data):
    for body_name in human_data.keys():
        pos, quat = human_data[body_name]
        human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
    return human_data
```

- Subtracts ground_offset from **ALL body Z positions**
- Ground offset set via `set_ground_offset(ground_offset)` in gmr_retarget_headless.py

### 9.2 Ground Offset Computation

```python
def compute_ground_offset(retarget, smplx_data_frames):
    offset = np.inf
    for frame_data in smplx_data_frames:
        human_data = retarget.to_numpy(frame_data)
        human_data = retarget.scale_human_data(...)
        human_data = retarget.offset_human_data(...)
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if pos[2] < offset:
                offset = pos[2]
    return offset
```

**PROBLEM #8.1: Finds GLOBAL minimum Z**
- Scans all frames and all bodies
- Takes absolute minimum Z position
- **Issue**: May be an outlier (one frame where foot is lower)
- **Result**: Most frames have feet above ground

**PROBLEM #8.2: Applied uniformly to all bodies**
- Same offset applied to pelvis, torso, feet
- **But**: Only feet should touch ground
- **Result**: If feet are at Z=0, pelvis is lowered too much
- **Or**: If pelvis is at right height, feet float above ground

---

## 10. BODY ORDERING & FK MISMATCH (ERROR #9)

### 10.1 MuJoCo Model Body Ordering

From g1_mocap_29dof.xml body hierarchy:
```
World (0)
├─ Pelvis (1) [freejoint]
│  ├─ left_hip_pitch_link (2)
│  │  └─ left_hip_roll_link (3)
│  │     └─ left_hip_yaw_link (4)
│  │        └─ left_knee_link (5)
│  │           └─ left_ankle_pitch_link (6)
│  │              └─ left_ankle_roll_link (7)
│  │                 └─ left_toe_link (8)
│  ├─ pelvis_contour_link (9)
│  ├─ right_hip_pitch_link (10)
│  │  └─ right_hip_roll_link (11)
│  │     └─ right_hip_yaw_link (12)
│  │        └─ right_knee_link (13)
│  │           └─ right_ankle_pitch_link (14)
│  │              └─ right_ankle_roll_link (15)
│  │                 └─ right_toe_link (16)
│  ├─ waist_yaw_link (17)
│  │  └─ waist_roll_link (18)
│  │     └─ torso_link (19)
│  │        ├─ head_link (20)
│  │        ├─ left_shoulder_pitch_link (21)
│  │        │  └─ left_shoulder_roll_link (22)
│  │        │     └─ left_shoulder_yaw_link (23)
│  │        │        └─ left_elbow_link (24)
│  │        │           ├─ left_wrist_roll_link (25)
│  │        │           ├─ left_wrist_pitch_link (26)
│  │        │           └─ left_wrist_yaw_link (27)
│  │        │              └─ left_rubber_hand (28)
│  │        └─ [symmetric for right arm]
└─ ... (33 total bodies)
```

### 10.2 FK Output Ordering

From gmr_to_protomotions.py lines 310-316:
```python
num_bodies = model.nbody - 1  # Exclude world body
body_pos_all = np.zeros((T, num_bodies, 3))
body_rot_all = np.zeros((T, num_bodies, 4))

for b in range(num_bodies):
    body_pos_all[t, b] = data.xpos[b + 1]          # +1 for world offset
    body_rot_all[t, b] = quat_wxyz_to_xyzw(data.xquat[b + 1])
```

**PROBLEM #9.1: Body 0 is pelvis (not world)**
- Assumes body index 0 = world in mujoco data
- But data.xpos[0] = pelvis position
- data.xpos[1] = left_hip_pitch_link position
- **Mismatch**: FK body ordering doesn't match stored index

**PROBLEM #9.2: 33 bodies assumed**
```python
"body_rot": body_rot_r,  # (T', 33, 4)
"body_pos": body_pos_r,  # (T', 33, 3)
```

- Hardcoded 33 in cache format
- **But**: Different MJCF files have different body counts
- g1_holo_compat.xml may have different structure than g1_mocap_29dof.xml
- **Result: Potential index out of bounds or silent data misalignment**

---

## 11. SMPL-X JOINT DEFINITION & RENDERING

### 11.1 SMPL-X Joints (22 total)
```
0:  Pelvis
1-3: Left leg (hip, knee, ankle)
4-6: Right leg (hip, knee, ankle)
7-9: Spine (spine1, spine2, spine3)
10-12: Left arm (shoulder, elbow, wrist)
13-15: Right arm (shoulder, elbow, wrist)
16: Neck
17: Head
18: Left eye
19: Right eye
20: Jaw
21: (unused)
```

### 11.2 Mapping to G1

**Left leg:**
- SMPL hip (pitch-roll-yaw) → G1 hip_pitch, hip_roll, hip_yaw (3 DOF)
- SMPL knee (1 DOF) → G1 knee (1 DOF) ✓
- SMPL ankle (1 DOF) → G1 ankle_pitch + ankle_roll (2 DOF) ✗

**PROBLEM #10.1: Over/under-constrained joints**
- SMPL ankle is 1 DOF but maps to G1 2 DOFs (pitch + roll)
- IK must infer ankle_roll from ankle_pitch
- **Result**: Ankle roll often zero or arbitrary**

**PROBLEM #10.2: No arm fingers**
- SMPL-X wrist (1 DOF) → G1 wrist_roll + wrist_pitch + wrist_yaw (3 DOF)
- G1 has fingers (in later versions) that SMPL doesn't model
- **Result**: Hand motion doesn't sync with finger motion**

---

## 12. SUMMARY OF CRITICAL ERRORS

### Tier 1: FUNDAMENTALLY WRONG
1. ✗ **Frame conversion incomplete** - IK targets in wrong frame
2. ✗ **Root position & rotation in different frames** - Inconsistent coordinate systems
3. ✗ **Joint mapping under-constrained** - Hip/ankle don't fully map to G1
4. ✗ **IK configuration wrong** - High damping, low iterations, no hip pitch/yaw targets

### Tier 2: CAUSES TREMBLING
5. ✗ **FK ground correction per-frame** - Root height jumps between frames
6. ✗ **Velocity computation naive** - Simple finite diff + tanh ramping
7. ✗ **Body position scaling** - SMPL positions scaled uniformly, G1 has different proportions
8. ✗ **Clamping at joint limits** - Soft clamping = artificial slowdown

### Tier 3: BUGS/ISSUES
9. ✗ **Ground offset global minimum** - May be based on outlier frame
10. ✗ **Body ordering mismatch** - FK output indices may not match cache indices
11. ✗ **No arm finger coordination** - Wrist doesn't map to finger motion
12. ✗ **Missing SMPL-X head/jaw** - Indices 16+ not retargeted to G1 neck/head

---

## 13. RECOMMENDATION FOR FIXING

### Phase 1: Coordinate Frame (CRITICAL)
1. **Define clear coordinate frames for each stage**
   - motion_135: ??? (verify from HyMotion code)
   - SMPL-X: Y-up, X-right, Z-forward
   - G1 MuJoCo: Z-up, X-forward, Y-left
   
2. **Pre-process motion_135 to Y-up SMPL**
   - Add frame conversion BEFORE gmr_retarget_headless.py
   - Ensure motion_135 is actually in expected frame

3. **Fix IK target setup**
   - Set targets in correct frame INSIDE IK solver
   - Not post-hoc conversion of IK output

### Phase 2: Joint Mapping (CRITICAL)
1. **Add explicit hip pitch/yaw targets to IK config**
   - Update smplx_to_g1.json to include hip_pitch_link and hip_yaw_link
   - Set appropriate position weights (0 for pitch/yaw hip frames)

2. **Add ankle roll constraint**
   - Explicitly target ankle_roll from SMPL ankle orientation
   - Currently inferred, should be explicit

3. **Validate joint limits**
   - Cross-check G1_JOINT_LIMITS against actual URDF

### Phase 3: IK Solver (HIGH PRIORITY)
1. **Reduce damping** from 0.5 → 0.1
2. **Increase max iterations** from 10 → 30
3. **Add task error normalization** by body scale
4. **Merge IK match tables** - solve in single pass instead of two

### Phase 4: Ground Correction (HIGH PRIORITY)
1. **Use smooth mode by default** (not perframe)
2. **Validate ground offset computation**
   - Should be median of per-frame offsets, not global minimum
3. **Re-run FK after ground correction**
   - Current code doesn't verify feet are actually at ground post-correction

### Phase 5: Velocity (MEDIUM)
1. **Validate velocity smoothing**
   - Window length = 9 is too small
   - Increase to 11-15 frames
2. **Remove velocity ramp** - artificially suppresses motion
3. **Smooth angular velocity before ramp**

---

## 14. VALIDATION TESTS

1. **Visual inspection**: Compare SMPL skeleton to G1 skeleton in viewer
   - Should have similar body proportions
   - Feet should be at ground level
   - Center of gravity should be reasonable

2. **Joint range check**: Log all joint values
   - Should be within gmr_retarget_headless.py limits
   - Should have smooth derivatives
   - No discontinuities

3. **Physics simulation check**: Drop robot on ground
   - Should have reasonable physics (not sinking or bouncing)
   - Center of mass should balance on feet

4. **Coordinate frame check**: Render both SMPL and G1 side-by-side
   - Bodies should overlap (same pose)
   - Not rotated 90° relative to each other

