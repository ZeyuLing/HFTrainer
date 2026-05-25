# SMPL MuJoCo Physics Simulation Reference Guide
## Comprehensive Analysis of Key Reference Files

**Generated:** 2026/05/14  
**Repository Root:** `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/`

---

## 1. SMPL-MuJoCo Converter Architecture
### File: `ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py`

#### 1.1 SMPLConverter Class (Lines 35-318)

**Purpose:** Converts SMPL pose representations ↔ MuJoCo qpos format with PD controller gains.

**Key Attributes:**

```python
class SMPLConverter:
    # Body weightings for loss computation
    self.body_ws = {
        "Pelvis": 1.0,
        "L_Hip": 1.0, "R_Hip": 1.0,
        "L_Knee": 1.0, "R_Knee": 1.0,
        "L_Ankle": 1.0, "R_Ankle": 1.0,
        "Torso": 1.0, "Spine": 1.0, "Chest": 1.0,
        "Neck": 1.0, "Head": 1.0,
        "L_Thorax": 1.0, "R_Thorax": 1.0,
        "L_Shoulder": 1.0, "R_Shoulder": 1.0,
        "L_Elbow": 1.0, "R_Elbow": 1.0,
        "L_Wrist": 1.0, "R_Wrist": 1.0,
        "L_Toe": 0.0, "R_Toe": 0.0,  # End effectors (not active)
        "L_Hand": 0.0, "R_Hand": 0.0,
    }
    
    # PD gains: [Kp, Kd, action_scale, torque_limit]
    self.body_params = {
        "L_Hip": [500, 50, 1, 500],        # Strong hip control
        "L_Knee": [500, 50, 1, 500],       # Knee stiffness = hip
        "L_Ankle": [400, 40, 1, 500],      # Slightly softer
        "Torso": [1000, 100, 1, 500],      # Very stiff torso
        "Spine": [1000, 100, 1, 500],      # Very stiff spine
        "Chest": [1000, 100, 1, 500],      # Very stiff chest
        "Neck": [100, 10, 1, 250],         # Soft neck control
        "Head": [100, 10, 1, 250],         # Soft head control
        "L_Shoulder": [400, 40, 1, 500],   # Moderate shoulder
        "L_Elbow": [300, 30, 1, 150],      # Lower elbow torque
        "L_Wrist": [100, 10, 1, 150],      # Very soft wrist
        # ... (symmetrical for right side)
    }
```

**SMPL Bone Order (22 joints from SMPL_BONE_ORDER_NAMES):**
```
Index  Joint Name
0      Pelvis (root)
1-2    Left Hip (x, y)
3      Left Knee
4-5    Right Hip (x, y)
6      Right Knee
7-9    Torso, Spine, Chest
10-11  Neck, Head
12-14  Left Shoulder, Elbow, Wrist
15-17  Right Shoulder, Elbow, Wrist
18-21  Left/Right Toe, Hand (end effectors)
```

**Critical Methods:**

- `get_new_jkp()` (lines 300-303): Extract joint stiffness Kp as 3D per DOF
  ```python
  return [[body_params[joint][0]] * 3 for joint in body_names]
  ```

- `get_new_jkd()` (lines 305-308): Extract joint damping Kd
  ```python
  return [[body_params[joint][1]] * 3 for joint in body_names]
  ```

- `get_new_a_scale()` (lines 310-313): Action scaling per DOF

- `get_new_torque_limit()` (lines 315-318): Maximum torque per DOF

---

#### 1.2 Core Conversion Functions

##### **smpl_to_qpose()** (Lines 331-405)
**Signature:**
```python
def smpl_to_qpose(
    pose,           # (batch, 72) or (batch, 156) axis-angle SMPL poses
    mj_model,       # MuJoCo model object
    trans=None,     # (batch, 3) translation [default: [0, 0, 0.91437225]]
    normalize=False,
    random_root=False,
    count_offset=True,
    use_quat=False,
    euler_order="ZYX",
    model="smpl"
) -> np.ndarray: # (batch, nq) MuJoCo qpos format
```

**Key Processing Steps:**

1. **Pose Format Validation** (lines 356-367):
   - If `model="smpl"` and pose shape is (batch, 156) → convert SMPLH to SMPL via `smplh_to_smpl()`
   - If `model="smplh"` and pose shape is (batch, 72) → convert SMPL to SMPLH via `smpl_to_smplh()`

2. **Rotation Matrix Conversion** (lines 378-389):
   ```python
   curr_pose_mat = angle_axis_to_rotation_matrix(pose.reshape(-1, 3))
                  # (batch*22, 3) → (batch, 22, 4, 4) rotation matrices
   
   # Convert rotation matrices to euler angles (ZYX convention)
   curr_spose = sRot.from_matrix(curr_pose_mat[:, :, :3, :3])
              .as_euler("ZYX", degrees=False)
                  # (batch, 22, 3) euler angles
   ```

3. **SMPL → MuJoCo Joint Reordering** (lines 371-393):
   ```python
   smpl_2_mujoco = [
       joint_names.index(q) for q in list(get_body_qposaddr(mj_model).keys())
       if q in joint_names
   ]
   # Example output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
   # Selects which SMPL joints to include in MuJoCo qpos
   
   curr_spose = curr_spose[:, smpl_2_mujoco, :].reshape(-1, num_angles)
   ```

4. **QPos Construction** (lines 394-399):
   ```python
   if use_quat:
       curr_qpos = np.concatenate([trans, curr_spose], axis=1)
       # (batch, 3 + 22*4) for quaternion representation
   else:
       root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
       curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]), axis=1)
       # (batch, 3 + 4 + 21*3) = (batch, 70) for mixed quat/euler
       # Root: 3D translation + 4D quat (wxyz format)
       # Body joints: 21 × 3D euler (ZYX convention)
   ```

5. **Offset Application** (lines 401-404):
   ```python
   if count_offset:
       curr_qpos[:, :3] = trans + mj_model.body_pos[1]
       # body_pos[1] = offset of first body (Pelvis) from origin
   ```

**Output Format (70-dim for humanoid):**
```
[0:3]    - Translation (xyz)
[3:7]    - Root quaternion (wxyz)
[7:10]   - Torso euler (ZYX)
[10:13]  - Spine euler (ZYX)
[13:16]  - Chest euler (ZYX)
[16:19]  - Neck euler (ZYX)
[19:22]  - Head euler (ZYX)
[22:25]  - L_Thorax euler (ZYX)
[25:28]  - L_Shoulder euler (ZYX)
[28:31]  - L_Elbow euler (ZYX)
[31:34]  - L_Wrist euler (ZYX)
[34:37]  - R_Thorax euler (ZYX)
[37:40]  - R_Shoulder euler (ZYX)
[40:43]  - R_Elbow euler (ZYX)
[43:46]  - R_Wrist euler (ZYX)
[46:49]  - L_Hip euler (ZYX)
[49:52]  - L_Knee euler (ZYX)
[52:55]  - L_Ankle euler (ZYX)
[55:58]  - R_Hip euler (ZYX)
[58:61]  - R_Knee euler (ZYX)
[61:64]  - R_Ankle euler (ZYX)
[64:67]  - L_Toe euler (ZYX)     [optional if included]
[67:70]  - R_Toe euler (ZYX)     [optional if included]
```

---

##### **qpos_to_smpl()** (Lines 552-571)
**Signature:**
```python
def qpos_to_smpl(
    qpos,         # (batch, nq) MuJoCo state
    mj_model,     # MuJoCo model
    smpl_model="smpl"  # "smpl" or "smplh"
) -> Tuple[np.ndarray, np.ndarray]:
    # Returns: (pose_aa, trans)
    # pose_aa: (batch, 22, 3) axis-angle SMPL pose
    # trans:   (batch, 3) translation
```

**Reverse Conversion Logic:**

```python
# Line 555
trans = qpos[:, :3] - mj_model.body_pos[1]  # Remove body offset

# Lines 559-569
pose = np.zeros([batch_size, len(smpl_bones_to_use), 3])
for ind1, bone_name in enumerate(smpl_bones_to_use):
    ind2 = body_qposaddr[bone_name]  # Get qpos indices for this joint
    if ind1 == 0:  # Root joint
        quat = qpos[:, 3:7]  # Extract root quaternion (wxyz)
        pose[:, ind1, :] = sRot.from_quat(quat[:, [1, 2, 3, 0]])  # wxyz -> xyzw
                                        .as_rotvec()  # quaternion -> axis-angle
    else:  # Non-root joints
        pose[:, ind1, :] = sRot.from_euler("ZYX", qpos[:, ind2[0]:ind2[1]])
                                          .as_rotvec()  # euler -> axis-angle
```

---

## 2. MuJoCo SMPL Humanoid Model Structure
### File: `ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml`

#### 2.1 Body Tree Structure

```
Pelvis (freejoint) [lines 27-28]
├─ L_Hip (3 DOF hinge: x,y,z) [lines 30-52]
│  ├─ L_Knee (3 DOF hinge: x,y,z) [lines 35-52]
│  │  └─ L_Ankle (3 DOF hinge: x,y,z) [lines 40-51]
│  │     └─ L_Toe (3 DOF hinge: x,y,z) [lines 45-50]
│  └─ Pelvis children (cont.)
└─ R_Hip (3 DOF hinge: x,y,z) [lines 54-76]
   ├─ R_Knee (3 DOF hinge: x,y,z) [lines 59-75]
   │  └─ R_Ankle (3 DOF hinge: x,y,z) [lines 64-74]
   │     └─ R_Toe (3 DOF hinge: x,y,z) [lines 69-74]
   └─ Pelvis children (cont.)
└─ Torso (3 DOF hinge: x,y,z) [lines 78-167]
   └─ Spine (3 DOF hinge: x,y,z) [lines 83-166]
      └─ Chest (3 DOF hinge: x,y,z) [lines 88-165]
         ├─ Neck (3 DOF hinge: x,y,z) [lines 93-103]
         │  └─ Head (3 DOF hinge: x,y,z) [lines 98-103]
         ├─ L_Thorax (3 DOF hinge: x,y,z) [lines 105-133]
         │  └─ L_Shoulder (3 DOF hinge: x,y,z) [lines 110-133]
         │     └─ L_Elbow (3 DOF hinge: x,y,z) [lines 115-132]
         │        └─ L_Wrist (3 DOF hinge: x,y,z) [lines 120-131]
         │           └─ L_Hand (3 DOF hinge: x,y,z) [lines 125-130]
         └─ R_Thorax (3 DOF hinge: x,y,z) [lines 135-163]
            └─ R_Shoulder (3 DOF hinge: x,y,z) [lines 140-163]
               └─ R_Elbow (3 DOF hinge: x,y,z) [lines 145-162]
                  └─ R_Wrist (3 DOF hinge: x,y,z) [lines 150-161]
                     └─ R_Hand (3 DOF hinge: x,y,z) [lines 155-160]
```

#### 2.2 Joint Configuration Details

**Pelvis (freejoint) - Line 28:**
- 6 DOF: [x, y, z translation] + [quaternion rotation]
- **Special:** Root body with free-floating base

**Example: L_Hip Joints (Lines 31-33):**
```xml
<joint name="L_Hip_x" type="hinge" axis="1 0 0" 
       stiffness="800" damping="80" armature="0.02" 
       range="-90.0 90.0"/>
<joint name="L_Hip_y" type="hinge" axis="0 1 0" 
       stiffness="800" damping="80" armature="0.02" 
       range="-90.0 90.0"/>
<joint name="L_Hip_z" type="hinge" axis="0 0 1" 
       stiffness="800" damping="80" armature="0.02" 
       range="-90.0 90.0"/>
```

**Joint Parameters Across Body:**

| Joint Group | Stiffness | Damping | Armature | Range (degrees) |
|---|---|---|---|---|
| L/R_Hip | 800 | 80 | 0.02 | ±90 |
| L/R_Knee | 800 | 80 | 0.02 | 0–180 (limited) |
| L/R_Ankle | 800 | 80 | 0.02 | ±45 |
| L/R_Toe | 500 | 50 | 0.02 | ±180 |
| Torso/Spine/Chest | 1000 | 100 | 0.02 | ±60 |
| Neck | 500 | 50 | 0.02 | ±5.625 (very limited) |
| Head | 500 | 50 | 0.02 | ±5.625 (very limited) |
| L/R_Shoulder | 500 | 50 | 0.02 | ±720 (unrestricted) |
| L/R_Elbow | 500 | 50 | 0.02 | ±5.625–180 (limited) |
| L/R_Wrist/Hand | 300 | 30 | 0.02 | ±180 |

#### 2.3 Actuators

**Total Actuators: 69** (Lines 170-239)
- **Structure:** 3 motors per joint (x, y, z rotations)
- **Actuator Type:** `motor` (torque control)
- **Gear Ratio:** All motors have `gear="500"`
- **Naming Convention:** `{JointName}_x`, `{JointName}_y`, `{JointName}_z`

**Control Range:** Each motor has `ctrlrange="-1 1"` (normalized, scaled by gear)

**Example (L_Hip motors, lines 171-173):**
```xml
<motor name="L_Hip_x" joint="L_Hip_x" gear="500"/>
<motor name="L_Hip_y" joint="L_Hip_y" gear="500"/>
<motor name="L_Hip_z" joint="L_Hip_z" gear="500"/>
```

#### 2.4 QPos Layout (70-dim total)

```
Pelvis freejoint:  [0:7]   = [tx, ty, tz, qw, qx, qy, qz]
L_Hip (3 DOF):     [7:10]  = [L_Hip_x, L_Hip_y, L_Hip_z]
L_Knee (3 DOF):    [10:13] = [L_Knee_x, L_Knee_y, L_Knee_z]
L_Ankle (3 DOF):   [13:16] = [L_Ankle_x, L_Ankle_y, L_Ankle_z]
L_Toe (3 DOF):     [16:19] = [L_Toe_x, L_Toe_y, L_Toe_z]
R_Hip (3 DOF):     [19:22] = [R_Hip_x, R_Hip_y, R_Hip_z]
R_Knee (3 DOF):    [22:25] = [R_Knee_x, R_Knee_y, R_Knee_z]
R_Ankle (3 DOF):   [25:28] = [R_Ankle_x, R_Ankle_y, R_Ankle_z]
R_Toe (3 DOF):     [28:31] = [R_Toe_x, R_Toe_y, R_Toe_z]
Torso (3 DOF):     [31:34] = [Torso_x, Torso_y, Torso_z]
Spine (3 DOF):     [34:37] = [Spine_x, Spine_y, Spine_z]
Chest (3 DOF):     [37:40] = [Chest_x, Chest_y, Chest_z]
Neck (3 DOF):      [40:43] = [Neck_x, Neck_y, Neck_z]
Head (3 DOF):      [43:46] = [Head_x, Head_y, Head_z]
L_Thorax (3 DOF):  [46:49] = [L_Thorax_x, L_Thorax_y, L_Thorax_z]
L_Shoulder (3 DOF):[49:52] = [L_Shoulder_x, L_Shoulder_y, L_Shoulder_z]
L_Elbow (3 DOF):   [52:55] = [L_Elbow_x, L_Elbow_y, L_Elbow_z]
L_Wrist (3 DOF):   [55:58] = [L_Wrist_x, L_Wrist_y, L_Wrist_z]
L_Hand (3 DOF):    [58:61] = [L_Hand_x, L_Hand_y, L_Hand_z]
R_Thorax (3 DOF):  [61:64] = [R_Thorax_x, R_Thorax_y, R_Thorax_z]
R_Shoulder (3 DOF):[64:67] = [R_Shoulder_x, R_Shoulder_y, R_Shoulder_z]
R_Elbow (3 DOF):   [67:70] = [R_Elbow_x, R_Elbow_y, R_Elbow_z]
R_Wrist (3 DOF):   [70:73] = [R_Wrist_x, R_Wrist_y, R_Wrist_z]
R_Hand (3 DOF):    [73:76] = [R_Hand_x, R_Hand_y, R_Hand_z]
```

---

## 3. Motion Conversion Utilities
### File: `scripts/embodied/motion135_to_smplx.py`

#### 3.1 Rotation Representation Conversion

**Input Format (HyMotion motion_135):**
- Shape: `(T, 135)` where `T` = number of frames
- Layout: `[translation(3), rotation_6d_joint1(6), ..., rotation_6d_joint22(6)]`
- **Rotation encoding:** 6D row-major layout: `[R00, R01, R10, R11, R20, R21]`

##### **rot6d_to_rotmat()** (Lines 26-55)
```python
def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """
    Input:  (..., 6) array in row-major layout
    Output: (..., 3, 3) rotation matrix
    
    Process:
    1. Reorder [0,2,4,1,3,5] to convert row-major → column-major
    2. Extract first two columns: a1 = rot6d[..., :3], a2 = rot6d[..., 3:6]
    3. Normalize first column: b1 = a1 / ||a1||
    4. Gram-Schmidt orthogonalization for second column:
       dot = b1 · a2
       b2 = (a2 - dot·b1) / ||a2 - dot·b1||
    5. Cross product for third column: b3 = b1 × b2
    6. Stack columns: R = [b1 | b2 | b3]
    """
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]  # Reorder for Gram-Schmidt
    a1 = rot6d[..., :3]                      # First two columns
    a2 = rot6d[..., 3:6]
    
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    
    rotmat = np.stack([b1, b2, b3], axis=-1)
    return rotmat  # (..., 3, 3)
```

##### **rotmat_to_axis_angle()** (Lines 58-66)
```python
def rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """
    Input:  (..., 3, 3) rotation matrices
    Output: (..., 3) axis-angle vectors (rotvec)
    
    Uses scipy.spatial.transform.Rotation
    """
    from scipy.spatial.transform import Rotation as R
    
    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    rot = R.from_matrix(rotmat_flat)
    aa_flat = rot.as_rotvec()
    return aa_flat.reshape(*orig_shape, 3)  # (..., 3) axis-angle
```

#### 3.2 Full Conversion Pipeline

**convert_motion135_to_smplx()** (Lines 69-110)

```python
def convert_motion135_to_smplx(input_npz, output_npz, fps=30):
    """
    Input NPZ: motion_135 (T, 135) format
    Output NPZ: SMPL-X format for GMR retargeting
    
    Steps:
    1. Load motion_135: (T, 135)
    2. Split: transl (T, 3), rot6d (T, 22, 6)
    3. Convert rot6d → rotation matrix (T, 22, 3, 3)
    4. Convert rotation matrix → axis-angle (T, 22, 3)
    5. Split root vs body: root_orient (T, 3), pose_body (T, 63)
    6. Save as SMPL-X NPZ with:
       - pose_body: (T, 63) axis-angle (21 joints × 3)
       - root_orient: (T, 3) axis-angle (root)
       - trans: (T, 3) translation
       - betas: (10,) zeros
       - gender: "neutral"
       - mocap_frame_rate: fps
    """
    data = np.load(input_npz, allow_pickle=True)
    motion = data['motion_135']  # (T, 135)
    
    transl = motion[:, :3]                    # (T, 3)
    rot6d = motion[:, 3:].reshape(T, 22, 6)   # (T, 22, 6)
    
    rotmat = rot6d_to_rotmat(rot6d)            # (T, 22, 3, 3)
    aa = rotmat_to_axis_angle(rotmat)          # (T, 22, 3)
    
    root_orient = aa[:, 0, :]                  # (T, 3) pelvis
    pose_body = aa[:, 1:22, :].reshape(T, -1) # (T, 63) 21 joints
    
    np.savez(output_npz,
        pose_body=pose_body.astype(np.float32),
        root_orient=root_orient.astype(np.float32),
        trans=transl.astype(np.float32),
        betas=np.zeros(10, dtype=np.float32),
        gender="neutral",
        mocap_frame_rate=np.array(fps))
```

---

## 4. MuJoCo Physics Simulation Loop
### File: `scripts/embodied/run_tracker_export.py`

#### 4.1 Model Loading and Configuration

##### **load_mujoco_model_for_sim()** (Lines 120-181)
```python
def load_mujoco_model_for_sim(
    mjcf_path: str,
    stiffness: list,        # [Kp for each DOF]
    damping: list,          # [Kd for each DOF]
    physics_dt: float
) -> Tuple[MjModel, MjData]:
    """
    Loads MuJoCo model and configures implicit PD actuators.
    
    Key steps:
    1. Patch MJCF XML (remove sensors, add ground/light)
    2. Create temporary XML with patched content
    3. Load model via mujoco.MjModel.from_xml_path()
    4. Create MjData for this model
    5. Set physics timestep (typically 0.005s = 200Hz)
    6. Zero passive forces (joint stiffness, damping, friction)
    7. Configure PD actuators with user-provided Kp/Kd gains
    """
    import tempfile
    mjcf_file = Path(mjcf_path)
    patched_xml = _patch_mjcf_xml(mjcf_file)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", 
                                     dir=str(mjcf_file.parent),
                                     delete=False) as tmp:
        tmp.write(patched_xml)
        tmp_path = tmp.name
    
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    
    data = mujoco.MjData(model)
    model.opt.timestep = physics_dt
    
    # Zero passive forces
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0
    
    # Configure PD actuators
    for i in range(model.nu):
        kp = stiffness[i]
        kd = damping[i]
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biastype[i] = 1
        model.actuator_biasprm[i, 0] = 0.0      # constant term
        model.actuator_biasprm[i, 1] = -kp      # position feedback gain
        model.actuator_biasprm[i, 2] = -kd      # velocity feedback gain
        model.actuator_ctrllimited[i] = 0       # unlimited ctrl range
    
    return model, data
```

**PD Control Equation Implemented in MuJoCo:**
```
τ = Kp * (q_target - q_current) - Kd * q_dot + gain * u_ctrl
```

#### 4.2 Main Simulation Loop

##### **run_tracker_and_export()** (Lines 189-491)

**Core Loop Structure (Lines 319-445):**

```python
for frame_idx in range(num_frames):
    # ---- 1. Record current state BEFORE stepping ----
    out_body_pos[frame_idx] = data.xpos[1:num_bodies+1].copy()
    out_body_rot[frame_idx] = mujoco_wxyz_to_xyzw(data.xquat[1:num_bodies+1])
    out_dof_pos[frame_idx] = data.qpos[7:].copy()
    out_dof_vel[frame_idx] = data.qvel[6:].copy()
    
    # ---- 2. Fall detection ----
    root_h = float(data.qpos[2])
    if root_h < FALL_HEIGHT_THRESHOLD and fall_frame is None:
        fall_frame = frame_idx
    
    # ---- 3. Read robot state for policy ----
    robot_state = {
        "dof_pos": data.qpos[7:].astype(np.float32),
        "dof_vel": data.qvel[6:].astype(np.float32),
        "body_rot": out_body_rot[frame_idx],
        "root_local_ang_vel": data.qvel[3:6].astype(np.float32),
    }
    
    # ---- 4. Compute heading offset (first step only) ----
    if heading_offset is None:
        heading_offset = compute_yaw_offset_np(
            robot_state["body_rot"][anchor_body_index],
            player.get_state_at_frame(0)["body_rot"][anchor_body_index]
        )
    
    # ---- 5. Get future motion references ----
    future_refs = player.get_future_references(frame_idx, future_step_indices)
    future_refs["body_rot"] = apply_heading_offset_np(
        heading_offset, future_refs["body_rot"]
    )
    
    # ---- 6. Build ONNX policy inputs ----
    key_to_array = {
        "current.dof_pos": robot_state["dof_pos"][None],
        "current.dof_vel": robot_state["dof_vel"][None],
        "current.anchor_rot": anchor_rot[None],
        "current.root_local_ang_vel": robot_state["root_local_ang_vel"][None],
        "historical.processed_actions": prev_actions_input[None, None],
        "mimic.future_anchor_rot": future_anchor_rot[None],
        "mimic.future_rot": future_refs["body_rot"][None],
        "mimic.future_dof_pos": future_refs["dof_pos"][None],
        "mimic.future_dof_vel": future_refs["dof_vel"][None],
    }
    
    # ---- 7. Run ONNX tracker policy ----
    ort_out = session.run(actual_out_names, onnx_inputs)
    pd_targets = ort_out[1].squeeze().copy()  # Joint position targets
    
    # ---- 8. Optional: PD target acceleration clamping ----
    if pd_target_max_accel is not None and prev_pd is not None:
        delta = pd_targets - prev_pd
        prev_delta = prev_pd - prev_prev_pd
        accel = delta - prev_delta
        clamped_accel = np.clip(accel, -pd_target_max_accel, pd_target_max_accel)
        pd_targets = prev_pd + prev_delta + clamped_accel
    
    # ---- 9. Optional: EMA action filtering ----
    if use_ema:
        pd_targets = (action_ema_alpha * pd_targets + 
                     (1.0 - action_ema_alpha) * ema_prev_targets)
    
    # ---- 10. Apply control and step physics ----
    data.ctrl[:] = pd_targets
    for _ in range(decimation):
        mujoco.mj_step(model, data)
```

**State Recording Details:**

```python
# Lines 322-337
# Body positions (world frame)
out_body_pos[frame_idx] = data.xpos[1:num_bodies+1]  # (num_bodies, 3)

# Body rotations (world frame, convert wxyz -> xyzw)
body_rot_wxyz = data.xquat[1:num_bodies+1]
out_body_rot[frame_idx] = mujoco_wxyz_to_xyzw(body_rot_wxyz)  # (num_bodies, 4)

# DOF positions (only actuated joints, skip root [0:7])
out_dof_pos[frame_idx] = data.qpos[7:]  # num_dofs

# DOF velocities (only actuated joints)
out_dof_vel[frame_idx] = data.qvel[6:]  # num_dofs (skip root linear vel)

# Body velocities from cvel [ang_vel(3), lin_vel(3)]
cvel = data.cvel[1:num_bodies+1]
out_body_ang_vel[frame_idx] = cvel[:, 0:3]
out_body_vel[frame_idx] = cvel[:, 3:6]
```

#### 4.3 Output Cache Format

**Saved Tracked Cache (Lines 457-469):**
```python
tracked_cache = {
    "dof_pos":        (num_frames, num_dofs) float32,
    "dof_vel":        (num_frames, num_dofs) float32,
    "body_rot":       (num_frames, num_bodies, 4) float32 [xyzw],
    "body_pos":       (num_frames, num_bodies, 3) float32,
    "body_vel":       (num_frames, num_bodies, 3) float32,
    "body_ang_vel":   (num_frames, num_bodies, 3) float32,
    "control_dt":     float,  # Control timestep (e.g., 0.02s)
    "num_frames":     int,    # Total number of frames
}
torch.save(tracked_cache, output_path)
```

---

## 5. Summary: Data Flow for SMPL → MuJoCo → Tracked Physics

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SMPL Motion Input (motion_135 format)                   │
│    shape: (T, 135) = [trans(3) + rot6d_22joints(132)]     │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Rotation Conversion (rot6d → rotmat → axis-angle)       │
│    [motion135_to_smplx.py]                                 │
│    - rot6d_to_rotmat(): row-major → Gram-Schmidt           │
│    - rotmat_to_axis_angle(): scipy.spatial.transform       │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. SMPL Pose to MuJoCo QPos (smpl_to_qpose)               │
│    [smpl_mujoco.py]                                        │
│    - Convert axis-angle → rotation matrix                  │
│    - Rotate to euler (ZYX convention)                      │
│    - Reorder to MuJoCo joint order                         │
│    - Output: (T, 70) qpos = [transl(3) + quat(4) + euler(63)]
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Load MuJoCo Model with PD Controllers                   │
│    [load_mujoco_model_for_sim]                             │
│    - Load MJCF XML (smpl_humanoid.xml)                     │
│    - Patch XML (sensors, ground, light)                    │
│    - Configure PD gains from body_params dict             │
│    τ = Kp(q_target - q) - Kd*qdot                         │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Physics Simulation Loop (per frame)                     │
│    [run_tracker_and_export]                                │
│    - ONNX policy: (state, future_refs) → pd_targets       │
│    - Apply PD control: data.ctrl = pd_targets             │
│    - Step physics: mujoco.mj_step() × decimation          │
│    - Record: body_pos, body_rot, dof_pos, dof_vel        │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Export Tracked Motion Cache                             │
│    - Format: {dof_pos, dof_vel, body_rot, body_pos, ...}  │
│    - Type: .pt torch tensor file                           │
│    - Use: convert_cache_to_json.py → Three.js viz         │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Key Implementation Notes

### 6.1 Rotation Conventions
- **SMPL Input:** Axis-angle (rotvec)
- **MuJoCo Root:** Quaternion (wxyz)
- **MuJoCo Joints:** Euler (ZYX order for ZYX convention)
- **Gram-Schmidt:** Required for numerically stable 6D→3×3 conversion

### 6.2 Physics Parameters
- **Timestep:** 0.005s (200 Hz physics), 0.02s (50 Hz control)
- **Decimation:** Physics steps per control frame (typical: 4×)
- **PD Gains:** Stiffness [100-1000], Damping [10-100]
- **Torque Limits:** 150-500 N⋅m per DOF

### 6.3 State Representation
- **QPos:** [tx, ty, tz, qw, qx, qy, qz] + 63 euler angles = 70 dims
- **QVel:** [vx, vy, vz, ωx, ωy, ωz] + 63 joint velocities = 69 dims
- **Body Indices:** [0]=World, [1]=Pelvis, [2:]=Limbs

### 6.4 Conversion Pitfalls
1. **Quaternion Order:** wxyz for MuJoCo, xyzw for output → always reorder!
2. **Body Offset:** `mj_model.body_pos[1]` = Pelvis offset from origin
3. **Euler Convention:** ZYX ≠ XYZ → verify consistency
4. **6D Reordering:** Row-major [R00,R01,R10,R11,R20,R21] → Column-major for Gram-Schmidt

---

**END OF REPORT**
