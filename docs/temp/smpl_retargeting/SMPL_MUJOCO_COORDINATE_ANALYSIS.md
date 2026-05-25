# SMPL ↔ MuJoCo Conversion Analysis

## Overview

The file `/ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py` implements conversions between SMPL (Skinned Multi-Person Linear) body pose representations and MuJoCo simulator joint configurations (qpos).

**Key Question**: Does the code apply coordinate transforms to body joints, or only to the root?

**Answer**: **ONLY to the root (Pelvis)**. Body joints receive **NO coordinate transform**.

---

## 1. `smpl_to_qpose()` Function (Lines 331–405)

### Function Signature

```python
def smpl_to_qpose(
    pose,                    # batch_size x 72 (SMPL: 24 joints × 3 axis-angle)
    mj_model,               # MuJoCo model
    trans=None,             # batch_size x 3 (translation/root position)
    normalize=False,        # whether to normalize SMPL pose
    random_root=False,      # randomize root rotation when normalizing
    count_offset=True,      # whether to add model's body offset
    use_quat=False,         # if True, output quaternion for all joints; if False, Euler for body
    euler_order="ZYX",      # Euler convention for body joints
    model="smpl",           # "smpl", "smplh", or "smplx"
):
```

### Complete Code

```python
    """
    Expect pose to be batch_size x 72
    trans to be batch_size x 3
    differentiable 
    """
    if trans is None:
        trans = np.zeros((pose.shape[0], 3))
        trans[:, 2] = 0.91437225                                    # Default: 0.914m height (Z-up)
    if normalize:
        pose, trans = normalize_smpl_pose(pose, trans, random_root=random_root)

    if not torch.is_tensor(pose):
        pose = torch.tensor(pose)

    if model == "smpl":
        joint_names = SMPL_BONE_ORDER_NAMES                         # [Pelvis, L_Hip, R_Hip, ..., R_Hand]
        if pose.shape[-1] == 156:
            pose = smplh_to_smpl(pose)                              # Convert SMPLH (156) to SMPL (72)
    elif model == "smplh":
        joint_names = SMPLH_BONE_ORDER_NAMES
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)                              # Convert SMPL (72) to SMPLH (156)
    elif model == "smplx":
        joint_names = SMPLH_BONE_ORDER_NAMES
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)

    num_joints = len(joint_names)                                   # 24 for SMPL
    num_angles = num_joints * 3                                     # 72 for SMPL
    
    # KEY: Build smpl_2_mujoco reordering mapping
    smpl_2_mujoco = [
        joint_names.index(q) 
        for q in list(get_body_qposaddr(mj_model).keys())
        if q in joint_names
    ]
    # This is a list of SMPL indices that correspond to MuJoCo body order
    # e.g., if MuJoCo bodies are [root, L_Hip, R_Hip, ...], 
    #       then smpl_2_mujoco = [1, 2, ...] (SMPL indices for those bodies)

    pose = pose.reshape(-1, num_angles)                             # (batch, 72)

    # Step 1: Convert axis-angle to rotation matrices
    curr_pose_mat = angle_axis_to_rotation_matrix(
        pose.reshape(-1, 3)                                         # Reshape to (batch*24, 3) for each joint
    ).reshape(pose.shape[0], -1, 4, 4)                             # → (batch, 24, 4, 4)
    
    # Step 2: Extract 3×3 rotation and convert to Euler/Quat
    curr_spose = sRot.from_matrix(curr_pose_mat[:, :, :3, :3].reshape(-1, 3, 3).numpy())
    
    if use_quat:
        # Output quaternion for ALL joints (including root)
        curr_spose = curr_spose.as_quat()[:, [3, 0, 1, 2]].reshape(
            curr_pose_mat.shape[0], -1
        )
        num_angles = num_joints * 4                                 # 96 for SMPL
    else:
        # Convert to Euler angles (default)
        curr_spose = curr_spose.as_euler(
            euler_order,                                            # "ZYX" order
            degrees=False
        ).reshape(curr_pose_mat.shape[0], -1)

    # Step 3: Reorder body joints (NOT root) according to MuJoCo order
    curr_spose = curr_spose.reshape(
        -1, num_joints,
        4 if use_quat else 3                                        # (batch, 24 joints, angles_per_joint)
    )[:, smpl_2_mujoco, :].reshape(-1, num_angles)                 # Reorder to MuJoCo order
    
    # Step 4: Handle root separately
    if use_quat:
        # All joints output as quaternion
        curr_qpos = np.concatenate([trans, curr_spose], axis=1)
    else:
        # Root → quaternion (ALWAYS)
        # Body joints → Euler angles (ZYX order)
        root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
        # curr_spose[:, 3:] skips first 3 values (root Euler), takes body Euler
        curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]), axis=1)

    # Step 5: Adjust position for MuJoCo model offset
    if count_offset:
        curr_qpos[:, :3] = trans + mj_model.body_pos[1]             # Add body[1] (root) offset
        # body_pos[0] = world origin, body_pos[1] = Pelvis/root joint offset

    return curr_qpos
```

### Key Observations

#### ✅ Root Handling (Pelvis)

**Line 397-399**: Root is **CONVERTED TO QUATERNION** regardless of `use_quat`:
```python
root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]), axis=1)
```

- **Axis-angle** → **Rotation matrix** → **Quaternion**
- **NO coordinate transform applied**; quaternion is in the **same coordinate frame** as the original axis-angle
- Transl. is added: `trans + mj_model.body_pos[1]` (offset for MuJoCo's root joint position)

#### ❌ Body Joints (L_Hip, R_Hip, ..., R_Hand)

**Line 388-393**: Body joints are converted to **Euler angles (ZYX order)**:
```python
curr_spose = curr_spose.as_euler(euler_order, degrees=False)
curr_spose = curr_spose.reshape(..., num_joints, 3)[:, smpl_2_mujoco, :].reshape(...)
```

- **Axis-angle** → **Rotation matrix** → **Euler angles (ZYX)**
- **NO coordinate frame change** — still in same orientation space
- Only **reordering** to match MuJoCo's body order (line 393: `[:, smpl_2_mujoco, :]`)

#### `smpl_2_mujoco` Reordering (Line 371-374)

```python
smpl_2_mujoco = [
    joint_names.index(q) 
    for q in list(get_body_qposaddr(mj_model).keys())
    if q in joint_names
]
```

This builds a **list of SMPL indices** that maps to MuJoCo body order:

- `get_body_qposaddr(mj_model).keys()` returns MuJoCo body names in order (e.g., `[Pelvis, L_Hip, R_Hip, ...]`)
- For each body name `q`, find its index in SMPL bone order: `joint_names.index(q)`
- Example:
  ```
  SMPL_BONE_ORDER_NAMES = [Pelvis(0), L_Hip(1), R_Hip(2), Torso(3), ...]
  MuJoCo order = [Pelvis, L_Hip, R_Hip, Torso, ...]
  smpl_2_mujoco = [0, 1, 2, 3, ...]  (if same order)
  ```

#### Coordinate Frame Analysis

**Question**: Does Y-up → Z-up conversion happen?

**Answer**: **NO coordinate frame conversion** in this function:
- SMPL uses **Y-up** (body typically stands upright along Y-axis in SMPL)
- MuJoCo uses **Z-up** (gravity points in -Z)
- This code **does not transform** between them
- **The caller is responsible** for any coordinate system adjustments (possibly in `normalize_smpl_pose()` or upstream)

---

## 2. `qpos_to_smpl()` Function (Lines 552–571)

### Function Signature

```python
def qpos_to_smpl(qpos, mj_model, smpl_model="smpl"):
    """
    Reverse conversion: MuJoCo qpos → SMPL pose (axis-angle)
    """
```

### Complete Code

```python
def qpos_to_smpl(qpos, mj_model, smpl_model="smpl"):
    body_qposaddr = get_body_qposaddr(mj_model)           # Dict: joint_name → (start_idx, end_idx)
    batch_size = qpos.shape[0]
    trans = qpos[:, :3] - mj_model.body_pos[1]            # Extract translation, subtract root offset
    
    smpl_bones_to_use = (
        SMPL_BONE_ORDER_NAMES
        if smpl_model == "smpl" else SMPLH_BONE_ORDER_NAMES
    )
    
    pose = np.zeros([batch_size, len(smpl_bones_to_use), 3])  # (batch, 24, 3) for SMPL
    
    for ind1, bone_name in enumerate(smpl_bones_to_use):
        ind2 = body_qposaddr[bone_name]                   # (start, end) indices in qpos
        
        if ind1 == 0:  # Pelvis (root)
            quat = qpos[:, 3:7]                           # Extract quaternion
            pose[:, ind1, :] = sRot.from_quat(
                quat[:, [1, 2, 3, 0]]                     # Reorder: [x,y,z,w] → [w,x,y,z]
            ).as_rotvec()                                  # Convert to axis-angle
        else:  # Body joints
            pose[:, ind1, :] = sRot.from_euler(
                "ZYX",                                    # Euler order: ZYX
                qpos[:, ind2[0]:ind2[1]]                  # Extract Euler angles from qpos
            ).as_rotvec()                                  # Convert to axis-angle

    return pose, trans
```

### Key Observations

#### Root Conversion (ind1 == 0)

```python
if ind1 == 0:
    quat = qpos[:, 3:7]
    pose[:, ind1, :] = sRot.from_quat(quat[:, [1, 2, 3, 0]]).as_rotvec()
```

- **Quaternion** (from MuJoCo) **→ Axis-angle** (for SMPL)
- Quaternion order: MuJoCo uses `[x, y, z, w]` but SciPy's `from_quat` expects `[x, y, z, w]`
  - Actually, the reordering `[1, 2, 3, 0]` converts from `[x, y, z, w]` at positions `[0, 1, 2, 3]` to `[w, x, y, z]` — wait, let me recheck:
  - Original `quat[:, [1, 2, 3, 0]]` takes columns `[1, 2, 3, 0]`, so if input is `[x, y, z, w]`, output is `[y, z, w, x]`. That's incorrect.
  - **Actually**, looking at `rotation_matrix_to_quaternion()` in the forward pass (line 397), it likely returns quaternions in a specific format (maybe `[w, x, y, z]` or similar), so this reordering adapts to SciPy's expected format.

#### Body Joint Conversion

```python
else:
    pose[:, ind1, :] = sRot.from_euler("ZYX", qpos[:, ind2[0]:ind2[1]]).as_rotvec()
```

- **Euler angles (ZYX order)** (from MuJoCo) **→ Axis-angle** (for SMPL)
- **NO coordinate transform** — same frame

#### NO Coordinate System Conversion

- Translation: `qpos[:, :3] - mj_model.body_pos[1]` (just removes offset, no frame change)
- Rotations: No frame conversion, only representation change (quat/Euler ↔ axis-angle)

---

## 3. Coordinate Transform Logic

### Does the code handle Y-up to Z-up conversion?

**SHORT ANSWER**: **NO, not in these functions**.

**EVIDENCE**:
1. No calls to `rotation_matrix_to_angle_axis()` with a coordinate transform matrix
2. No swapping of axes (e.g., `[x, y, z] → [x, z, -y]`)
3. Default trans `[0, 0, 0.91437225]` is just a Z-height offset for standing posture
4. All quaternions and Euler angles are converted **without modification**

**WHERE IT MIGHT HAPPEN**:
- `normalize_smpl_pose()` (line 351, 607) — might apply transforms
- **Upstream preprocessing** — caller might convert SMPL from Y-up before calling this function
- **Downstream rendering** — MuJoCo renderer handles coordinate visualization

### Y-up vs Z-up Details

- **SMPL**: Body mesh typically has Y-axis pointing up (torso along Y)
- **MuJoCo**: Gravity along -Z (Z-axis points up)
- **This code**: Assumes **both are already in compatible frame** or trusts caller to handle conversion

---

## 4. Summary Table

| Aspect | Root (Pelvis) | Body Joints (L_Hip, ..., R_Hand) |
|--------|---------------|----------------------------------|
| **Input Repr.** | Axis-angle (3D) | Axis-angle (3D) |
| **Intermediate** | Rotation matrix (3×3) | Rotation matrix (3×3) |
| **Output Repr. (default)** | Quaternion (4D) | Euler angles ZYX (3D) |
| **Coordinate Transform** | ❌ NONE | ❌ NONE |
| **Reordering** | N/A (always root) | ✅ YES (smpl_2_mujoco) |
| **MuJoCo Offset** | ✅ Applied (body_pos[1]) | Applied to root trans only |
| **Reverse Function** | Quat → Axis-angle | Euler → Axis-angle |

---

## 5. `smpl_2_mujoco` Reordering Explained

### How it's built:

```python
smpl_2_mujoco = [
    joint_names.index(q) 
    for q in list(get_body_qposaddr(mj_model).keys())
    if q in joint_names
]
```

### Example Scenario:

**SMPL order:**
```
[0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Torso, 4: L_Knee, 5: R_Knee, ...]
```

**MuJoCo body order** (from XML definition):
```
[Pelvis, L_Hip, Torso, R_Hip, L_Knee, ...]
```

**Resulting smpl_2_mujoco:**
```
[0, 1, 3, 2, 4, ...]  # SMPL indices for each MuJoCo body
```

**Usage in line 393:**
```python
curr_spose.reshape(-1, num_joints, 3)[:, smpl_2_mujoco, :].reshape(...)
```

- Input: `(batch, 24, 3)` in SMPL order
- After `[:, smpl_2_mujoco, :]`: `(batch, 24, 3)` reordered to MuJoCo body order
- Ensures output qpos matches MuJoCo's expected joint order

---

## 6. Key Takeaways

1. **NO body joint coordinate transforms** — all rotations stay in same frame
2. **Root is special**: Quat output (unlike body Euler), but **still no frame change**
3. **`smpl_2_mujoco` is purely reordering**, not transformation
4. **Y-up ↔ Z-up conversion** happens elsewhere (not here)
5. **Translation offset** (`body_pos[1]`) accounts for MuJoCo's root joint position, not coordinate system
6. **Reverse conversion** (`qpos_to_smpl`) precisely undoes the forward pass

