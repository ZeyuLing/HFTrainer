# PHC SMPL-MuJoCo Conversion - Quick Answers

## Question 1: Exact Euler Convention Used

**Answer: ZYX (Z-Y-X order, also known as intrinsic or extrinsic ZYX)**

| Aspect | Details |
|--------|---------|
| Convention | ZYX (not XYZ, not YXZ) |
| Order | Z-axis (yaw) → Y-axis (pitch) → X-axis (roll) |
| Location in Code | Line 339 (parameter), Line 388 (usage) |
| Hardcoded Instances | Line 567 (qpos_to_smpl), Line 593 (qpos_to_smpl_torch) |
| Flexibility | Parameter exists but inverse is rigid - changing breaks symmetry |
| Degrees | Always in radians (degrees=False) |

**Code Reference:**
```python
# Line 388 - smpl_to_qpose()
curr_spose = curr_spose.as_euler(euler_order, degrees=False)

# Line 567 - qpos_to_smpl() [HARDCODED]
pose[:, ind1, :] = sRot.from_euler("ZYX", qpos[:, ind2[0]:ind2[1]]).as_rotvec()
```

---

## Question 2: How Euler Angles Stored in QPOS Slots

**Answer: Root uses quaternion [x,y,z,w], body joints use ZYX Euler [3 floats each]**

| Slot Range | Content | Format | DOF | Notes |
|------------|---------|--------|-----|-------|
| [0:3] | Translation | [x, y, z] | 3 | World position |
| [3:7] | Root Rotation | Quaternion [x,y,z,w] | 4 | **NOT Euler** |
| [7:10] | Joint 1 | Euler ZYX | 3 | Body joint |
| [10:13] | Joint 2 | Euler ZYX | 3 | Body joint |
| ... | ... | Euler ZYX | 3 | Continue pattern |

**Key Point: Root is quaternion (4 DOF), all body joints are Euler angles (3 DOF each)**

### Quaternion Storage Details

| Operation | Scipy Format | MuJoCo Format | Code |
|-----------|--------------|---------------|------|
| Scipy native | [w, x, y, z] | N/A | `sRot.as_quat()` |
| Stored in qpos | N/A | [x, y, z, w] | `[:, [3, 0, 1, 2]]` |
| Encoding | → | ← | Line 384 |
| Decoding | ← | → | Line 563: `[:, [1, 2, 3, 0]]` |

**Code Reference:**
```python
# Line 384 - Encoding (scipy → MuJoCo)
curr_spose.as_quat()[:, [3, 0, 1, 2]]  # [w,x,y,z] → [x,y,z,w]

# Line 563 - Decoding (MuJoCo → scipy)
sRot.from_quat(quat[:, [1, 2, 3, 0]])  # [x,y,z,w] → [w,x,y,z]
```

---

## Question 3: Coordinate Transform - Body Joints vs Root

**Answer: Root stays global (quaternion), body joints get reordered (Euler angles)**

### ROOT JOINT (Index 0)
```
Input:  Rotation Matrix [1, 3, 3]
  ↓
Convert: rotation_matrix_to_quaternion()
  ↓
Output: Quaternion [w, x, y, z] (scipy format)
  ↓
Reorder: [:, [3, 0, 1, 2]] → [x, y, z, w] (MuJoCo format)
  ↓
Store:   qpos[3:7]
  
Key: NOT reordered by smpl_2_mujoco (inserted before reordering step)
```

**Code Location: Lines 397-399**
```python
root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]), axis=1)
```

### BODY JOINTS (Indices 1+)
```
Input:  Rotation Matrices [24, 3, 3]
  ↓
Convert: sRot.from_matrix() → Rotation objects
  ↓
Extract: as_euler("ZYX") → Euler angles [24, 3]
  ↓
Reshape: [batch, 24, 3] → prepare for reordering
  ↓
Reorder: [:, smpl_2_mujoco, :] → [batch, num_mujoco, 3]
  ↓
Store:   qpos[7:] (after root quaternion)

Key: Reordered via smpl_2_mujoco indices (applied after extraction)
```

**Code Location: Lines 388-393**
```python
curr_spose = curr_spose.as_euler(euler_order, degrees=False).reshape(
    curr_pose_mat.shape[0], -1)

curr_spose = curr_spose.reshape(
    -1, num_joints,
    4 if use_quat else 3)[:, smpl_2_mujoco, :].reshape(-1, num_angles)
```

---

## Question 4: SMPL_2_MUJOCO Reorder Mapping

**Answer: List of SMPL indices ordered by MuJoCo body order**

### Creation

**Location: Lines 371-374**
```python
smpl_2_mujoco = [
    joint_names.index(q) for q in list(get_body_qposaddr(mj_model).keys())
    if q in joint_names
]
```

### Meaning

| Step | Details |
|------|---------|
| Input | `get_body_qposaddr(mj_model).keys()` = MuJoCo bodies in MuJoCo order |
| Process | For each MuJoCo body name, find its index in SMPL_BONE_ORDER_NAMES |
| Output | List of SMPL indices sorted by MuJoCo order |
| Type | List[int] with length = num_mujoco_bodies |

### Example

```
SMPL bones: [0:Pelvis, 1:L_Hip, 2:L_Knee, 3:R_Hip, 4:R_Knee, ...]
MuJoCo order: [Pelvis, R_Hip, R_Knee, L_Hip, L_Knee, ...]
                       ↓
smpl_2_mujoco: [0, 3, 4, 1, 2, ...]
```

### Usage

**Location: Line 393**
```python
# Before: [batch, 24_SMPL_joints, 3_angles]
[:, smpl_2_mujoco, :]

# After: [batch, num_MuJoCo_bodies, 3_angles]
```

### Important Note

**ROOT IS NOT REORDERED**
- `smpl_2_mujoco` only applied to body_joints (indices 1+)
- Root quaternion concatenated BEFORE reordering step
- Intentional design: root in global frame, joints in local frame

---

## Question 5: Body_Pos[1] Offset Handling

**Answer: Pelvis position offset; ADD in forward, SUBTRACT in inverse**

### Definition

| Item | Value |
|------|-------|
| Element | `mj_model.body_pos[1]` |
| Meaning | Position of pelvis/second body in MuJoCo |
| Type | 3D vector [x, y, z] |
| Typically | [0, 0, some_height] |

### Forward Direction (SMPL → MuJoCo)

**Location: Lines 401-403 (smpl_to_qpose)**
```python
if count_offset:
    curr_qpos[:, :3] = trans + mj_model.body_pos[1]
```

| Aspect | Details |
|--------|---------|
| Operation | **ADD** |
| Direction | Relative → Absolute world |
| Effect | Shifts translation by model's pelvis position |
| Parameter | count_offset (default=True) |
| When False | Translation remains relative to origin |

### Inverse Direction (MuJoCo → SMPL)

**Location: Line 555 (qpos_to_smpl)**
```python
trans = qpos[:, :3] - mj_model.body_pos[1]
```

| Aspect | Details |
|--------|---------|
| Operation | **SUBTRACT** |
| Direction | Absolute → Relative |
| Effect | Removes model's position bias |
| Parameter | **NONE** (always subtracts) |
| Asymmetry | qpos_to_smpl always assumes count_offset behavior |

### Symmetry Check

```
Forward:  new_trans = old_trans + offset
Inverse:  old_trans = new_trans - offset
          = (old_trans + offset) - offset
          = old_trans ✓
```

**Symmetric** - operations cancel out

### Default Translation

**Location: Lines 348-349**
```python
if trans is None:
    trans = np.zeros((pose.shape[0], 3))
    trans[:, 2] = 0.91437225  # Standing height
```

| Value | Meaning |
|-------|---------|
| 0.91437225 | Approximate human pelvis height when standing |
| XY | Zero (at world origin horizontally) |
| Z | Standing height (vertical offset) |

### Torch Version

**Location: Lines 545-547 (smpl_to_qpose_torch)**
```python
curr_qpos[:, :3] = trans + torch.from_numpy(
    mj_model.body_pos[1]).to(root_quat)
```
Same logic but converts numpy array to torch tensor on same device.

---

## Summary Table: All 5 Answers

| Question | Answer | Location | Critical Detail |
|----------|--------|----------|-----------------|
| 1. Euler Convention | ZYX | Lines 388, 567 | Hardcoded in inverse - not flexible |
| 2. QPOS Storage | [trans(3), quat(4), euler(3)...] | Lines 0-7+ | Root=quat, body=euler |
| 3. Body vs Root Transform | Root=quat(global), Body=euler(local) | Lines 397-399, 388-393 | Different DOF, different handling |
| 4. SMPL_2_MUJOCO Mapping | SMPL indices in MuJoCo order | Lines 371-374, 393 | Only applied to body joints |
| 5. body_pos[1] Offset | Pelvis position; add forward, subtract inverse | Lines 403, 555 | Default standing height 0.914 |

---

## Critical Code Sections

### All Euler Conversion Points
- Line 339: Parameter default `euler_order="ZYX"`
- Line 388: `as_euler(euler_order, degrees=False)`
- Line 418: Parameter default (smpl_to_qpose_multi)
- Line 466: `as_euler(euler_order, degrees=False)` 
- Line 494: Parameter default (smpl_to_qpose_torch)
- Line 536-537: `matrix_to_euler_angles(..., convention=euler_order)`
- **Line 567: `from_euler("ZYX", ...)` HARDCODED qpos_to_smpl**
- **Line 593: `from_euler("ZYX", ...)` HARDCODED qpos_to_smpl_torch**

### All Quaternion Reordering Points
- Line 384: Encoding `as_quat()[:, [3, 0, 1, 2]]` (scipy → MuJoCo)
- Line 563: Decoding `from_quat(quat[:, [1, 2, 3, 0]])` (MuJoCo → scipy)

### All body_pos[1] Offset Points
- Line 403: Forward ADD (smpl_to_qpose)
- Line 481: Forward ADD (smpl_to_qpose_multi)
- Line 547: Forward ADD (smpl_to_qpose_torch)
- Line 555: Inverse SUBTRACT (qpos_to_smpl)
- Line 577: Inverse SUBTRACT (qpos_to_smpl_torch)

---

## Caveats

⚠️ **Euler order is NOT flexible** - changing euler_order in one function without the other breaks symmetry

⚠️ **Quaternion format mismatch** - scipy [w,x,y,z] vs MuJoCo [x,y,z,w] requires explicit reordering

⚠️ **Root not reordered** - root quaternion inserted before body joint reordering (intentional)

⚠️ **body_pos[1] asymmetric** - smpl_to_qpose has parameter, qpos_to_smpl always subtracts

⚠️ **Default standing height** - 0.91437225 may not match all character models

