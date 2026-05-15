# PHC SMPL-MuJoCo Functions Analysis
## File: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py`

---

## 1. EULER CONVENTION

**Euler Order Parameter: `euler_order="ZYX"` (DEFAULT)**

The euler convention is **hardcoded as "ZYX"** by default across all functions:
- Line 339: `euler_order="ZYX"` (smpl_to_qpose)
- Line 418: `euler_order="ZYX"` (smpl_to_qpose_multi)
- Line 494: `euler_order="ZYX"` (smpl_to_qpose_torch)

This means rotations are applied in the order: **Z (yaw) → Y (pitch) → X (roll)**

**Usage in conversion:**
```python
# Line 388 (smpl_to_qpose)
curr_spose = curr_spose.as_euler(euler_order, degrees=False).reshape(
    curr_pose_mat.shape[0], -1)

# Line 536-537 (smpl_to_qpose_torch - torch version)
curr_spose = tR.matrix_to_euler_angles(curr_pose_mat[:, :, :3, :3],
                                       convention=euler_order)
```

In `qpos_to_smpl()`, the inverse uses **hardcoded "ZYX"** (Line 567):
```python
pose[:, ind1, :] = sRot.from_euler("ZYX",
                                   qpos[:, ind2[0]:ind2[1]]).as_rotvec()
```

---

## 2. EULER ANGLES STORAGE IN QPOS SLOTS

### QPOS Structure:
```
[0:3]   - Translation (x, y, z)
[3:7]   - Root Quaternion (w, x, y, z) → stored as (x, y, z, w) in scipy
[7:10]  - Joint 1 Euler angles (3 DOF)
[10:13] - Joint 2 Euler angles (3 DOF)
... and so on
```

### Root Joint (Index 0):
- **Storage Format**: Quaternion (4 components)
- **Layout**: `[3:7]` in qpos
- **Reordering**: scipy uses [w, x, y, z] but stored as [x, y, z, w]
  - Line 384: `as_quat()[:, [3, 0, 1, 2]]` converts from scipy [w,x,y,z] to [x,y,z,w]
  - Line 563-564: Inverse: `from_quat(quat[:, [1, 2, 3, 0]])` converts [x,y,z,w] back to [w,x,y,z]

### Body Joints (Index 1+):
- **Storage Format**: Euler angles in **ZYX order** (3 components each)
- **Layout**: Starting from index [7:10], [10:13], etc.
- **Conversion**: 
  - Line 388: `as_euler("ZYX", degrees=False)` - converts rotation matrix → ZYX Euler
  - Line 567: `from_euler("ZYX", qpos[:, ind2[0]:ind2[1]])` - converts ZYX Euler → rotation matrix

---

## 3. COORDINATE TRANSFORMS FOR BODY JOINTS VS ROOT

### Key Difference:
**Body joints:** Euler angles (local rotation representation)
**Root joint:** Quaternion (global representation)

### The Root Gets Special Treatment:

**In `smpl_to_qpose()` (lines 397-399):**
```python
root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]),
                           axis=1)
```

- Root is kept as **quaternion** (from 4x4 matrix)
- Body joints use **Euler angles ZYX** (from 3x3 matrices)
- Line 393: Only body joints are reordered by `smpl_2_mujoco`: `[:, smpl_2_mujoco, :]`

**In `qpos_to_smpl()` (lines 561-569):**
```python
if ind1 == 0:  # Root joint
    quat = qpos[:, 3:7]
    pose[:, ind1, :] = sRot.from_quat(quat[:, [1, 2, 3, 0]]).as_rotvec()
else:  # Body joints
    pose[:, ind1, :] = sRot.from_euler("ZYX",
                                       qpos[:, ind2[0]:ind2[1]]).as_rotvec()
```

- Root: Extracted from quaternion slots [3:7], converted to rotvec (axis-angle)
- Body joints: Extracted from Euler slots, converted to rotvec

---

## 4. SMPL_2_MUJOCO REORDER MAPPING

### Definition (Line 371-374):
```python
smpl_2_mujoco = [
    joint_names.index(q) for q in list(get_body_qposaddr(mj_model).keys())
    if q in joint_names
]
```

### What it does:
- Maps SMPL bone order to MuJoCo body order
- `joint_names` = SMPL_BONE_ORDER_NAMES (24 joints total)
- `get_body_qposaddr(mj_model).keys()` = MuJoCo body names in MuJoCo order
- Result: List of SMPL indices for each MuJoCo body

### Application (Line 391-393):
```python
curr_spose = curr_spose.reshape(
    -1, num_joints,
    4 if use_quat else 3)[:, smpl_2_mujoco, :].reshape(-1, num_angles)
```

- Reshape SMPL poses to [batch, 24_joints, 3_or_4_angles]
- Reorder using `smpl_2_mujoco` indices
- Result: [batch, num_mujoco_joints, 3_angles]

### Root is NOT reordered:
The root quaternion (index 0) is concatenated AFTER reordering:
```python
curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]),
                           axis=1)
```
- `trans` (translation) - position 0-2
- `root_quat` (quaternion) - position 3-6 (NEW, not reordered)
- `curr_spose[:, 3:]` (body joint Eulers, ALREADY REORDERED)

---

## 5. BODY_POS[1] OFFSET HANDLING

### Definition:
`mj_model.body_pos[1]` = Position of the second body in MuJoCo (typically the root/pelvis)

### In `smpl_to_qpose()` (Line 401-403):
```python
if count_offset:
    curr_qpos[:, :3] = trans + mj_model.body_pos[1]
```

- When `count_offset=True` (default):
  - Translation is **ADDED** to `body_pos[1]`
  - Result: Absolute world position
- When `count_offset=False`:
  - Translation remains as-is (relative to origin)

### In `qpos_to_smpl()` (Line 555):
```python
trans = qpos[:, :3] - mj_model.body_pos[1]
```

- **INVERSE operation**: Subtract `body_pos[1]`
- Converts from absolute world position back to relative translation
- Makes it symmetric with `smpl_to_qpose()`

### In `smpl_to_qpose_torch()` (Line 545-547):
```python
if count_offset:
    curr_qpos[:, :3] = trans + torch.from_numpy(
        mj_model.body_pos[1]).to(root_quat)
```

- Same logic but with PyTorch tensors
- Converts numpy offset to torch tensor on same device

### Default Offset Value (Line 349):
```python
if trans is None:
    trans = np.zeros((pose.shape[0], 3))
    trans[:, 2] = 0.91437225  # Z-height offset (standing height)
```

- Default height: 0.91437225 units (approximately human standing height)

---

## 6. COMPLETE FUNCTION SIGNATURES

### `smpl_to_qpose()` - Lines 331-405
```python
def smpl_to_qpose(
    pose,                  # batch_size x 72 (SMPL pose in axis-angle)
    mj_model,              # MuJoCo model
    trans=None,            # batch_size x 3 (translation, default standing height)
    normalize=False,       # normalize SMPL pose
    random_root=False,     # random root rotation during normalization
    count_offset=True,     # add body_pos[1] offset to translation
    use_quat=False,        # unused (kept for compatibility)
    euler_order="ZYX",     # Euler convention for body joints
    model="smpl",          # "smpl", "smplh", or "smplx"
):
    # Returns: curr_qpos (batch_size x n_qpos)
```

### `qpos_to_smpl()` - Lines 552-571
```python
def qpos_to_smpl(
    qpos,                  # batch_size x n_qpos (MuJoCo joint positions)
    mj_model,              # MuJoCo model
    smpl_model="smpl",     # "smpl" or "smplh"
):
    # Returns: (pose, trans)
    #   pose: batch_size x 24 x 3 (SMPL pose in axis-angle)
    #   trans: batch_size x 3 (translation relative to body_pos[1])
```

---

## 7. HELPER FUNCTIONS CALLED

### From `uhc.utils.torch_geometry_transforms`:
- **`angle_axis_to_rotation_matrix()`** (Line 378)
  - Converts axis-angle vectors → 4x4 rotation matrices
  - Input: [batch, num_joints, 3]
  - Output: [batch, num_joints, 4, 4]

- **`rotation_matrix_to_quaternion()`** (Lines 397, 542)
  - Converts 3x3 rotation matrix → quaternion
  - Used for root joint conversion

### From `scipy.spatial.transform`:
- **`sRot.from_matrix()`** (Line 381)
  - Creates Rotation object from 3x3 matrices

- **`sRot.as_euler()`** (Line 388)
  - Converts Rotation → Euler angles in specified convention
  - Input: convention string (e.g., "ZYX")
  - Output: [batch, num_joints*3] flattened Euler angles

- **`sRot.as_quat()`** (Line 384)
  - Converts Rotation → quaternion [w, x, y, z]
  - Reordered to [x, y, z, w] for storage

- **`sRot.from_euler()`** (Line 567)
  - Converts Euler angles → Rotation object

- **`sRot.from_quat()`** (Line 563)
  - Converts quaternion → Rotation object

- **`sRot.as_rotvec()`** (Lines 564, 569)
  - Converts Rotation → axis-angle rotvec

### From `uhc.utils.rotation_conversions` (imported as `tR`):
- **`tR.matrix_to_euler_angles()`** (Line 536)
  - PyTorch version of Euler conversion
  - Used in `smpl_to_qpose_torch()`
  - Input: [batch, num_joints, 3, 3] matrices
  - Convention: "ZYX" by default

### From `uhc.khrylib.utils`:
- **`get_body_qposaddr()`** (Lines 372, 553)
  - Returns dict: {body_name: (start_idx, end_idx)}
  - Maps body names to qpos slice indices in MuJoCo

---

## 8. KEY FLOW DIAGRAM

### `smpl_to_qpose()` Flow:
```
SMPL Pose (axis-angle, 24x3)
    ↓
angle_axis_to_rotation_matrix() 
    ↓
Rotation Matrices (24x3x3)
    ↓
sRot.from_matrix() → Rotation objects
    ↓
├─ Root (index 0): as_quat() → quaternion (keep as-is)
│
└─ Bodies (index 1+): as_euler("ZYX") → Euler angles
    ↓
    Reorder via smpl_2_mujoco indices
    ↓
Concatenate: [translation, root_quat, body_eulers]
    ↓
Add body_pos[1] offset to translation (if count_offset=True)
    ↓
MuJoCo QPos
```

### `qpos_to_smpl()` Flow:
```
MuJoCo QPos
    ↓
Extract translation: qpos[:, :3] - body_pos[1]
    ↓
├─ Root (index 0): Extract quat [3:7] → from_quat() → as_rotvec() → axis-angle
│
└─ Bodies: Extract Euler angles → from_euler("ZYX") → as_rotvec() → axis-angle
    ↓
SMPL Pose (axis-angle, 24x3)
```

---

## 9. IMPORTANT NOTES

1. **ZYX is NOT flexible**: While `euler_order` is a parameter, the inverse operation in `qpos_to_smpl()` hardcodes "ZYX" (Line 567). Changing the default would break conversion symmetry unless `qpos_to_smpl()` is updated too.

2. **Root quaternion ordering**: scipy uses [w,x,y,z] but MuJoCo expects [x,y,z,w]. The code handles this with explicit reordering:
   - Encoding: `as_quat()[:, [3, 0, 1, 2]]` (w→index 3)
   - Decoding: `from_quat(quat[:, [1, 2, 3, 0]])` (w back to index 0)

3. **Body joints are reordered, root is not**: The `smpl_2_mujoco` mapping only applies to body joints (indices 1+). The root quaternion is inserted directly after translation.

4. **body_pos[1] is additive**: It represents the offset of the pelvis body in the MuJoCo model. When `count_offset=True`, all translations are shifted by this offset to account for the model's initial placement.

5. **Default standing height**: 0.91437225 is approximately the height of a standing human (from ground to pelvis).

