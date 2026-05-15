# PHC SMPL-MuJoCo Function Code Reference

## Source File
`/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py`

---

## FUNCTION 1: `smpl_to_qpose()` - Lines 331-405

### Complete Code:
```python
def smpl_to_qpose(
    pose,
    mj_model,
    trans=None,
    normalize=False,
    random_root=False,
    count_offset=True,
    use_quat=False,
    euler_order="ZYX",
    model="smpl",
):
    """
    Expect pose to be batch_size x 72
    trans to be batch_size x 3
    differentiable 
    """
    if trans is None:
        trans = np.zeros((pose.shape[0], 3))
        trans[:, 2] = 0.91437225
    if normalize:
        pose, trans = normalize_smpl_pose(pose, trans, random_root=random_root)

    if not torch.is_tensor(pose):
        pose = torch.tensor(pose)

    if model == "smpl":
        joint_names = SMPL_BONE_ORDER_NAMES
        if pose.shape[-1] == 156:
            pose = smplh_to_smpl(pose)
    elif model == "smplh":
        joint_names = SMPLH_BONE_ORDER_NAMES
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)
    elif model == "smplx":
        joint_names = SMPLH_BONE_ORDER_NAMES
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)

    num_joints = len(joint_names)
    num_angles = num_joints * 3
    smpl_2_mujoco = [
        joint_names.index(q) for q in list(get_body_qposaddr(mj_model).keys())
        if q in joint_names
    ]

    pose = pose.reshape(-1, num_angles)

    curr_pose_mat = angle_axis_to_rotation_matrix(pose.reshape(-1, 3)).reshape(
        pose.shape[0], -1, 4, 4)

    curr_spose = sRot.from_matrix(curr_pose_mat[:, :, :3, :3].reshape(
        -1, 3, 3).numpy())
    if use_quat:
        curr_spose = curr_spose.as_quat()[:, [3, 0, 1, 2]].reshape(
            curr_pose_mat.shape[0], -1)
        num_angles = num_joints * (4 if use_quat else 3)
    else:
        curr_spose = curr_spose.as_euler(euler_order, degrees=False).reshape(
            curr_pose_mat.shape[0], -1)

    curr_spose = curr_spose.reshape(
        -1, num_joints,
        4 if use_quat else 3)[:, smpl_2_mujoco, :].reshape(-1, num_angles)
    if use_quat:
        curr_qpos = np.concatenate([trans, curr_spose], axis=1)
    else:
        root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
        curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]),
                                   axis=1)

    if count_offset:

        curr_qpos[:, :3] = trans + mj_model.body_pos[1]

    return curr_qpos
```

### Key Steps:
1. **Line 347-349**: Initialize translation with default standing height (0.91437225)
2. **Line 353-354**: Convert pose to tensor if needed
3. **Line 356-367**: Load appropriate joint names (SMPL/SMPLH/SMPLX)
4. **Line 369-374**: Create `smpl_2_mujoco` reorder mapping
5. **Line 376**: Reshape pose to [batch, num_angles]
6. **Line 378-379**: Convert axis-angle to 4x4 rotation matrices
7. **Line 381-382**: Create scipy Rotation objects from matrices
8. **Line 388**: **KEY: Convert to Euler angles using ZYX convention**
9. **Line 391-393**: Reshape to [batch, num_joints, 3] and reorder using `smpl_2_mujoco`
10. **Line 397-399**: Extract root quaternion separately (NOT reordered)
11. **Line 401-403**: Add body_pos[1] offset to translation
12. **Line 405**: Return final qpos

### Critical Lines Explained:

**Line 371-374: Creating smpl_2_mujoco mapping**
```python
smpl_2_mujoco = [
    joint_names.index(q) for q in list(get_body_qposaddr(mj_model).keys())
    if q in joint_names
]
```
- Iterates through MuJoCo body names in MuJoCo order
- Finds the corresponding index in SMPL_BONE_ORDER_NAMES
- Result: list of indices to reorder SMPL joints to MuJoCo order

**Line 381-382: Converting to Rotation objects**
```python
curr_spose = sRot.from_matrix(curr_pose_mat[:, :, :3, :3].reshape(
    -1, 3, 3).numpy())
```
- Extracts 3x3 rotation parts from 4x4 matrices
- Reshapes to [batch*num_joints, 3, 3] for batch processing
- Creates scipy Rotation objects

**Line 388: The Euler Conversion (Root of ZYX Convention)**
```python
curr_spose = curr_spose.as_euler(euler_order, degrees=False).reshape(
    curr_pose_mat.shape[0], -1)
```
- Converts Rotation objects to Euler angles
- `euler_order="ZYX"` specifies the rotation order
- Reshapes back to [batch, num_joints*3]

**Line 391-393: Reordering and reshaping**
```python
curr_spose = curr_spose.reshape(
    -1, num_joints,
    4 if use_quat else 3)[:, smpl_2_mujoco, :].reshape(-1, num_angles)
```
1. Reshape to [batch, 24_joints, 3_angles]
2. Index with `smpl_2_mujoco` to reorder
3. Reshape back to [batch, num_mujoco_joints*3]

**Line 397-399: Root quaternion extraction (NOT reordered)**
```python
root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]),
                           axis=1)
```
- Extract root (index 0) quaternion separately
- Concatenate: translation + root_quat + body_eulers[1:]
- Root quaternion is NOT included in `curr_spose` reordering

**Line 401-403: The body_pos[1] offset**
```python
if count_offset:
    curr_qpos[:, :3] = trans + mj_model.body_pos[1]
```
- Adds MuJoCo's pelvis body position to the translation
- Converts from relative to absolute world coordinates

---

## FUNCTION 2: `qpos_to_smpl()` - Lines 552-571

### Complete Code:
```python
def qpos_to_smpl(qpos, mj_model, smpl_model="smpl"):
    body_qposaddr = get_body_qposaddr(mj_model)
    batch_size = qpos.shape[0]
    trans = qpos[:, :3] - mj_model.body_pos[1]
    smpl_bones_to_use = (SMPL_BONE_ORDER_NAMES
                         if smpl_model == "smpl" else SMPLH_BONE_ORDER_NAMES)
    pose = np.zeros([batch_size, len(smpl_bones_to_use), 3])
    for ind1, bone_name in enumerate(smpl_bones_to_use):
        ind2 = body_qposaddr[bone_name]
        if ind1 == 0:
            quat = qpos[:, 3:7]
            pose[:, ind1, :] = sRot.from_quat(quat[:,
                                                   [1, 2, 3, 0]]).as_rotvec()
        else:
            pose[:,
                 ind1, :] = sRot.from_euler("ZYX",
                                            qpos[:,
                                                 ind2[0]:ind2[1]]).as_rotvec()

    return pose, trans
```

### Key Steps:
1. **Line 553**: Get body position address mapping from MuJoCo model
2. **Line 554**: Extract batch size from qpos
3. **Line 555**: Extract translation and subtract body_pos[1] offset (INVERSE of smpl_to_qpose)
4. **Line 556-557**: Select appropriate bone order (SMPL or SMPLH)
5. **Line 558**: Initialize output pose array [batch, 24_joints, 3_angles]
6. **Line 559-569**: Iterate through each SMPL bone
7. **Lines 561-564**: If root joint (ind1==0):
   - Extract quaternion from slots [3:7]
   - Reorder from [x,y,z,w] to [w,x,y,z] for scipy
   - Convert to axis-angle (rotvec)
8. **Lines 566-569**: If body joint (ind1>0):
   - Extract Euler angles from qpos using body_qposaddr indices
   - Convert from ZYX Euler to axis-angle (rotvec)
9. **Line 571**: Return axis-angle pose and translation

### Critical Lines Explained:

**Line 555: The body_pos[1] offset (INVERSE)**
```python
trans = qpos[:, :3] - mj_model.body_pos[1]
```
- Subtracts body_pos[1] to convert from absolute to relative coordinates
- **Symmetric inverse of** `curr_qpos[:, :3] = trans + mj_model.body_pos[1]`

**Line 563-564: Root quaternion with reordering**
```python
pose[:, ind1, :] = sRot.from_quat(quat[:, [1, 2, 3, 0]]).as_rotvec()
```
- Reorder from MuJoCo [x,y,z,w] to scipy [w,x,y,z]
- **Note**: This uses [1,2,3,0] indexing which means:
  - index 0 (x) → index 1 (scipy index 0 position gets x)
  - index 1 (y) → index 2 (scipy index 1 position gets y)
  - index 2 (z) → index 3 (scipy index 2 position gets z)
  - index 3 (w) → index 0 (scipy index 3 position gets w)
  - Result: [w, x, y, z] as expected

**Line 567-569: Body joint conversion**
```python
pose[:, ind1, :] = sRot.from_euler("ZYX",
                                   qpos[:, ind2[0]:ind2[1]]).as_rotvec()
```
- **HARDCODED "ZYX"** (not flexible like smpl_to_qpose)
- This is why smpl_to_qpose's euler_order parameter won't work symmetrically
- ind2[0]:ind2[1] extracts the 3-element Euler angle slice from qpos

---

## ADDITIONAL VARIANTS

### `smpl_to_qpose_torch()` - Lines 486-549
Same as `smpl_to_qpose()` but:
- Takes torch tensors directly
- Uses `tR.matrix_to_euler_angles()` instead of scipy's as_euler()
- Returns torch tensors instead of numpy

**Line 536-537: The PyTorch Euler conversion**
```python
curr_spose = tR.matrix_to_euler_angles(curr_pose_mat[:, :, :3, :3],
                                       convention=euler_order)
```

### `smpl_to_qpose_multi()` - Lines 408-483
Same as `smpl_to_qpose()` but:
- Takes `mujoco_body_order` parameter instead of deriving from mj_model
- Takes `offset` parameter instead of using `mj_model.body_pos[1]`
- Used for multi-person scenarios

---

## HELPER FUNCTIONS USED

### From imports (Lines 17-32):
```python
from uhc.khrylib.utils import get_body_qposaddr, get_body_qveladdr
from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES
from uhc.utils.torch_geometry_transforms import (
    angle_axis_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from scipy.spatial.transform import Rotation as sRot
import uhc.utils.rotation_conversions as tR
```

### Called Functions:
1. **`angle_axis_to_rotation_matrix()`** - Converts [batch*num_joints, 3] axis-angle to [batch*num_joints, 4, 4] rotation matrices
2. **`rotation_matrix_to_quaternion()`** - Converts [batch, 3, 3] rotation matrices to [batch, 4] quaternions
3. **`sRot.from_matrix()`** - scipy Rotation from 3x3 matrices
4. **`sRot.as_euler()`** - scipy Rotation to Euler angles
5. **`sRot.as_quat()`** - scipy Rotation to quaternion
6. **`sRot.from_quat()`** - scipy Rotation from quaternion
7. **`sRot.from_euler()`** - scipy Rotation from Euler angles
8. **`sRot.as_rotvec()`** - scipy Rotation to axis-angle
9. **`tR.matrix_to_euler_angles()`** - torch Rotation matrix to Euler angles
10. **`get_body_qposaddr()`** - MuJoCo: {body_name: (start_idx, end_idx)}
11. **`normalize_smpl_pose()`** - Normalizes SMPL pose (normalize=True case)

---

## QUICK REFERENCE: Quaternion Reordering

### scipy format: [w, x, y, z]
### MuJoCo format: [x, y, z, w]

**Encoding (smpl_to_qpose, Line 384):**
```python
as_quat()[:, [3, 0, 1, 2]]
# Takes scipy [w, x, y, z] and rearranges to [x, y, z, w]
# Index mapping: 0→1, 1→2, 2→3, 3→0
```

**Decoding (qpos_to_smpl, Line 563):**
```python
from_quat(quat[:, [1, 2, 3, 0]])
# Takes MuJoCo [x, y, z, w] and rearranges to [w, x, y, z] for scipy
# Index mapping: 0→3, 1→0, 2→1, 3→2
```

---

## QUICK REFERENCE: Body Joint Reordering

**SMPL order:** 24 bones in SMPL_BONE_ORDER_NAMES sequence
**MuJoCo order:** N bodies in MuJoCo XML order

**Mapping creation (Line 371-374):**
```python
smpl_2_mujoco = [SMPL_index for MuJoCo_body_name in MuJoCo_bodies]
```

**Application (Line 391-393):**
```python
# Before: [batch, 24_SMPL_joints, 3_angles]
[:, smpl_2_mujoco, :]  # Index to reorder
# After: [batch, num_MuJoCo_bodies, 3_angles]
```

---

## QUICK REFERENCE: Body Position Offset

**MuJoCo model.body_pos[0]:** World/root body (usually at origin)
**MuJoCo model.body_pos[1]:** Pelvis/second body (the actual character root)

**Forward (smpl_to_qpose):**
```
relative_trans → abs_trans = relative_trans + body_pos[1]
```

**Inverse (qpos_to_smpl):**
```
abs_trans → relative_trans = abs_trans - body_pos[1]
```

**Default if trans is None:**
```python
trans = [0, 0, 0.91437225]  # Standing height
```

