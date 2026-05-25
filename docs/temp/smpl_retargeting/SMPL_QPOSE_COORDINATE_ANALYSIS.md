# PHC `smpl_to_qpose()` Coordinate System Analysis

## Summary
- **NO explicit Y-up→Z-up transform** in `smpl_to_qpose()`
- Motion data **arrives in Y-up** (SMPL convention)
- **Euler order used: "ZYX"** (default parameter, line 339)
- **Root body position offset applied**: line 403 adds `mj_model.body_pos[1]` (Pelvis offset)

---

## 1. EXACT CODE: `smpl_to_qpose()` Function (lines 331-405)

```python
def smpl_to_qpose(
    pose,
    mj_model,
    trans=None,
    normalize=False,
    random_root=False,
    count_offset=True,
    use_quat=False,
    euler_order="ZYX",  # ← EULER CONVENTION (line 339)
    model="smpl",
):
    """
    Expect pose to be batch_size x 72
    trans to be batch_size x 3
    differentiable 
    """
    # Line 347-349: Default trans if None - sets Z component to 0.91437225
    if trans is None:
        trans = np.zeros((pose.shape[0], 3))
        trans[:, 2] = 0.91437225
    
    # Line 350-351: Optional normalization (rotates root orientation, not entire body)
    if normalize:
        pose, trans = normalize_smpl_pose(pose, trans, random_root=random_root)
    
    # Lines 353-367: Prepare pose as torch tensor (in SMPL's Y-up convention)
    if not torch.is_tensor(pose):
        pose = torch.tensor(pose)
    
    if model == "smpl":
        joint_names = SMPL_BONE_ORDER_NAMES
        # ... (other models)
    
    # Lines 376-389: Convert angle-axis to Euler angles
    pose = pose.reshape(-1, num_angles)
    
    # Line 378-379: Convert to rotation matrix (still Y-up)
    curr_pose_mat = angle_axis_to_rotation_matrix(pose.reshape(-1, 3)).reshape(
        pose.shape[0], -1, 4, 4)
    
    # Line 381-382: Create scipy Rotation objects
    curr_spose = sRot.from_matrix(curr_pose_mat[:, :, :3, :3].reshape(
        -1, 3, 3).numpy())
    
    # Line 388-389: Convert to Euler angles using "ZYX" order (NO COORD TRANSFORM!)
    if not use_quat:
        curr_spose = curr_spose.as_euler(euler_order, degrees=False).reshape(
            curr_pose_mat.shape[0], -1)
    
    # Lines 391-399: Build output qpos
    curr_spose = curr_spose.reshape(
        -1, num_joints,
        4 if use_quat else 3)[:, smpl_2_mujoco, :].reshape(-1, num_angles)
    
    if use_quat:
        curr_qpos = np.concatenate([trans, curr_spose], axis=1)
    else:
        # Line 397: Root quaternion (from rotation matrix)
        root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
        # Line 398-399: Concatenate [trans, root_quat, joint_euler_angles...]
        curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]),
                                   axis=1)
    
    # Lines 401-403: CRITICAL - Apply Pelvis body offset
    if count_offset:
        curr_qpos[:, :3] = trans + mj_model.body_pos[1]  # ← ADD OFFSET!
    
    return curr_qpos
```

---

## 2. Key Finding: Root Position Offset (Line 403)

```python
if count_offset:
    curr_qpos[:, :3] = trans + mj_model.body_pos[1]
```

**What this does:**
- `trans` = root translation from motion data (Y-up, e.g., shape [N, 3])
- `mj_model.body_pos[1]` = Pelvis body offset from XML
- **Result:** Final qpos position = `motion_trans + pelvis_offset`

**From smpl_humanoid.xml (line 27):**
```xml
<body name="Pelvis" pos="-0.0018 -0.2233 0.0282">
```

So `mj_model.body_pos[1] ≈ [-0.0018, -0.2233, 0.0282]` in **LOCAL coordinates**.

---

## 3. SMPL Humanoid XML Root Body (smpl_humanoid.xml, line 27)

```xml
<mujoco model="humanoid">
  <compiler coordinate="local"/>  <!-- ← Local coordinate frame -->
  <worldbody>
    <light .../>
    <geom .../>  <!-- floor -->
    <body name="Pelvis" pos="-0.0018 -0.2233 0.0282">  <!-- Root body offset -->
      <freejoint name="Pelvis"/>  <!-- 6-DOF free joint -->
      <geom type="sphere" .../>
      <!-- All child bodies follow... -->
```

**Key observations:**
- Pelvis has a `pos` offset: `[-0.0018, -0.2233, 0.0282]` (likely Y-up SMPL offset)
- Y-coordinate (middle): `-0.2233` suggests this is height offset in SMPL's convention
- `<freejoint>` means the root is 6-DOF (position + orientation)

---

## 4. Euler Convention in `smpl_to_qpose()`

**Default: `euler_order="ZYX"` (line 339)**

This means when converting rotation matrix to Euler angles:
```python
curr_spose = curr_spose.as_euler("ZYX", degrees=False)
```

**What ZYX means:** Rotation is applied in order Z → Y → X (intrinsic rotations in scipy)

---

## 5. Where Motion Data Comes From (motion_lib_smpl.py, lines 108-109)

```python
trans = curr_file['root_trans_offset'].clone()[start:end]  # (N, 3) in Y-up
pose_aa = to_torch(curr_file['pose_aa'][start:end])        # (N, 72) angle-axis in Y-up
pose_quat_global = curr_file['pose_quat_global'][start:end]  # (N, 24, 4) quaternions
```

**Motion data format:**
- `pose_aa`: 72-dim angle-axis (SMPL convention, Y-up)
- `trans`: (N, 3) root translation **in Y-up coordinate system**
- `pose_quat_global`: global joint rotations in quaternions

**NO coordinate system conversion before passing to `smpl_to_qpose()`!**

---

## 6. Normalization Function (normalize_smpl_pose, lines 607-635)

When `normalize=True`, the function applies a **rotation to the root orientation** but:

```python
def normalize_smpl_pose(pose_aa, trans=None, random_root=False):
    root_aa = pose_aa[0, :3]  # Root angle-axis
    root_rot = sRot.from_rotvec(np.array(root_aa))
    root_euler = np.array(root_rot.as_euler("xyz", degrees=False))  # ← XYZ order
    target_root_euler = root_euler.copy()
    
    # ... manipulate Z rotation ...
    
    if not trans is None:
        trans[:, [0, 1]] -= trans[0, [0, 1]]  # Center XY
        trans[:, [2]] = trans[:, [2]] - trans[0, [2]] + 0.91437225  # Adjust height
        trans = np.matmul(apply_mat, trans.T).T  # Apply rotation transform
    return pose_aa, trans
```

**Important:** This uses **"xyz" order** for normalization inspection, but doesn't change coordinates.
The height adjustment `0.91437225` appears to be the default Pelvis Z-height in mujoco.

---

## CONCLUSION

| Aspect | Finding |
|--------|---------|
| **Input motion coordinate system** | **Y-up** (SMPL convention, NOT transformed) |
| **Y-up → Z-up transform in smpl_to_qpose()** | **NO - NOT APPLIED** |
| **Euler convention used** | **"ZYX"** (default line 339) |
| **Root rotation format** | Angle-axis → Rotation matrix → Quaternion (for MuJoCo root) |
| **Root position handling** | `trans + mj_model.body_pos[1]` offset applied |
| **Root body Pelvis offset** | `[-0.0018, -0.2233, 0.0282]` from XML |
| **CRITICAL ISSUE** | Motion data stays in Y-up; no coordinate transform! |

---

## ACTION ITEMS FOR USERS

1. **If motion looks rotated/twisted**: Check if a Y-up → Z-up conversion is needed BEFORE calling `smpl_to_qpose()`
2. **If motion floats/sinks**: Adjust the Pelvis offset or the default `trans[:, 2] = 0.91437225` value
3. **If root rotation is wrong**: Verify that input `pose` is truly in angle-axis format
4. **If you need Z-up**: Apply rotation: `pose[..., :3] = rotate_yup_to_zup(pose[..., :3])`
