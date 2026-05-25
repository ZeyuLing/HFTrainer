# SMPL ↔ MuJoCo Conversion — Quick Reference

## TL;DR

| Question | Answer |
|----------|--------|
| **Does root get coord transform?** | ❌ NO — just representation change (axis-angle → quat) + offset |
| **Do body joints get coord transform?** | ❌ NO — just representation change (axis-angle → Euler) + reordering |
| **Does Y-up → Z-up conversion happen?** | ❌ NO — handled elsewhere or assumed already done |
| **What is `smpl_2_mujoco`?** | List of SMPL indices to reorder joints to MuJoCo body order |
| **Is reordering applied to root?** | ❌ NO — root always at index 0 |

---

## Forward Conversion: `smpl_to_qpose()` (Lines 331–405)

```
INPUT:  SMPL pose (batch, 24×3 axis-angle) + trans (batch, 3)
        └─ SMPL order: [Pelvis, L_Hip, R_Hip, Torso, L_Knee, ...]

STEP 1: Axis-angle → Rotation matrix (24, 3)×(3,3)
        └─ rotation_matrix_to_angle_axis() from torch_geometry_transforms

STEP 2: Rotation matrix → Euler/Quat 
        ├─ Root (joint 0):     Rotation → QUATERNION (4D) [NOT reordered]
        └─ Body (joint 1-23):  Rotation → EULER ZYX (3D)

STEP 3: Reorder body joints to MuJoCo order
        └─ smpl_2_mujoco = [SMPL indices in MuJoCo order]
        └─ Applied to body only, root stays first

STEP 4: Assemble qpos
        └─ qpos = [trans (3), root_quat (4), body_euler_reordered (63)]
        └─ Position adjustment: trans += mj_model.body_pos[1]

OUTPUT: qpos (batch, 70) = [trans, root_quat, body_euler_reordered]
        └─ MuJoCo order: [Pelvis_trans(3), Pelvis_quat(4), L_Hip_euler(3), R_Hip_euler(3), ...]
```

### Root Special Treatment

```python
# Lines 397-399: Root ALWAYS becomes quaternion
root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]), axis=1)
```

- Root output is **quaternion** (even if `use_quat=False`)
- Body output is **Euler angles** (when `use_quat=False`)

---

## Reverse Conversion: `qpos_to_smpl()` (Lines 552–571)

```
INPUT:  MuJoCo qpos (batch, 70)
        └─ Structure: [trans(3), root_quat(4), body_euler(63)]
        └─ Joint order: [Pelvis, ..., R_Hand] in MuJoCo order

STEP 1: Extract translation
        └─ trans = qpos[:, :3] - mj_model.body_pos[1]

STEP 2: For each joint in SMPL order:
        ├─ Joint 0 (Pelvis):    Quat → Axis-angle [rotvec]
        └─ Joint 1-23:          Euler ZYX → Axis-angle [rotvec]

OUTPUT: pose (batch, 24×3 axis-angle in SMPL order) + trans (batch, 3)
```

**Note**: No reordering in reverse! The function iterates SMPL order and directly indexes qpos from `body_qposaddr` dict.

---

## The `smpl_2_mujoco` Mapping

### What it is:
```python
smpl_2_mujoco = [
    joint_names.index(q)                            # SMPL index
    for q in list(get_body_qposaddr(mj_model).keys())  # MuJoCo order
    if q in joint_names
]
```

### Example:
```
SMPL_BONE_ORDER_NAMES:
  0: Pelvis,  1: L_Hip,  2: R_Hip,  3: Torso,  4: L_Knee,  ...

MuJoCo body order (from mj_model):
  Pelvis,  L_Hip,  Torso,  R_Hip,  L_Knee,  ...

smpl_2_mujoco =
  [  0,      1,      3,      2,      4,    ...]
```

### How it's used:
```python
curr_spose.reshape(-1, num_joints, 3)[:, smpl_2_mujoco, :].reshape(...)
```

- Input shape: `(batch, 24, 3)` in **SMPL order**
- After indexing `[:, smpl_2_mujoco, :]`: reordered to **MuJoCo order**
- Only applied to **body joints** (joint 1-23), **NOT root**

---

## Coordinate System Status

| Aspect | Status | Details |
|--------|--------|---------|
| **Y-up → Z-up conversion in smpl_to_qpose()** | ❌ NO | All rotations stay in input frame |
| **Axis swap (e.g., [x,y,z]→[x,z,-y])** | ❌ NO | No evidence in code |
| **Where it might happen** | Elsewhere | `normalize_smpl_pose()`, caller preprocessing, or MuJoCo renderer |
| **Default Z-height** | 0.91437225 | Standing height for Z-up world (not coord conversion) |

---

## qpos Structure in MuJoCo

```
qpos = [
  0:3      Translation (x, y, z)
  3:7      Root quaternion (w, x, y, z) [reorder needed for scipy]
  7:10     L_Hip Euler (ZYX order)
  10:13    R_Hip Euler (ZYX order)
  ...
  67:70    R_Wrist Euler (ZYX order)
]
```

**Index lookup**: Use `body_qposaddr[joint_name]` to find (start, end) indices for each joint.

---

## Key Code Snippets

### Building smpl_2_mujoco (Lines 371-374)
```python
smpl_2_mujoco = [
    joint_names.index(q) for q in list(get_body_qposaddr(mj_model).keys())
    if q in joint_names
]
```

### Root to quaternion (Lines 397-399)
```python
root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]), axis=1)
```

### Reordering body joints (Lines 391-393)
```python
curr_spose = curr_spose.reshape(
    -1, num_joints, 4 if use_quat else 3
)[:, smpl_2_mujoco, :].reshape(-1, num_angles)
```

### Reverse: Root quaternion to axis-angle (Lines 562-564)
```python
quat = qpos[:, 3:7]
pose[:, ind1, :] = sRot.from_quat(quat[:, [1, 2, 3, 0]]).as_rotvec()
```

### Reverse: Body Euler to axis-angle (Lines 567-569)
```python
pose[:, ind1, :] = sRot.from_euler("ZYX", qpos[:, ind2[0]:ind2[1]]).as_rotvec()
```

---

## Conclusions

1. **No coordinate frame transformation** between SMPL and MuJoCo in this file
2. **Only representation changes**: axis-angle ↔ quaternion/Euler
3. **Joint reordering**: Via `smpl_2_mujoco`, applied to body only
4. **Root special**: Always quaternion output, never reordered
5. **Reversible**: `qpos_to_smpl()` correctly inverts `smpl_to_qpose()`
6. **Y-up ↔ Z-up**: Handled elsewhere, not here

