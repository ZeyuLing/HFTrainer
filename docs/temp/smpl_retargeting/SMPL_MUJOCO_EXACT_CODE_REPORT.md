# SMPL ↔ MuJoCo Exact Code Inspection Report

## File Location
`/ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py`

---

## 1. `smpl_to_qpose()` — Exact Code (Lines 331–405)

### Function Definition with Annotations

```python
def smpl_to_qpose(
    pose,                    # Input: (batch_size, 72) for SMPL
    mj_model,               # MuJoCo model instance
    trans=None,             # (batch_size, 3) or None
    normalize=False,        # apply normalize_smpl_pose()
    random_root=False,      # used if normalize=True
    count_offset=True,      # apply body_pos[1] offset
    use_quat=False,         # if True, all joints as quat; else root=quat, body=euler
    euler_order="ZYX",      # Euler convention (e.g., "ZYX", "XYZ")
    model="smpl",           # "smpl", "smplh", or "smplx"
):
    """
    Expect pose to be batch_size x 72
    trans to be batch_size x 3
    differentiable 
    """
    
    # ========== STEP 0: Initialize translation if needed ==========
    if trans is None:
        trans = np.zeros((pose.shape[0], 3))
        trans[:, 2] = 0.91437225  # ← DEFAULT: Standing height for Z-up
        # NOTE: This is NOT a coordinate transform; just a default position
    
    if normalize:
        pose, trans = normalize_smpl_pose(pose, trans, random_root=random_root)
        # ← Possible coordinate adjustment here (check normalize_smpl_pose())

    if not torch.is_tensor(pose):
        pose = torch.tensor(pose)

    # ========== STEP 1: Select joint names and validate dimensions ==========
    if model == "smpl":
        joint_names = SMPL_BONE_ORDER_NAMES  # [Pelvis, L_Hip, R_Hip, ..., R_Hand]
        if pose.shape[-1] == 156:
            pose = smplh_to_smpl(pose)        # Convert SMPLH (156) → SMPL (72)
    elif model == "smplh":
        joint_names = SMPLH_BONE_ORDER_NAMES
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)        # Convert SMPL (72) → SMPLH (156)
    elif model == "smplx":
        joint_names = SMPLH_BONE_ORDER_NAMES  # Treat as SMPLH
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)

    num_joints = len(joint_names)              # = 24 for SMPL
    num_angles = num_joints * 3                # = 72 for SMPL (axis-angle)

    # ========== STEP 2: BUILD smpl_2_mujoco REORDERING MAPPING ==========
    smpl_2_mujoco = [
        joint_names.index(q)                   # SMPL index of joint q
        for q in list(get_body_qposaddr(mj_model).keys())  # MuJoCo body order
        if q in joint_names                    # Filter to joints in SMPL
    ]
    # 
    # Example output:
    #   SMPL:    [Pelvis(0), L_Hip(1), R_Hip(2), Torso(3), L_Knee(4), ...]
    #   MuJoCo:  [Pelvis,    L_Hip,    Torso,    R_Hip,    L_Knee,    ...]
    #   Result:  [0,         1,        3,        2,        4,         ...]
    #            ↑ SMPL indices in MuJoCo order

    # ========== STEP 3: Reshape pose for batch processing ==========
    pose = pose.reshape(-1, num_angles)        # (batch, 72)

    # ========== STEP 4: AXIS-ANGLE → ROTATION MATRIX ==========
    curr_pose_mat = angle_axis_to_rotation_matrix(
        pose.reshape(-1, 3)                    # Reshape to (batch*24, 3)
    ).reshape(pose.shape[0], -1, 4, 4)         # → (batch, 24, 4, 4)
    # 
    # NOTE: angle_axis_to_rotation_matrix() is from torch_geometry_transforms
    # INPUT: Axis-angle vectors (batch*24, 3)
    # OUTPUT: 4×4 homogeneous matrices (batch*24, 4, 4)

    # ========== STEP 5: EXTRACT 3×3 ROTATION, CONVERT TO EULER/QUAT ==========
    curr_spose = sRot.from_matrix(
        curr_pose_mat[:, :, :3, :3].reshape(-1, 3, 3).numpy()
    )                                          # SciPy Rotation objects
    
    if use_quat:
        # Output quaternion for ALL joints (including root)
        curr_spose = curr_spose.as_quat()[:, [3, 0, 1, 2]].reshape(
            curr_pose_mat.shape[0], -1
        )
        # NOTE: SciPy as_quat() returns (w, x, y, z), but indexed with [3, 0, 1, 2]
        #       This suggests the function expects a different convention
        num_angles = num_joints * 4            # = 96 for SMPL
    else:
        # ROTATION MATRIX → EULER ANGLES
        # ⚠️ KEY: ALL joints (including root) converted to Euler here
        curr_spose = curr_spose.as_euler(
            euler_order,                       # e.g., "ZYX"
            degrees=False
        ).reshape(curr_pose_mat.shape[0], -1)  # (batch*24, 3) or (batch*24, 3)

    # ========== STEP 6: REORDER TO MUJOCO BODY ORDER ==========
    curr_spose = curr_spose.reshape(
        -1, num_joints,
        4 if use_quat else 3                   # (batch, 24, 4) or (batch, 24, 3)
    )[:, smpl_2_mujoco, :].reshape(-1, num_angles)  # Reorder by smpl_2_mujoco
    # 
    # This reorders from SMPL order to MuJoCo order
    # ⚠️ NOTE: This applies to the ENTIRE array (including root index 0)
    #          But smpl_2_mujoco[0] is typically 0 (Pelvis at both ends)

    # ========== STEP 7: HANDLE ROOT SEPARATELY (MOST IMPORTANT) ==========
    if use_quat:
        # All joints output as quaternion (no special root handling)
        curr_qpos = np.concatenate([trans, curr_spose], axis=1)
    else:
        # ❌ NO COORDINATE TRANSFORM HERE ❌
        # Root: Use quaternion (NOT Euler)
        root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
        # 
        # Body: Use Euler angles (already computed above)
        # curr_spose[:, 3:] = Body Euler angles (skip first 3 values which are root Euler)
        curr_qpos = np.concatenate(
            (trans, root_quat, curr_spose[:, 3:]),
            axis=1
        )
        # 
        # Output structure:
        # [trans(3), root_quat(4), body_euler(63)]
        # = (batch, 70) total

    # ========== STEP 8: APPLY MUJOCO BODY OFFSET ==========
    if count_offset:
        curr_qpos[:, :3] = trans + mj_model.body_pos[1]
        # 
        # body_pos[0] = world origin
        # body_pos[1] = root joint (Pelvis) offset in MuJoCo
        # This adjusts the position to account for MuJoCo's internal offset

    return curr_qpos
```

### Coordinate Transform Analysis

```python
# ❌ NO coordinate transform applied:
# - angle_axis_to_rotation_matrix() just converts representation
# - sRot.from_matrix() → as_quat() or as_euler() just converts representation
# - rotation_matrix_to_quaternion() converts representation only
# - No axis swapping (e.g., x↔z) or sign flipping
# - No multiplication by a coordinate frame matrix
```

**Conclusion**: The function **applies NO coordinate system transformation** (Y-up ↔ Z-up). All rotations stay in their original coordinate frame.

---

## 2. `qpos_to_smpl()` — Exact Code (Lines 552–571)

### Function Definition with Annotations

```python
def qpos_to_smpl(qpos, mj_model, smpl_model="smpl"):
    """
    Reverse conversion: MuJoCo qpos → SMPL pose (axis-angle)
    """
    
    # ========== STEP 1: RETRIEVE BODY JOINT INDICES ==========
    body_qposaddr = get_body_qposaddr(mj_model)
    # 
    # Returns: Dict[str, Tuple[int, int]]
    # Example: {"Pelvis": (0, 7), "L_Hip": (7, 10), "R_Hip": (10, 13), ...}
    #          (start_idx, end_idx) in qpos for each joint
    
    batch_size = qpos.shape[0]
    
    # ========== STEP 2: EXTRACT AND ADJUST TRANSLATION ==========
    trans = qpos[:, :3] - mj_model.body_pos[1]
    # 
    # ❌ NO COORDINATE TRANSFORM: Just removes offset
    # qpos[:, :3] = position from MuJoCo
    # mj_model.body_pos[1] = root offset
    # Result: position relative to root in world frame (same frame as input)
    
    # ========== STEP 3: SELECT SMPL BONE ORDER ==========
    smpl_bones_to_use = (
        SMPL_BONE_ORDER_NAMES
        if smpl_model == "smpl" else SMPLH_BONE_ORDER_NAMES
    )
    # = [Pelvis, L_Hip, R_Hip, Torso, L_Knee, ..., R_Hand]
    
    # ========== STEP 4: Initialize output ==========
    pose = np.zeros([batch_size, len(smpl_bones_to_use), 3])
    # (batch, 24, 3) for SMPL = (batch, 24 joints, 3 axis-angle dims)
    
    # ========== STEP 5: CONVERT EACH JOINT ==========
    for ind1, bone_name in enumerate(smpl_bones_to_use):
        ind2 = body_qposaddr[bone_name]  # (start_idx, end_idx) in qpos
        
        if ind1 == 0:  # ROOT (Pelvis)
            # ❌ NO COORDINATE TRANSFORM: Just representation change
            quat = qpos[:, 3:7]  # Extract quaternion from qpos
            
            # Reorder quaternion for SciPy compatibility
            pose[:, ind1, :] = sRot.from_quat(
                quat[:, [1, 2, 3, 0]]  # Reorder: [a,b,c,d] → [b,c,d,a]
            ).as_rotvec()               # Convert to axis-angle
        else:  # BODY JOINTS
            # ❌ NO COORDINATE TRANSFORM: Just representation change
            pose[:, ind1, :] = sRot.from_euler(
                "ZYX",                  # Euler convention (same as forward pass)
                qpos[:, ind2[0]:ind2[1]]  # Extract Euler angles from qpos
            ).as_rotvec()               # Convert to axis-angle

    return pose, trans
```

### Key Observations

1. **Quaternion reordering** (Line 564): `quat[:, [1, 2, 3, 0]]`
   - If qpos stores `[x, y, z, w]`, this extracts `[y, z, w, x]`
   - SciPy `from_quat()` expects `[x, y, z, w]` or specific format
   - This reordering ensures compatibility

2. **Euler convention** (Line 567): `"ZYX"` matches forward pass (Line 388)
   - Ensures reversibility: `qpos_to_smpl(smpl_to_qpose(...)) ≈ ...`

3. **No coordinate system change**: All operations are representation conversions only

---

## 3. Coordinate Transform Evidence Table

| Element | Forward Pass | Reverse Pass | Coordinate Transform? |
|---------|--------------|--------------|----------------------|
| **Translation** | Add `body_pos[1]` offset | Subtract `body_pos[1]` offset | ❌ NO (just offset) |
| **Root rotation** | Axis-angle → Quat | Quat → Axis-angle | ❌ NO (representation) |
| **Body rotations** | Axis-angle → Euler ZYX | Euler ZYX → Axis-angle | ❌ NO (representation) |
| **Joint reordering** | Apply `smpl_2_mujoco` | Read from `body_qposaddr` | ❌ NO (order mapping) |
| **Axis swapping** | None | None | ❌ NO |
| **Frame rotation** | None | None | ❌ NO |
| **Scaling** | None | None | ❌ NO |

---

## 4. Root vs Body Joint Treatment

### Forward: `smpl_to_qpose()`

```
ROOT (Pelvis):
  Input:   Axis-angle (3D) in SMPL frame
  → Matrix (3×3)
  → Quaternion (4D) ← SPECIAL CASE
  Output:  Quaternion (4D) in same frame (NO transform)

BODY (L_Hip, ..., R_Hand):
  Input:   Axis-angle (3D) in SMPL frame
  → Matrix (3×3)
  → Euler ZYX (3D)
  → Reorder by smpl_2_mujoco
  Output:  Euler ZYX (3D) in MuJoCo order, same frame (NO transform)
```

### Reverse: `qpos_to_smpl()`

```
ROOT (Pelvis):
  Input:   Quaternion (4D) at qpos[3:7] (MuJoCo order)
  → Axis-angle (3D) ← SPECIAL CASE
  Output:  Axis-angle (3D) in SMPL frame (NO transform)

BODY (L_Hip, ..., R_Hand):
  Input:   Euler ZYX (3D) at qpos[ind2[0]:ind2[1]] (MuJoCo order)
  → Axis-angle (3D)
  Output:  Axis-angle (3D) in SMPL frame (NO transform)
```

---

## 5. smpl_2_mujoco Building & Usage

### Building (Lines 371–374)

```python
smpl_2_mujoco = [
    joint_names.index(q)                      # Find SMPL index
    for q in list(get_body_qposaddr(mj_model).keys())  # For each MuJoCo body
    if q in joint_names                       # That exists in SMPL
]
```

**Process**:
1. `get_body_qposaddr(mj_model).keys()` returns MuJoCo body names in order
2. For each name, find its index in `joint_names` (SMPL order)
3. Result is a list of SMPL indices corresponding to MuJoCo order

### Usage (Lines 391–393)

```python
curr_spose = curr_spose.reshape(
    -1, num_joints,
    4 if use_quat else 3
)[:, smpl_2_mujoco, :].reshape(-1, num_angles)
```

**Process**:
1. Input: `(batch, 24, 3)` in SMPL order
2. Index with `[:, smpl_2_mujoco, :]` to reorder
3. Output: `(batch, 24, 3)` in MuJoCo order
4. Flatten to `(batch, 72)`

**Example**:
```
Input array (SMPL order):
  Joint 0: Pelvis euler [a, b, c]
  Joint 1: L_Hip euler  [d, e, f]
  Joint 2: R_Hip euler  [g, h, i]
  Joint 3: Torso euler  [j, k, l]

smpl_2_mujoco = [0, 1, 3, 2]  # MuJoCo order: Pelvis, L_Hip, Torso, R_Hip

After [:, smpl_2_mujoco, :]:
  Index 0: Pelvis euler [a, b, c]  ← from SMPL index 0
  Index 1: L_Hip euler  [d, e, f]  ← from SMPL index 1
  Index 2: Torso euler  [j, k, l]  ← from SMPL index 3
  Index 3: R_Hip euler  [g, h, i]  ← from SMPL index 2
```

---

## 6. Where Coordinate Conversion Might Occur

Since `smpl_to_qpose()` doesn't do Y-up ↔ Z-up conversion, it must happen elsewhere:

### Option 1: `normalize_smpl_pose()` (Lines 607–635)

```python
def normalize_smpl_pose(pose_aa, trans=None, random_root=False):
    root_aa = pose_aa[0, :3]
    root_rot = sRot.from_rotvec(np.array(root_aa))
    root_euler = np.array(root_rot.as_euler("xyz", degrees=False))
    target_root_euler = root_euler.copy()
    if random_root:
        target_root_euler[2] = np.random.random(1) * np.pi * 2
    else:
        target_root_euler[2] = -1.57
    target_root_rot = sRot.from_euler("xyz", target_root_euler, degrees=False)
    target_root_aa = target_root_rot.as_rotvec()

    target_root_mat = target_root_rot.as_matrix()
    root_mat = root_rot.as_matrix()
    apply_mat = np.matmul(target_root_mat, np.linalg.inv(root_mat))

    if torch.is_tensor(pose_aa):
        pose_aa = vertizalize_smpl_root(pose_aa, root_vec=target_root_aa)
    else:
        pose_aa = vertizalize_smpl_root(torch.from_numpy(pose_aa),
                                        root_vec=target_root_aa)

    if not trans is None:
        trans[:, [0, 1]] -= trans[0, [0, 1]]
        trans[:, [2]] = trans[:, [2]] - trans[0, [2]] + 0.91437225
        trans = np.matmul(apply_mat, trans.T).T
    return pose_aa, trans
```

**Possible coordinate adjustment**: 
- `apply_mat = target_root_mat @ inv(root_mat)` could include frame rotation
- `trans = apply_mat @ trans.T` applies this to translation
- But this is **optional** (only if `normalize=True`)

### Option 2: Caller Preprocessing

The caller of `smpl_to_qpose()` might preprocess SMPL data:
- Convert from Y-up (SMPL) to Z-up (MuJoCo) before calling
- Apply a fixed rotation matrix to all poses

### Option 3: MuJoCo Renderer

The MuJoCo visualization/rendering might handle coordinate display, not the physics.

---

## 7. Summary

| Aspect | Answer | Evidence |
|--------|--------|----------|
| **Root gets coord transform?** | ❌ NO | Lines 397-399: just Quat conversion, no frame change |
| **Body joints get coord transform?** | ❌ NO | Lines 388-393: just Euler conversion, no frame change |
| **Y-up ↔ Z-up handled here?** | ❌ NO | No axis swapping, no frame matrices in code |
| **smpl_2_mujoco is coord transform?** | ❌ NO | Just list of indices, only reordering |
| **Where does it happen?** | UNKNOWN | Possibly `normalize_smpl_pose()`, caller, or MuJoCo renderer |
| **Reversible?** | ✅ YES | `qpos_to_smpl()` perfectly inverts `smpl_to_qpose()` |

