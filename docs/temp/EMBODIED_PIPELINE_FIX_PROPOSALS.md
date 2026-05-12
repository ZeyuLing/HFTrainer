# Embodied Pipeline - Proposed Fixes with Code

## Bug #1: Double-Failure Ground Correction Flow (CRITICAL)

**Current Code (pipeline_motion_to_robot.py:119-126)**:
```python
gmr_cmd = [
    sys.executable, SCRIPT_DIR / "gmr_retarget_headless.py",
    "--smplx_file", str(smplx_path),
    "--robot", args.robot,
    "--save_path", str(gmr_pkl_path),
    "--tgt_fps", str(args.tgt_fps),
    "--no-offset-to-ground",  # <-- PROBLEM: Disables per-frame grounding
]
```

**Problem**: 
- GMR outputs feet potentially below ground
- FK correction tries to fix but only adjusts Z, not joint angles
- Causes foot sliding and unnatural poses

**Fix Option A: Enable offset_to_ground in GMR (RECOMMENDED - Simpler)**:
```python
gmr_cmd = [
    sys.executable, SCRIPT_DIR / "gmr_retarget_headless.py",
    "--smplx_file", str(smplx_path),
    "--robot", args.robot,
    "--save_path", str(gmr_pkl_path),
    "--tgt_fps", str(args.tgt_fps),
    # Remove --no-offset-to-ground, let GMR handle grounding
]
```

Then disable FK correction:
```python
proto_cmd = [
    sys.executable, SCRIPT_DIR / "gmr_to_protomotions.py",
    "--input", str(gmr_pkl_path),
    "--output", str(output_path),
    "--mjcf", str(args.mjcf),
    "--control-dt", str(args.control_dt),
    "--no-fk-ground-correction",  # <-- Disable redundant correction
]
```

**Fix Option B: Correct FK grounding logic (if FK is needed)**:
```python
# Would need to run FK correction on SMPL-X data BEFORE GMR retargeting
# This is more complex and requires restructuring the pipeline
# Not recommended unless GMR's offset_to_ground has issues
```

---

## Bug #2: Hardcoded Wrong Foot Body Indices (CRITICAL)

**Current Code (gmr_to_protomotions.py:183-184)**:
```python
if foot_body_indices is None:
    foot_body_indices = [7, 13]  # left_ankle_roll_link, right_ankle_roll_link
```

**Problem**: Hardcoded indices likely wrong for G1 MJCF structure

**Fix: Dynamic body index lookup**:
```python
def get_foot_body_indices_from_mjcf(mjcf_path):
    """Extract foot body indices from MJCF at runtime."""
    import mujoco
    import tempfile
    import os
    from pathlib import Path
    
    # Patch and load MJCF
    patched_xml = _patch_mjcf_xml(mjcf_path)
    asset_dir = str(Path(mjcf_path).parent)
    
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=asset_dir, delete=False
    ) as tmp:
        tmp.write(patched_xml)
        tmp_path = tmp.name
    
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    
    # Search for foot bodies
    foot_indices = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and ("foot" in name.lower() or "ankle" in name.lower()):
            # Convert from MuJoCo body index to data.xpos index
            # data.xpos is (nbody,) where index 0 is world
            foot_indices.append(i)
    
    if not foot_indices:
        print("WARNING: No foot bodies found, using default [7, 13]")
        return [7, 13]
    
    print(f"Found foot bodies at indices: {foot_indices}")
    return foot_indices


def fk_ground_correction(mjcf_path, root_pos, root_rot_xyzw, dof_pos, 
                        foot_body_indices=None, ground_clearance=0.0):
    """Correct root_pos Z so feet are at ground level based on FK."""
    import mujoco
    import tempfile
    import os

    if foot_body_indices is None:
        foot_body_indices = get_foot_body_indices_from_mjcf(mjcf_path)

    # ... rest of function unchanged
```

---

## Bug #3: Wrong MuJoCo Body Index Offset (CRITICAL)

**Current Code (gmr_to_protomotions.py:217-220)**:
```python
min_foot_z = np.inf
for bi in foot_body_indices:
    foot_z = data.xpos[bi + 1][2]  # <-- WRONG: +1 offset
    if foot_z < min_foot_z:
        min_foot_z = foot_z
```

**Problem**: 
- `data.xpos` already includes world body at index 0
- Using `bi + 1` assumes the indices are 0-indexed relative to bodies, but they're already world-inclusive
- OR the indices are already correct, and +1 causes overflow

**Fix: Remove +1 offset (most likely)**:
```python
min_foot_z = np.inf
for bi in foot_body_indices:
    if bi < len(data.xpos):  # Add bounds check
        foot_z = data.xpos[bi][2]  # Remove +1
        if foot_z < min_foot_z:
            min_foot_z = foot_z
```

**Alternative: Verify index is body index (not data index)**:
```python
# If foot_body_indices come from get_foot_body_indices_from_mjcf(),
# they're MuJoCo body indices (0=world, 1+=bodies)
# data.xpos[0] is world, data.xpos[i] is body i
# So the correct code is:
for bi in foot_body_indices:
    # bi is already a valid data.xpos index (0=world, 1+=bodies)
    foot_z = data.xpos[bi][2]
```

---

## Bug #4: Inconsistent Frame Conversion (CRITICAL)

**Current Code (gmr_to_protomotions.py:69-111)**:

Root rotation conversion (right-multiply):
```python
def remove_gmr_root_offset(root_rot_xyzw):
    rot_offset = _get_gmr_rot_offset()
    root_rots = R.from_quat(root_rot_xyzw)
    corrected = root_rots * rot_offset.inv()  # Right multiply
    return corrected.as_quat()
```

Root position conversion (active rotation):
```python
def convert_root_pos_to_zup(root_pos):
    rot_offset = _get_gmr_rot_offset()
    return rot_offset.inv().apply(root_pos)  # Apply as active rotation
```

**Problem**: These use different conventions and may not be consistent

**Fix: Verify & Document the convention**:
```python
def remove_gmr_root_offset(root_rot_xyzw):
    """Remove GMR's Y-up→Z-up frame conversion from root rotation.
    
    GMR applies rot_offset to convert SMPL-X (Y-up) to MuJoCo (Z-up).
    If GMR applies: q_gmr = q_smplx * rot_offset (left multiply in quat mult)
    Then we undo:   q_corrected = q_gmr * rot_offset.inv() (right multiply)
    
    To verify consistency, check if position and rotation conversions
    produce aligned results with a test vector.
    """
    rot_offset = _get_gmr_rot_offset()
    root_rots = R.from_quat(root_rot_xyzw)
    corrected = root_rots * rot_offset.inv()
    return corrected.as_quat()


def convert_root_pos_to_zup(root_pos):
    """Convert root position from SMPL-X (Y-up) to MuJoCo (Z-up).
    
    Consistency check: If q maps [1,0,0] to [z,x,y], then position
    [x,y,z] in SMPL-X should map to [z,x,y] in MuJoCo.
    """
    rot_offset = _get_gmr_rot_offset()
    pos_converted = rot_offset.inv().apply(root_pos)
    
    # Debug: Verify consistency
    test_q = np.array([0, 0, 0, 1], dtype=np.float32)  # Identity
    test_q_corrected = np.array(remove_gmr_root_offset(test_q))
    test_pos = np.array([1, 0, 0], dtype=np.float32)
    test_pos_corrected = convert_root_pos_to_zup(test_pos)
    
    # For identity quat, position should just be rotated by rot_offset.inv()
    # Both should produce consistent transforms
    
    return pos_converted.astype(root_pos.dtype)
```

**Recommendation**: Add test case to verify consistency:
```python
def test_frame_conversion_consistency():
    """Verify root position and rotation conversions are consistent."""
    from scipy.spatial.transform import Rotation as R_test
    
    # Test vectors
    test_positions = np.array([
        [1, 0, 0],  # X axis in SMPL-X
        [0, 1, 0],  # Y axis (height in SMPL-X)
        [0, 0, 1],  # Z axis in SMPL-X
    ], dtype=np.float32)
    
    test_quaternions = np.array([
        [0, 0, 0, 1],  # Identity
        [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2],  # 90° around X
    ], dtype=np.float32)
    
    for pos in test_positions:
        pos_conv = convert_root_pos_to_zup(pos.reshape(1, 3))[0]
        print(f"Position {pos} -> {pos_conv}")
    
    for quat in test_quaternions:
        quat_conv = remove_gmr_root_offset(quat.reshape(1, 4))[0]
        print(f"Quat {quat} -> {quat_conv}")
    
    # Verify: rotating [1,0,0] by quat should give same result as
    # rotating by (rot_offset.inv() applied to quaternion)
    print("\nIf these match, frame conversions are consistent!")
```

---

## Bug #5: Unverified rot6d Layout (HIGH)

**Current Code (motion135_to_smplx.py:26-55)**:
```python
def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """
    HyMotion outputs rot6d in row-major layout: [R00,R01, R10,R11, R20,R21]
    Gram-Schmidt expects column-major layout: [R00,R10,R20, R01,R11,R21]
    We reorder [0,2,4,1,3,5] to convert row-major → column-major before decoding.
    """
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]  # <-- UNVERIFIED REORDER
    # ... Gram-Schmidt
```

**Problem**: No verification this is correct

**Fix: Add validation test**:
```python
def validate_rot6d_layout():
    """Verify that the rot6d reordering [0,2,4,1,3,5] is correct.
    
    Test with known rotations and compare with expected SMPL-X joint angles.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    # Create known rotation: 45° around Z axis (yaw)
    angle = np.pi / 4
    known_rot = R.from_rotvec([0, 0, angle])
    known_mat = known_rot.as_matrix()  # (3, 3)
    
    # Assume row-major layout (needs verification from HyMotion code)
    rot6d_row_major = np.array([
        known_mat[0, 0], known_mat[0, 1],
        known_mat[1, 0], known_mat[1, 1],
        known_mat[2, 0], known_mat[2, 1],
    ])
    
    # Apply reorder [0,2,4,1,3,5]
    rot6d_reordered = rot6d_row_major[[0, 2, 4, 1, 3, 5]]
    
    # Decode via Gram-Schmidt (from rot6d_to_rotmat)
    a1 = rot6d_reordered[:3]
    a2 = rot6d_reordered[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    dot = np.sum(b1 * a2)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    reconstructed = np.stack([b1, b2, b3], axis=-1)
    
    # Compare
    error = np.abs(reconstructed - known_mat).max()
    print(f"Reconstruction error: {error:.8f}")
    
    if error > 1e-6:
        print("ERROR: rot6d reordering [0,2,4,1,3,5] produces wrong rotations!")
        print("Reconstruction:")
        print(reconstructed)
        print("Expected:")
        print(known_mat)
        raise ValueError("rot6d layout verification failed")
    else:
        print("✓ rot6d layout [0,2,4,1,3,5] is correct")


def convert_motion135_to_smplx(input_npz, output_npz, fps=30):
    """Convert motion_135 NPZ to SMPL-X NPZ format."""
    # Add validation
    print("Validating rot6d layout...")
    validate_rot6d_layout()
    
    # ... rest of function
```

---

## Bug #6: FK Ground Correction Overcorrection (HIGH)

**Current Code (gmr_to_protomotions.py:207-228)**:
```python
for t in range(T):
    # Set qpos...
    mujoco.mj_forward(model, data)
    
    # Find minimum foot Z
    min_foot_z = np.inf
    for bi in foot_body_indices:
        foot_z = data.xpos[bi + 1][2]
        if foot_z < min_foot_z:
            min_foot_z = foot_z
    
    # Single pass: apply offset
    z_offset = ground_clearance - min_foot_z
    corrected_root_pos[t, 2] = root_pos[t, 2] + z_offset
    # No re-verification!
```

**Problem**: Single-pass correction can overshoot, no iteration

**Fix: Add iterative correction with validation**:
```python
def fk_ground_correction_iterative(mjcf_path, root_pos, root_rot_xyzw, dof_pos, 
                                   foot_body_indices=None, ground_clearance=0.0,
                                   max_iterations=3, tolerance=0.001):
    """Correct root_pos Z iteratively so feet are at ground level.
    
    Because changing root_pos Z doesn't change DOF angles, a single
    pass of correction might overshoot if the initial pose was deformed.
    This version iterates to converge.
    """
    import mujoco
    import tempfile
    import os

    if foot_body_indices is None:
        foot_body_indices = get_foot_body_indices_from_mjcf(mjcf_path)

    patched_xml = _patch_mjcf_xml(mjcf_path)
    asset_dir = str(Path(mjcf_path).parent)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=asset_dir, delete=False
    ) as tmp:
        tmp.write(patched_xml)
        tmp_path = tmp.name

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)

    data = mujoco.MjData(model)

    T = root_pos.shape[0]
    corrected_root_pos = root_pos.copy()
    foot_min_z_before = np.zeros(T, dtype=np.float64)
    iterations_used = np.zeros(T, dtype=np.int32)

    for t in range(T):
        current_z = corrected_root_pos[t, 2]
        
        for iteration in range(max_iterations):
            # Set qpos with current root_pos
            root_rot_wxyz = quat_xyzw_to_wxyz(root_rot_xyzw[t])
            data.qpos[:3] = corrected_root_pos[t]
            data.qpos[3:7] = root_rot_wxyz
            data.qpos[7:] = dof_pos[t]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            # Find minimum foot Z
            min_foot_z = np.inf
            for bi in foot_body_indices:
                if bi < len(data.xpos):
                    foot_z = data.xpos[bi][2]  # Fixed: removed +1
                    if foot_z < min_foot_z:
                        min_foot_z = foot_z

            if iteration == 0:
                foot_min_z_before[t] = min_foot_z

            # Check if converged
            z_offset = ground_clearance - min_foot_z
            if abs(z_offset) < tolerance:
                iterations_used[t] = iteration + 1
                break

            # Apply correction
            corrected_root_pos[t, 2] = corrected_root_pos[t, 2] + z_offset
            
            if iteration == max_iterations - 1:
                print(f"Frame {t}: FK correction did not converge after {max_iterations} iterations")
                iterations_used[t] = max_iterations
        
        if t % 100 == 0:
            print(f"Frame {t}/{T}: z={corrected_root_pos[t, 2]:.4f}, iterations={iterations_used[t]}")

    print(f"Average iterations used: {iterations_used.mean():.1f}")
    return corrected_root_pos, foot_min_z_before
```

Update the call:
```python
if args.fk_ground_correction:
    print(f"\nApplying iterative FK-based ground correction...")
    root_pos, foot_min_z = fk_ground_correction_iterative(
        args.mjcf, root_pos, root_rot, dof_pos,
        ground_clearance=args.ground_clearance,
        max_iterations=3,
        tolerance=0.001,
    )
```

---

## Bug #7: Joint Limit Clamping (MEDIUM)

**Current Code**: No joint limit checking

**Fix: Add clamping after resampling**:
```python
def clamp_to_joint_limits(dof_pos, mjcf_path):
    """Clamp joint positions to valid ranges from MJCF.
    
    Args:
        dof_pos: (T, N_dof) joint positions
        mjcf_path: Path to MJCF model
    
    Returns:
        dof_pos_clamped: (T, N_dof) with values in valid ranges
        violations: (T,) count of violations per frame
    """
    import mujoco
    import tempfile
    import os
    
    patched_xml = _patch_mjcf_xml(mjcf_path)
    asset_dir = str(Path(mjcf_path).parent)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=asset_dir, delete=False
    ) as tmp:
        tmp.write(patched_xml)
        tmp_path = tmp.name

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    
    # Extract joint limits
    T = dof_pos.shape[0]
    dof_pos_clamped = dof_pos.copy()
    violations = np.zeros(T, dtype=np.int32)
    
    for i in range(min(dof_pos.shape[1], model.nq)):
        if model.jnt_limited[i]:
            limit_min = model.jnt_range[i, 0]
            limit_max = model.jnt_range[i, 1]
            
            # Count violations
            below = (dof_pos[:, i] < limit_min)
            above = (dof_pos[:, i] > limit_max)
            violations += below.astype(np.int32) + above.astype(np.int32)
            
            # Clamp
            dof_pos_clamped[:, i] = np.clip(dof_pos[:, i], limit_min, limit_max)
            
            if (below.sum() + above.sum()) > 0:
                print(f"DOF {i}: clamped {below.sum()} frames below {limit_min:.3f}, "
                      f"{above.sum()} frames above {limit_max:.3f}")
    
    if violations.sum() > 0:
        print(f"\nJoint limit violations: {violations.sum()} total, "
              f"{(violations > 0).sum()} frames affected")
    
    return dof_pos_clamped, violations


# In main():
print(f"\nClamping to joint limits...")
dof_pos_r, violations = clamp_to_joint_limits(dof_pos_r, args.mjcf)
```

---

## Bug #8: NPZ Validation (MEDIUM)

**Current Code (motion135_to_smplx.py:77-79)**:
```python
data = np.load(input_npz, allow_pickle=True)
motion = data['motion_135']  # Assumes key exists!
```

**Fix: Add validation**:
```python
def convert_motion135_to_smplx(input_npz, output_npz, fps=30):
    """Convert motion_135 NPZ to SMPL-X NPZ format."""
    
    # Load with validation
    try:
        data = np.load(input_npz, allow_pickle=True)
    except Exception as e:
        print(f"ERROR: Failed to load {input_npz}: {e}")
        raise

    # Validate keys
    required_keys = ['motion_135']
    for key in required_keys:
        if key not in data.files:
            print(f"ERROR: Required key '{key}' not found in NPZ")
            print(f"Available keys: {data.files}")
            raise KeyError(f"Missing required key: {key}")
    
    motion = data['motion_135']  # (T, 135)
    
    # Validate shape
    if motion.ndim != 2:
        print(f"ERROR: motion_135 should be 2D, got shape {motion.shape}")
        raise ValueError(f"Invalid motion_135 shape: {motion.shape}")
    
    if motion.shape[1] != 135:
        print(f"ERROR: motion_135 should have 135 columns, got {motion.shape[1]}")
        raise ValueError(f"Invalid motion_135 width: {motion.shape[1]}")
    
    # Validate values
    if not np.isfinite(motion).all():
        print(f"WARNING: motion_135 contains NaN or Inf")
        nan_count = (~np.isfinite(motion)).sum()
        print(f"  Non-finite values: {nan_count} / {motion.size}")
    
    T = motion.shape[0]
    print(f"✓ Input validation passed")
    print(f"  Input motion_135 shape: {motion.shape}")
    print(f"  Frames: {T}")
    
    # ... rest of conversion
```

---

## Summary of Fixes by Priority

### Priority 1 (CRITICAL - Do First)
- [ ] **Bug #1**: Remove `--no-offset-to-ground` from pipeline OR verify ground correction strategy
- [ ] **Bug #2**: Implement `get_foot_body_indices_from_mjcf()` for dynamic index lookup
- [ ] **Bug #3**: Remove `+ 1` offset in FK ground correction body index access
- [ ] **Bug #4**: Add documentation and test for frame conversion consistency

### Priority 2 (HIGH - Do Next)
- [ ] **Bug #5**: Add `validate_rot6d_layout()` test to verify reordering
- [ ] **Bug #6**: Replace single-pass with iterative FK correction

### Priority 3 (MEDIUM - Best Practices)
- [ ] **Bug #7**: Add `clamp_to_joint_limits()` with violation reporting
- [ ] **Bug #8**: Add NPZ input validation

### Priority 4 (LOW - Nice to Have)
- [ ] **Bug #9**: Fix first frame velocity handling
- [ ] **Bug #10**: Add quaternion normalization validation

