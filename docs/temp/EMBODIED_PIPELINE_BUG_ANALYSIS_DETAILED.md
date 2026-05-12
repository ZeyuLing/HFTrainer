# Embodied Pipeline Retargeting - Comprehensive Bug Analysis

## Executive Summary

The embodied pipeline has **multiple critical bugs** causing the reported quality issues (foot sliding, ground penetration, deformed poses, joint limit violations). The issues span coordinate system mismatches, quaternion conventions, and a dangerous ground correction flow.

---

## Critical Bugs Found

### 🔴 BUG #1: COORDINATE SYSTEM MISMATCH IN GROUND CORRECTION FLOW (PIPELINE LOGIC ERROR)

**Location**: `pipeline_motion_to_robot.py`, line 125

**Issue**: The pipeline passes `--no-offset-to-ground` to GMR but relies on FK-based ground correction later. However, this creates a **double-failure scenario**:

```python
gmr_cmd = [
    sys.executable, SCRIPT_DIR / "gmr_retarget_headless.py",
    ...
    "--no-offset-to-ground",  # FK correction handles grounding  <-- LINE 125
]
```

**Problems**:
1. GMR outputs `root_pos` without ground offset applied (feet may be below Z=0)
2. FK correction in `gmr_to_protomotions.py` tries to fix this, BUT...
3. The FK correction only adjusts root_pos Z, doesn't recalculate joint angles
4. When feet are initially below ground, the FK correction has to raise the robot excessively
5. This creates **unnatural tall poses** or **impossible joint configurations**

**Why it causes foot sliding**:
- Without per-frame grounding in GMR (offset_to_ground=False), the IK solver outputs a Z position assuming the target feet ARE on ground
- Later FK correction adjusts root_pos Z upward, but the joint angles weren't computed for this adjusted height
- The ankle/hip angles don't match the adjusted Z position → feet slide to stay "planted" relative to the joint angles

**Fix Needed**:
- EITHER: Keep `offset_to_ground=True` in GMR and disable FK correction
- OR: Run FK correction BEFORE converting to robot frame, not after

---

### 🔴 BUG #2: FK GROUND CORRECTION ASSUMES FEET AT INDICES [7, 13] (HARDCODED WRONG INDICES)

**Location**: `gmr_to_protomotions.py`, lines 155-229

**Issue**: FK correction uses hardcoded foot body indices [7, 13]:

```python
def fk_ground_correction(mjcf_path, root_pos, root_rot_xyzw, dof_pos, 
                        foot_body_indices=None, ...):
    if foot_body_indices is None:
        foot_body_indices = [7, 13]  # left_ankle_roll_link, right_ankle_roll_link  <-- LINE 184
```

**Problem**: 
- These indices refer to body IDs in MuJoCo's internal ordering
- The actual foot bodies depend on the MJCF structure
- For G1 robot, the actual foot link indices may be different
- Using wrong indices causes the algorithm to check non-foot bodies (hip, knee, etc.)
- **Result**: Root position corrected based on wrong body heights → feet end up below/above ground

**How to verify**:
- Check `g1_holo_compat.xml` body ordering
- The foot bodies might be at different indices

**Fix Needed**:
- Parse MJCF to find foot body names/indices at runtime
- OR: Pass body names and look them up at load time
- Document the correct G1 foot indices

---

### 🔴 BUG #3: QUATERNION CONVENTION ERROR IN FK GROUND CORRECTION

**Location**: `gmr_to_protomotions.py`, lines 209-211

**Issue**: The code converts quaternions back and forth incorrectly:

```python
for t in range(T):
    # Set qpos with current root_pos
    root_rot_wxyz = quat_xyzw_to_wxyz(root_rot_xyzw[t])  # xyzw→wxyz
    data.qpos[:3] = root_pos[t]
    data.qpos[3:7] = root_rot_wxyz
    data.qpos[7:] = dof_pos[t]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    
    # Find minimum foot Z
    min_foot_z = np.inf
    for bi in foot_body_indices:
        foot_z = data.xpos[bi + 1][2]  # <-- WRONG INDEX OFFSET
```

**Multiple issues**:
1. `bi + 1` offset assumes world body is at index 0, but this isn't always true
2. If body indices are already correct (not relative to world), +1 causes array overflow
3. No bounds checking → reads garbage memory or crashes silently

**Fix Needed**:
- Check actual MuJoCo body indexing for the specific MJCF
- Remove or verify the +1 offset

---

### 🔴 BUG #4: ROOT ROTATION FRAME CONVERSION IN `gmr_to_protomotions.py` (OVERCOMPLICATED & POTENTIALLY WRONG)

**Location**: `gmr_to_protomotions.py`, lines 52-90

**Issue**: The code attempts to remove GMR's rot_offset to convert from SMPL-X (Y-up) to MuJoCo (Z-up):

```python
def remove_gmr_root_offset(root_rot_xyzw):
    """Remove the Y-up→Z-up frame conversion baked into GMR's pelvis quaternion."""
    rot_offset = _get_gmr_rot_offset()
    root_rots = R.from_quat(root_rot_xyzw)
    corrected = root_rots * rot_offset.inv()  # <-- RIGHT MULTIPLY
    return corrected.as_quat()
```

**Problems**:
1. The comment says "right-multiply by rot_offset.inv()" but the math assumes it's the only rotation
2. If GMR actually applied `q_out = q_smplx * rot_offset`, then undoing requires `q_corrected = q_out * rot_offset.inv()` (which is what the code does)
3. BUT: The root_pos conversion does something different (applies rot_offset.inv() as a rotation matrix):

```python
def convert_root_pos_to_zup(root_pos):
    rot_offset = _get_gmr_rot_offset()
    return rot_offset.inv().apply(root_pos)  # <-- APPLIES AS ACTIVE ROTATION
```

**Inconsistency**: 
- Root rotation: right-multiply in quaternion space
- Root position: apply as active rotation
- These don't commute! The frame conversion should be consistent for position and rotation

**Expected behavior**: 
- If you transform position by a rotation, you must transform rotation the same way
- If p' = R * p, then q' = q * R (in quaternion form)
- But here position and rotation use different conventions

**Fix Needed**:
- Verify GMR's exact frame conversion
- Make position and rotation conversions consistent (both active or both passive)

---

### 🟡 BUG #5: MOTION135 ROTATION INTERPRETATION (POTENTIAL ROW-MAJOR/COLUMN-MAJOR ERROR)

**Location**: `motion135_to_smplx.py`, lines 26-55

**Issue**: The code claims HyMotion uses row-major layout and reorders [0,2,4,1,3,5]:

```python
def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """
    HyMotion outputs rot6d in row-major layout: [R00,R01, R10,R11, R20,R21]
    Gram-Schmidt expects column-major layout: [R00,R10,R20, R01,R11,R21]
    We reorder [0,2,4,1,3,5] to convert row-major → column-major before decoding.
    """
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
```

**Problem**:
- The reordering assumes a specific layout: `[R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]]`
- After reorder [0,2,4,1,3,5]: `[R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]]`
- This matches column-major of first 2 columns
- But is this what HyMotion actually outputs?
- **No verification or reference provided**

**Why it matters**:
- Wrong interpretation → rotation matrix rows/columns swapped → pose distorted
- Limbs could appear rotated or bent incorrectly

**Fix Needed**:
- Cross-reference with HyMotion M2M source code
- Verify the rot6d layout matches the claim
- Add a test: compute a known pose (T-pose) and verify it produces correct joint angles

---

### 🟡 BUG #6: MISSING VALIDATION IN FK GROUND CORRECTION (NUMERICAL STABILITY)

**Location**: `gmr_to_protomotions.py`, line 226

**Issue**: The Z offset calculation doesn't check if ground_clearance is achievable:

```python
z_offset = ground_clearance - min_foot_z
corrected_root_pos[t, 2] = root_pos[t, 2] + z_offset
```

**Problems**:
1. If min_foot_z is very negative (e.g., -0.5m from a deformed pose), z_offset becomes huge
2. This could move root far above ground, making joint angles infeasible
3. Subsequent FK with new root_pos[t,2] will produce different min_foot_z
4. But the algorithm doesn't re-check! It applies the offset once and stops
5. **Result**: Ground correction over-corrects for bad poses

**Better approach**: 
- Iterate: adjust root_pos, recompute FK, check foot height, adjust again
- Add max_z_correction limit to prevent overcorrection

---

### 🟡 BUG #7: JOINT LIMIT CLAMPING NOT MENTIONED

**Location**: Throughout `gmr_to_protomotions.py`

**Issue**: The pipeline outputs `dof_pos` without any joint limit validation:

```python
dof_pos_resampled = dof_interp(times_tgt).astype(np.float32)
# No clipping to joint limits!
```

**Why it matters**:
- GMR's IK solver might output angles at mechanical limits (elbows fully extended, knees locked)
- The IK solver should respect limits, but if it doesn't, this pipeline won't catch it
- Resampling could extrapolate beyond limits
- **Result**: Deformed poses, mechanical violations visible in simulation

**Fix Needed**:
- Load robot joint limits from MJCF
- Clamp dof_pos to valid ranges
- Log violations for debugging

---

### 🟡 BUG #8: SMPL-X NPZ LOADING ASSUMES SPECIFIC DATA KEYS

**Location**: `motion135_to_smplx.py`, line 77

**Issue**: Loading relies on correct NPZ keys but no verification:

```python
data = np.load(input_npz, allow_pickle=True)
motion = data['motion_135']  # Assumes key exists!
```

**Problems**:
1. If input has wrong key (e.g., 'motion135' without underscore), fails silently or crashes
2. No check for required fields
3. No dimension validation (assumes (T, 135))

**Fix Needed**:
- Validate NPZ keys before using
- Check shapes match expectations
- Print detailed error messages

---

### 🟡 BUG #9: RESAMPLING EDGE CASE (FIRST/LAST FRAMES)

**Location**: `gmr_to_protomotions.py`, lines 363-365

**Issue**: Velocity computation repeats first frame value:

```python
dof_vel = np.zeros_like(dof_pos)
dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt
dof_vel[0] = dof_vel[1]  # Repeat first computed velocity
```

**Problem**:
- First frame velocity is zero → discontinuity
- At animation start, sudden velocity jump on frame 1
- Less critical but causes non-physical behavior

**Better approach**: Use central differences or mirror the velocity of frame 1

---

### 🟡 BUG #10: NO VALIDATION THAT ROOT_ROT IS UNIT QUATERNION

**Location**: After `remove_gmr_root_offset()` in `gmr_to_protomotions.py`

**Issue**: After frame conversions, quaternions might not be normalized:

```python
corrected = root_rots * rot_offset.inv()
return corrected.as_quat()  # Should be unit quaternion
```

**Problem**:
- Quaternion multiplication should produce unit quat if both inputs are unit
- But numerical errors can accumulate
- scipy normalizes by default, but not guaranteed
- Non-unit quaternions cause rotation errors in FK

**Fix Needed**:
- Normalize quaternions after conversions
- Add assertion checks

---

## Summary Table

| Bug # | Severity | Component | Issue | Impact |
|-------|----------|-----------|-------|--------|
| 1 | 🔴 CRITICAL | Pipeline orchestrator | Double-failure ground correction flow | Foot sliding, unnatural poses |
| 2 | 🔴 CRITICAL | FK ground correction | Hardcoded wrong foot indices | Feet below/above ground |
| 3 | 🔴 CRITICAL | FK ground correction | Wrong body index offset in MuJoCo | Array overflow, garbage data |
| 4 | 🔴 CRITICAL | Frame conversion | Inconsistent position/rotation transforms | Pose misalignment |
| 5 | 🟡 HIGH | motion135_to_smplx | Unverified rot6d layout assumption | Distorted joints/limbs |
| 6 | 🟡 HIGH | FK ground correction | No iteration/validation on correction | Overcorrection, infeasible poses |
| 7 | 🟡 MEDIUM | gmr_to_protomotions | No joint limit clamping | Mechanical violations |
| 8 | 🟡 MEDIUM | motion135_to_smplx | No NPZ key validation | Silent failures |
| 9 | 🟡 LOW | gmr_to_protomotions | First frame velocity edge case | Minor animation discontinuity |
| 10 | 🟡 LOW | gmr_to_protomotions | No quaternion normalization check | Rotation errors (cumulative) |

---

## Recommended Fix Priority

### Phase 1 (Critical - Fix First)
1. **Bug #1**: Review ground correction strategy. Choose ONE:
   - Enable offset_to_ground in GMR, disable FK correction
   - OR: Move FK correction to correct place in pipeline
   
2. **Bug #2**: Verify foot body indices for G1 MJCF
   - Parse MJCF at runtime to find "ankle" or foot body indices
   - Add unit test with known robot state

3. **Bug #3**: Fix MuJoCo body index offset
   - Understand MuJoCo body indexing for loaded MJCF
   - Remove incorrect +1 if world body not at 0

### Phase 2 (High - Fix Next)
4. **Bug #4**: Verify frame conversion consistency
   - Cross-check GMR's IK config for exact rot_offset application
   - Test with known poses (T-pose, arm extended, etc.)

5. **Bug #5**: Validate motion135 rot6d layout
   - Compare with HyMotion source
   - Add regression test with ground truth poses

### Phase 3 (Medium - Best Practices)
6. **Bug #6**: Add FK ground correction validation loop
7. **Bug #7**: Add joint limit clamping
8. **Bug #8**: Add NPZ validation
9. **Bug #9**: Fix first frame velocity
10. **Bug #10**: Add quaternion normalization

---

## Testing Strategy

1. **Unit Tests**:
   - rot6d_to_rotmat with known T-pose
   - Frame conversions with test quaternions
   - FK ground correction with fixed poses

2. **Integration Tests**:
   - Run full pipeline with ground truth motion
   - Check for foot penetration, joint limits, pose plausibility
   - Compare before/after fixes

3. **Debugging Tools**:
   - Visualization: Render robot at each stage
   - Diagnostics: Print body positions before/after FK correction
   - Validation: Log joint angles vs limits

---

## Flags to Watch

- `--no-offset-to-ground`: Disables GMR per-frame grounding (related to Bug #1)
- `--no-fk-ground-correction`: Disables FK-based Z adjustment (related to Bug #1)
- `--ground-clearance`: FK target foot height (used in Bug #2, #6)

**Current Pipeline Logic**:
- Passes `--no-offset-to-ground` to GMR
- Relies on `--fk-ground-correction` (default True)
- **This is the dangerous double-failure configuration described in Bug #1**
