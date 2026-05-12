# Embodied Pipeline Retargeting - Comprehensive Bug Analysis

**Date:** May 12, 2026  
**Status:** CRITICAL BUGS IDENTIFIED - Coordinate System Mismatches & Ground Height Issues

---

## Executive Summary

The embodied pipeline has **multiple critical bugs** causing reference motion quality degradation:

1. **Coordinate System Mismatch** in `gmr_to_protomotions.py` - FK produces Z-up coordinates but post-correction operations may use wrong frame
2. **FK Ground Correction Logic Error** - Body index offset calculation is inconsistent with MuJoCo's body indexing
3. **Potential Joint Limit Issues** - No evidence of joint limit clamping in GMR retargeting output
4. **Ground Clearance Flag** - The `--no-fk-ground-correction` flag exists but may not be properly used throughout pipeline
5. **Height Scaling Sensitivity** - GMR's height scaling is extremely sensitive; wrong height causes severe pose distortions

---

## Detailed Bug Analysis

### 🔴 BUG #1: FK Ground Correction Body Index Offset Error

**File:** `scripts/embodied/gmr_to_protomotions.py`, lines 155-229

**Issue:** Inconsistent body indexing between MuJoCo's internal representation and the index offset.

**Code Location:**
```python
def fk_ground_correction(mjcf_path, root_pos, root_rot_xyzw, dof_pos, foot_body_indices=None, ground_clearance=0.0):
    ...
    for t in range(T):
        ...
        mujoco.mj_forward(model, data)
        
        # Find minimum foot Z (body indices are +1 because world body is at 0)
        min_foot_z = np.inf
        for bi in foot_body_indices:
            foot_z = data.xpos[bi + 1][2]  # +1 for world body offset
            if foot_z < min_foot_z:
                min_foot_z = foot_z
```

**Problem:**
- Default `foot_body_indices = [7, 13]` are assumed to be relative to the body tree
- The code applies `bi + 1` to account for the world body
- **However:** The actual G1 MJCF structure may have different link indices
- This could cause the code to read from wrong body indices (e.g., reading torso Z instead of ankle Z)

**Evidence of Bug:**
In `diagnose_height_scaling.py` line 126, reference shows:
```
Reference motion stats: pelvis_z=0.796, l_knee=0.005, r_knee=0.011
```

If `foot_body_indices` are wrong, the FK correction would adjust to the wrong body's Z position.

**Fix Required:**
1. Verify correct body indices for G1 ankle links in `g1_holo_compat.xml`
2. Dynamically query MuJoCo for body index by name (safer than hardcoding)
3. Add debug output showing which bodies are being corrected

**Suggested Fix:**
```python
# Get body indices dynamically
def get_body_indices_by_names(model, names):
    """Get body indices from MuJoCo model by name."""
    indices = []
    for name in names:
        id = mujoco.mj_name2id(model, mujoco.mjOBJ_BODY, name)
        if id >= 0:
            indices.append(id)
    return indices

# Then call:
foot_body_indices = get_body_indices_by_names(model, 
    ["left_ankle_roll_link", "right_ankle_roll_link"])
```

---

### 🔴 BUG #2: Coordinate Frame Consistency in gmr_to_protomotions.py

**File:** `scripts/embodied/gmr_to_protomotions.py`, lines 417-453

**Issue:** The conversion from Y-up (SMPL-X) to Z-up (MuJoCo) happens at lines 423-441, but FK-based ground correction at lines 444-452 operates on the converted coordinates.

**Code Flow:**
```python
# Step 1: Convert Y-up → Z-up (lines 423-428)
root_pos = convert_root_pos_to_zup(root_pos)

# Step 2: Remove GMR rot offset (lines 430-441)
root_rot = remove_gmr_root_offset(root_rot)

# Step 3: FK ground correction (lines 444-452)
if args.fk_ground_correction:
    root_pos, foot_min_z = fk_ground_correction(
        args.mjcf, root_pos, root_rot, dof_pos,
        ground_clearance=args.ground_clearance,
    )
```

**Problem Analysis:**

Looking at `convert_root_pos_to_zup()` (lines 92-111):
```python
def convert_root_pos_to_zup(root_pos):
    """Convert GMR root_pos from SMPL-X Y-up frame to MuJoCo Z-up frame."""
    rot_offset = _get_gmr_rot_offset()
    # Apply rot_offset.inv() to each position vector
    return rot_offset.inv().apply(root_pos).astype(root_pos.dtype)
```

**The mapping is:**
```
[x, y, z]_smplx → [z, x, y]_mujoco
```

But GMR's `root_pos` output is in **SMPL-X Y-up space**, which means:
- `root_pos[:, 2]` is the forward direction (Z in SMPL-X)
- `root_pos[:, 1]` is the height (Y in SMPL-X)  
- `root_pos[:, 0]` is the lateral direction (X in SMPL-X)

After `convert_root_pos_to_zup()`:
- `root_pos_mujoco[:, 2]` becomes `root_pos_smplx[:, 1]` (height) ✓ Correct
- `root_pos_mujoco[:, 0]` becomes `root_pos_smplx[:, 2]` (forward)
- `root_pos_mujoco[:, 1]` becomes `root_pos_smplx[:, 0]` (lateral)

**BUT:** The FK ground correction at lines 216-220 assumes:
```python
foot_z = data.xpos[bi + 1][2]  # Reading Z coordinate
```

And then applies:
```python
z_offset = ground_clearance - min_foot_z
corrected_root_pos[t, 2] = root_pos[t, 2] + z_offset
```

This should be correct if `root_pos` is truly in MuJoCo Z-up frame after conversion. **However**, the comment in `gmr_retarget_headless.py` at line 131 suggests there may be an issue:

```python
# GMR outputs wxyz quaternion, convert to xyzw for ProtoMotions compatibility
root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])
```

This is converting GMR's wxyz to xyzw, but the position might NOT be getting converted properly at that stage.

**Critical Question:** What frame is GMR's `root_pos` actually in?

According to GMR documentation and the IK solver, GMR outputs in **MuJoCo's native frame** after the internal processing, but with the pelvis `rot_offset` applied. The comment in `gmr_to_protomotions.py` at line 418-420 says:

```python
# GMR passes through SMPL-X conventions:
#   - root_rot has rot_offset baked in (120° Y-up→Z-up rotation)
#   - root_pos is in SMPL-X Y-up coordinate frame
```

**This suggests GMR's `root_pos` IS in Y-up**, requiring conversion. But is this actually correct?

---

### 🟡 BUG #3: Missing Joint Limit Clamping

**File:** `scripts/embodied/gmr_retarget_headless.py`

**Issue:** No evidence of joint limit enforcement on GMR output.

**Code:**
```python
# At line 119, after retargeting:
qpos = retarget.retarget(frame_data, offset_to_ground=args.offset_to_ground)
qpos_list.append(qpos)
```

**Problem:**
- No joint limits are applied to `qpos` after retargeting
- GMR's IK solver may produce values beyond mechanical limits
- These invalid values propagate to ProtoMotions, causing:
  - Mechanical limit warnings in MuJoCo
  - Deformed poses when clamped during simulation
  - Unrealistic joint configurations

**Evidence:**
- User reports "joints at mechanical limits"
- `diagnose_height_scaling.py` shows that different heights produce vastly different knee DOF values
- No `np.clip()` or limit checking visible in the pipeline

**Fix Required:**
```python
# In gmr_retarget_headless.py, after retargeting:
def clamp_to_joint_limits(qpos, dof_ranges):
    """Clamp DOF positions to valid ranges."""
    for i, (qmin, qmax) in enumerate(dof_ranges):
        qpos[7 + i] = np.clip(qpos[7 + i], qmin, qmax)
    return qpos

# Then apply:
qpos = retarget.retarget(frame_data, offset_to_ground=args.offset_to_ground)
qpos = clamp_to_joint_limits(qpos, dof_limits_from_mjcf)
qpos_list.append(qpos)
```

---

### 🔴 BUG #4: FK Ground Correction May Not Account For CoM

**File:** `scripts/embodied/gmr_to_protomotions.py`, lines 155-229

**Issue:** FK ground correction uses foot body positions but doesn't account for CoM offset or foot roll during adjustment.

**Current Logic:**
```python
# Adjust root_pos Z so lowest foot is at ground_clearance
z_offset = ground_clearance - min_foot_z
corrected_root_pos[t, 2] = root_pos[t, 2] + z_offset
```

**Problem:**
1. **Foot Body Coordinates:** The `foot_body_indices` point to `ankle_roll_link`, which is a joint, not the actual foot contact point
2. **COM Shift:** When adjusting Z, the CoM height changes, which affects IK solutions
3. **Multi-Frame Consistency:** Different frames may have different foot contact positions due to motion dynamics

**Better Approach:**
- Use foot contact points (from MJCF) rather than body CoM
- Account for foot contact geometry (sole/heel positions)
- Ensure consistency across frames by tracking foot trajectory

---

### 🟡 BUG #5: Quaternion Convention Inconsistency

**File:** Multiple files

**Issue:** Quaternion formats (wxyz vs xyzw) are converted multiple times, creating opportunities for error.

**Conversions:**
1. **GMR output (wxyz)** → `gmr_retarget_headless.py` line 132:
   ```python
   root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz → xyzw
   ```

2. **Back to wxyz in FK** → `gmr_to_protomotions.py` line 274:
   ```python
   root_rot_wxyz = quat_xyzw_to_wxyz(root_rot_xyzw[t])
   data.qpos[3:7] = root_rot_wxyz
   ```

3. **Back to xyzw for output** → `gmr_to_protomotions.py` line 290:
   ```python
   body_rot_all[t, b] = quat_wxyz_to_xyzw(body_rot_wxyz)
   ```

**Problem:**
Each conversion is a chance for error. The functions are:
```python
def quat_xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]

def quat_wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]
```

These look correct (xyzw [a,b,c,d] → wxyz [d,a,b,c] and back), but **they should be verified against scipy's convention**:

```python
>>> from scipy.spatial.transform import Rotation as R
>>> import numpy as np
>>> q_xyzw = np.array([0.0, 0.707, 0.0, 0.707])  # 90° around Y
>>> r = R.from_quat(q_xyzw)  # scipy expects xyzw!
>>> print(r.as_quat())  # confirms xyzw
```

**The functions appear correct, but add a test to be sure.**

---

### 🔴 BUG #6: `--no-fk-ground-correction` Flag Not Consistently Applied

**File:** `scripts/embodied/pipeline_motion_to_robot.py`, lines 139-142

**Code:**
```python
if args.no_fk_ground_correction:
    proto_cmd.append("--no-fk-ground-correction")
```

**But** at `gmr_to_protomotions.py` line 393-396:
```python
parser.add_argument("--fk-ground-correction", action="store_true", default=True,
                    help="Adjust root Z so feet are at ground level based on FK (default: True)")
parser.add_argument("--no-fk-ground-correction", dest="fk_ground_correction", action="store_false",
                    help="Disable FK-based ground correction")
```

**Problem:**
The default is `True` (FK correction enabled). However:
1. In `gmr_retarget_headless.py` line 125, the orchestrator calls:
   ```python
   "--no-offset-to-ground",  # FK correction handles grounding
   ```
   This passes `--no-offset-to-ground` to GMR, which disables **per-frame foot grounding in GMR**.

2. Then in `gmr_to_protomotions.py` lines 444-452, FK correction is re-applied.

**Issue:** There's a disconnect between:
- GMR's `offset_to_ground` parameter (per-frame foot grounding during IK)
- `gmr_to_protomotions.py`'s `fk_ground_correction` (post-hoc Z adjustment)

These are **two different mechanisms** and their interaction is unclear:
- GMR's `offset_to_ground=False` means feet may not be grounded when IK solves
- Then `gmr_to_protomotions.py` tries to fix this with post-hoc FK correction

**This could cause:**
- **Foot Sliding:** If IK produces ungrounded poses, post-hoc correction won't fix joint angles
- **Deformed Poses:** The joints may be solved for air-standing, then Z is adjusted without resolving IK

---

### 🟡 BUG #7: Height Scaling Sensitivity and Auto-Detection

**File:** `scripts/embodied/gmr_retarget_headless.py`, lines 82-89

**Code:**
```python
smplx_data, body_model, smplx_output, auto_human_height = load_smplx_file(
    args.smplx_file, SMPLX_FOLDER
)
actual_human_height = auto_human_height
if args.actual_human_height is not None:
    actual_human_height = args.actual_human_height
    print(f"  Human height override: {actual_human_height:.3f}m (auto-detected was: {auto_human_height:.3f}m)")
```

**Problem:**
The auto-detection uses `1.66 + 0.1*betas[0]` (from SMPL-X), but:
1. **SMPL-X betas range is typically [-3, +3]**, so heights range from **1.36 to 1.96m**
2. This assumes a specific SMPL-X training distribution
3. **User-provided motions might have very different statistics**
4. Even small errors cause large scale mismatches

**Evidence from diagnose script:**
```python
heights_to_test = [auto_height, 1.8, 2.0, 2.1, 2.2]
```

Varying height by just 0.2m causes massive changes in IK output (see outputs).

**Result:**
- If auto-height is wrong by even 0.1m, IK produces very different poses
- Knees go from `0.005 rad` (straight) to `0.5 rad` (bent 28°)
- Pelvis height changes significantly

---

### 🟡 BUG #8: Potential Issue with Rot6D Column/Row Major Convention

**File:** `scripts/embodied/motion135_to_smplx.py`, lines 26-55

**Code:**
```python
def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix.
    
    HyMotion outputs rot6d in row-major layout: [R00,R01, R10,R11, R20,R21]
    Gram-Schmidt expects column-major layout: [R00,R10,R20, R01,R11,R21]
    We reorder [0,2,4,1,3,5] to convert row-major → column-major before decoding.
    """
    # Row-major → column-major reorder
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    ...
```

**Verification:**
- Row-major layout: `[R00, R01, R10, R11, R20, R21]` (first two cols, all three rows)
- Reorder to: `[0, 2, 4, 1, 3, 5]` gives: `[R00, R10, R20, R01, R11, R21]` (all rows of first col, all rows of second col)
- This is correct column-major layout ✓

**However:**
- This reorder assumes HyMotion outputs are truly row-major
- **Need to verify against actual HyMotion output format**
- If wrong, rotations will be completely incorrect from frame 1

---

### 🟡 BUG #9: No Validation of SMPL-X Joint Consistency

**File:** `scripts/embodied/motion135_to_smplx.py`

**Issue:** After conversion, there's no check that SMPL-X joint positions match the original motion's joint positions.

**Current Code:**
```python
# Convert rot6d → rotation matrix → axis-angle
rotmat = rot6d_to_rotmat(rot6d)                   # (T, 22, 3, 3)
aa = rotmat_to_axis_angle(rotmat)                 # (T, 22, 3)

# Split root and body
root_orient = aa[:, 0, :]                         # (T, 3)
pose_body = aa[:, 1:22, :].reshape(T, -1)         # (T, 63)

print(f"root_orient shape: {root_orient.shape}")
print(f"pose_body shape: {pose_body.shape}")
```

**Missing:**
- Forward kinematics with SMPL-X to compute body positions
- Comparison with original `motion_135` positions (if provided as `positions` in NPZ)
- Verification that axis-angle representations are valid

**Could Cause:**
- Silent conversion errors where rot6d → matrix conversion is subtly wrong
- Propagation of errors through entire pipeline

---

### 🟡 BUG #10: Ground Offset Computation May Be Frame-Dependent

**File:** `scripts/embodied/gmr_retarget_headless.py`, lines 35-62

**Code:**
```python
def compute_ground_offset(retarget, smplx_data_frames):
    """Pre-scan all frames to find ground offset (lowest body Z position)."""
    offset = np.inf
    for frame_data in smplx_data_frames:
        human_data = retarget.to_numpy(frame_data)
        human_data = retarget.scale_human_data(...)
        human_data = retarget.offset_human_data(...)
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if pos[2] < offset:
                offset = pos[2]
    return offset
```

**Problem:**
- Scans all frames to find minimum Z across all bodies
- This is correct for finding the lowest point the human reaches
- **BUT:** If the motion includes crouching or lying down, this Z might be extreme
- Could cause over-correction in standing frames

**Better Approach:**
- Use statistic (e.g., 5th percentile of standing frames)
- Or specify target ground contact points (ankles, feet)
- Not all body points should contact ground

---

## Symptom-to-Bug Mapping

Based on user's reported issues:

| User Report | Likely Cause(s) | Bug # |
|-------------|-----------------|-------|
| **Foot Sliding** | Wrong ground height, incorrect foot tracking | #1, #4, #6 |
| **Ground Penetration** | FK correction reading wrong bodies | #1 |
| **Deformed Poses** | IK solved without grounding, then Z adjusted | #6 |
| **Joints at Limits** | No clamping applied | #3 |
| **Severe Quality Loss** | Height scaling auto-detection incorrect | #7 |

---

## Recommended Immediate Actions

### Priority 1: Verify FK Ground Correction Body Indices

```bash
python -c "
import mujoco
model = mujoco.MjModel.from_xml_path('ref_repo/ProtoMotions/data/robot_assets/g1/mjcf/g1_holo_compat.xml')
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjOBJ_BODY, i)
    print(f'{i}: {name}')
" | grep -i ankle
```

This will show which body indices correspond to ankle links. Compare with hardcoded `[7, 13]`.

### Priority 2: Add Comprehensive Logging

Add debug output showing:
- Root position before/after frame conversions
- Body indices used in FK correction
- Foot Z values before/after correction
- Joint limits vs actual DOF values

### Priority 3: Test With Ground Truth

Create a test motion (e.g., simple stand → walk → stand) and verify:
- Pelvis height is reasonable (0.79m for G1)
- Feet stay above ground (Z ≥ 0)
- Joints stay within limits
- No sudden jumps in root position

### Priority 4: Disable GMR's offset_to_ground and rely on post-hoc FK

Current pipeline does this at line 125:
```python
"--no-offset-to-ground",  # FK correction handles grounding
```

This is correct, but verify that `fk_ground_correction` is actually enabled and working.

---

## Testing Checklist

- [ ] Print actual G1 body indices from MJCF
- [ ] Verify foot_body_indices `[7, 13]` are correct
- [ ] Test with known good SMPL-X motion
- [ ] Check auto-height matches actual human in reference motion
- [ ] Verify no joint limits are violated
- [ ] Visualize motion in MuJoCo (use ProtoMotions viewer)
- [ ] Compare with original reference motion (if available)
- [ ] Check for frame-to-frame discontinuities

---

## Questions for Investigation

1. **Is GMR's `root_pos` actually in Y-up or Z-up?** The code assumes Y-up, but GMR might output in Z-up already
2. **What are the correct foot body indices for G1?** `[7, 13]` might be wrong
3. **Should GMR use `offset_to_ground=True` or `False`?** Current pipeline uses False, relying on post-hoc FK
4. **What's the expected pelvis height for G1 standing?** Code mentions 0.796m, but is this correct?
5. **How sensitive is IK to height scaling?** `diagnose_height_scaling.py` shows massive differences for small changes

---

## Coordinate System Reference

### SMPL-X (Y-up)
- X: Right
- Y: Up
- Z: Forward

### MuJoCo (Z-up)  
- X: Forward
- Y: Left/Lateral
- Z: Up

### Conversion (SMPL-X → MuJoCo)
```
[x, y, z]_smplx → [z, x, y]_mujoco
```

Via quaternion: `rot_offset = [0.5, -0.5, -0.5, -0.5]` (wxyz) = 120° rotation

---

## Code Quality Issues

1. **Magic numbers:** `[7, 13]` for foot indices hardcoded without documentation
2. **Missing docstrings:** Pipeline flow not clearly documented
3. **No assertions:** No validation that coordinate frames match expectations
4. **No unit tests:** Individual conversion functions not tested
5. **Verbose output:** Great for debugging, but should be behind `-v` flag

---

**Generated:** 2026-05-12  
**Analysis Depth:** Full source code review of 4 scripts, 20KB+ of Python code

