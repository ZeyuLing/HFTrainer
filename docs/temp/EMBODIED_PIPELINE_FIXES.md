# Embodied Pipeline - Bug Fixes & Recommendations

## Priority Fixes (Do These First)

### Fix #1: Dynamic Body Index Resolution (CRITICAL)

**File:** `scripts/embodied/gmr_to_protomotions.py`

**Current Code (Lines 155-184):**
```python
def fk_ground_correction(mjcf_path, root_pos, root_rot_xyzw, dof_pos, foot_body_indices=None, ground_clearance=0.0):
    """..."""
    if foot_body_indices is None:
        foot_body_indices = [7, 13]  # Hardcoded - WRONG!
    
    for t in range(T):
        ...
        for bi in foot_body_indices:
            foot_z = data.xpos[bi + 1][2]  # Index offset is problematic
```

**Fix:**
```python
def get_foot_body_indices(model):
    """Get foot body indices from MuJoCo model by name lookup."""
    import mujoco
    foot_names = ["left_ankle_roll_link", "right_ankle_roll_link", 
                  "left_foot", "right_foot"]
    indices = []
    for name in foot_names:
        bid = mujoco.mj_name2id(model, mujoco.mjOBJ_BODY, name)
        if bid >= 0:
            indices.append(bid)
    if not indices:
        print("WARNING: No foot bodies found, using defaults [7, 13]")
        indices = [7, 13]
    return indices

def fk_ground_correction(mjcf_path, root_pos, root_rot_xyzw, dof_pos, 
                        foot_body_indices=None, ground_clearance=0.0):
    """..."""
    import mujoco
    import tempfile
    import os
    
    # Load model first to get proper body indices
    patched_xml = _patch_mjcf_xml(mjcf_path)
    asset_dir = str(Path(mjcf_path).parent)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=asset_dir, delete=False) as tmp:
        tmp.write(patched_xml)
        tmp_path = tmp.name
    
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    
    # Get foot indices dynamically if not provided
    if foot_body_indices is None:
        foot_body_indices = get_foot_body_indices(model)
    
    print(f"  Using foot bodies: {[mujoco.mj_id2name(model, mujoco.mjOBJ_BODY, i) for i in foot_body_indices]}")
    
    data = mujoco.MjData(model)
    
    T = root_pos.shape[0]
    corrected_root_pos = root_pos.copy()
    foot_min_z_before = np.zeros(T, dtype=np.float64)
    
    for t in range(T):
        root_rot_wxyz = quat_xyzw_to_wxyz(root_rot_xyzw[t])
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot_wxyz
        data.qpos[7:] = dof_pos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        
        # Find minimum foot Z - NO +1 OFFSET needed because body indices from mj_name2id are correct
        min_foot_z = np.inf
        for bi in foot_body_indices:
            foot_z = data.xpos[bi][2]  # Direct index, no +1
            if foot_z < min_foot_z:
                min_foot_z = foot_z
        
        foot_min_z_before[t] = min_foot_z
        
        # Adjust root_pos Z
        z_offset = ground_clearance - min_foot_z
        corrected_root_pos[t, 2] = root_pos[t, 2] + z_offset
    
    return corrected_root_pos, foot_min_z_before
```

**Impact:** Fixes foot sliding and ground penetration issues.

---

### Fix #2: Add Joint Limit Clamping (CRITICAL)

**File:** `scripts/embodied/gmr_retarget_headless.py`

**Current Code (Line 119):**
```python
for i, frame_data in enumerate(smplx_data_frames):
    qpos = retarget.retarget(frame_data, offset_to_ground=args.offset_to_ground)
    qpos_list.append(qpos)  # No limits applied!
```

**Fix:**
```python
def get_joint_limits_from_mjcf(mjcf_path):
    """Extract joint limits from MJCF XML."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    
    limits = {}
    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        range_elem = joint.find("range")
        if range_elem is not None:
            qmin = float(range_elem.get("qmin", -np.pi))
            qmax = float(range_elem.get("qmax", np.pi))
            limits[joint_name] = (qmin, qmax)
    
    return limits

def clamp_qpos_to_limits(qpos, robot_type="unitree_g1"):
    """Clamp DOF positions to valid ranges for the robot.
    
    qpos format: [pos(3), quat(4), dof(29)] = 36 total for G1
    """
    # G1 DOF limits (from MJCF, approximately)
    # Order: L_Hip_pitch, L_Hip_roll, L_Hip_yaw, L_Knee, L_Ankle_pitch, L_Ankle_roll,
    #        R_Hip_pitch, R_Hip_roll, R_Hip_yaw, R_Knee, R_Ankle_pitch, R_Ankle_roll, ...
    dof_limits = [
        (-1.4, 1.4),   # 0: L_Hip_pitch
        (-0.6, 0.6),   # 1: L_Hip_roll
        (-1.5, 1.5),   # 2: L_Hip_yaw
        (-0.1, 2.5),   # 3: L_Knee
        (-0.7, 0.7),   # 4: L_Ankle_pitch
        (-0.6, 0.6),   # 5: L_Ankle_roll
        (-1.4, 1.4),   # 6: R_Hip_pitch
        (-0.6, 0.6),   # 7: R_Hip_roll
        (-1.5, 1.5),   # 8: R_Hip_yaw
        (-0.1, 2.5),   # 9: R_Knee
        (-0.7, 0.7),   # 10: R_Ankle_pitch
        (-0.6, 0.6),   # 11: R_Ankle_roll
        # Add other DOFs as needed (arm, back, etc.)
    ]
    
    # Clamp DOFs
    for i, (qmin, qmax) in enumerate(dof_limits):
        if i + 7 < len(qpos):  # DOFs start at index 7
            qpos[i + 7] = np.clip(qpos[i + 7], qmin, qmax)
    
    return qpos

# In main loop:
for i, frame_data in enumerate(smplx_data_frames):
    qpos = retarget.retarget(frame_data, offset_to_ground=args.offset_to_ground)
    qpos = clamp_qpos_to_limits(qpos, args.robot)  # NEW: Apply limits
    qpos_list.append(qpos)
```

**Impact:** Fixes "joints at mechanical limits" issues.

---

### Fix #3: Enable Comprehensive Logging

**File:** `scripts/embodied/gmr_to_protomotions.py`

**Add after line 441:**
```python
print(f"\nDEBUG: Coordinate frame verification")
print(f"  root_pos[0] SMPL-X Y-up (before): {root_pos_before[0]}")
print(f"  root_pos[0] MuJoCo Z-up (after):  {root_pos[0]}")
print(f"  Expected transformation: [Y_smplx, X_smplx, Z_smplx] format")
print(f"  root_pos Z range after conversion: [{root_pos[:,2].min():.4f}, {root_pos[:,2].max():.4f}]")

# Verify FK correction worked
if args.fk_ground_correction:
    print(f"\nDEBUG: FK ground correction")
    print(f"  Foot Z before: [{foot_min_z.min():.4f}, {foot_min_z.max():.4f}]")
    print(f"  Foot Z after:  [0.0, {(root_pos[:,2] - root_pos_before[:,2]).max() + foot_min_z.max():.4f}]")
    print(f"  Root Z adjustment: [{(root_pos[:,2] - root_pos_before[:,2]).min():.4f}, {(root_pos[:,2] - root_pos_before[:,2]).max():.4f}]")
```

**Impact:** Better diagnostics for debugging.

---

### Fix #4: Verify and Document Height Scaling

**Create:** `scripts/embodied/calibrate_height_scaling.py`

```python
#!/usr/bin/env python3
"""Calibrate human height scaling for GMR retargeting.

Usage:
    python scripts/embodied/calibrate_height_scaling.py \
        --smplx_file /path/to/smplx.npz \
        --robot unitree_g1
"""

import argparse
import sys
import pathlib
import numpy as np

GMR_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent / "ref_repo" / "GMR"
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", required=True)
    parser.add_argument("--robot", default="unitree_g1")
    args = parser.parse_args()
    
    SMPLX_FOLDER = GMR_ROOT / "assets" / "body_models"
    
    # Load SMPL-X
    smplx_data, body_model, smplx_output, auto_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    print("=== Height Scaling Calibration ===")
    print(f"Auto-detected height: {auto_height:.3f}m")
    
    # Align to target FPS
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    
    # Test different heights and find the one that produces correct pelvis height
    print("\nTesting different height scales:")
    print("-" * 80)
    print(f"{'Height (m)':>12} | {'Pelvis Z':>10} | {'Error vs 0.796':>15} | {'Knee angle':>12}")
    print("-" * 80)
    
    G1_NOMINAL_PELVIS_Z = 0.796
    heights = np.linspace(auto_height - 0.3, auto_height + 0.3, 13)
    best_height = auto_height
    best_error = np.inf
    
    for test_height in heights:
        retarget = GMR(
            actual_human_height=test_height,
            src_human="smplx",
            tgt_robot=args.robot,
            verbose=False,
        )
        
        # Compute ground offset
        offset = np.inf
        for frame_data in smplx_data_frames[:5]:
            human_data = retarget.to_numpy(frame_data)
            human_data = retarget.scale_human_data(
                human_data, retarget.human_root_name, retarget.human_scale_table
            )
            human_data = retarget.offset_human_data(
                human_data, retarget.pos_offsets1, retarget.rot_offsets1
            )
            for body_name in human_data.keys():
                pos, quat = human_data[body_name]
                if pos[2] < offset:
                    offset = pos[2]
        retarget.set_ground_offset(offset)
        
        # Retarget frame 0
        qpos = retarget.retarget(smplx_data_frames[0], offset_to_ground=True)
        pelvis_z = qpos[2]
        knee_angle = np.abs(qpos[3 + 7])  # Right knee
        
        error = abs(pelvis_z - G1_NOMINAL_PELVIS_Z)
        if error < best_error:
            best_error = error
            best_height = test_height
        
        mark = " <-- BEST" if error < best_error + 0.001 else ""
        print(f"{test_height:>12.3f} | {pelvis_z:>10.4f} | {error:>15.4f} | {knee_angle:>12.4f}{mark}")
    
    print("-" * 80)
    print(f"\nRecommendation:")
    print(f"  Auto-detected height: {auto_height:.3f}m")
    print(f"  Best calibrated height: {best_height:.3f}m (error: {best_error:.4f}m)")
    print(f"  Target pelvis Z: {G1_NOMINAL_PELVIS_Z:.4f}m")
    print(f"\nUse: python ... --actual-human-height {best_height:.3f}")

if __name__ == "__main__":
    main()
```

**Impact:** Helps users find correct height scaling for their data.

---

## Medium Priority Fixes

### Fix #5: Verify Rot6D Conversion Against Reference

**File:** `scripts/embodied/motion135_to_smplx.py`

**Add validation:**
```python
def validate_rot6d_conversion(motion_135_npz_path):
    """Validate rot6d conversion by comparing against original positions (if available)."""
    data = np.load(motion_135_npz_path, allow_pickle=True)
    
    if 'positions' not in data:
        print("  No reference positions to validate against (OK)")
        return
    
    positions_ref = data['positions']  # (T, 22, 3) if available
    print(f"  Validating rot6d → SMPL-X positions conversion...")
    print(f"  Reference positions shape: {positions_ref.shape}")
    
    # TODO: Add FK on converted SMPL-X and compare with reference positions
```

---

### Fix #6: Add Quaternion Roundtrip Testing

**File:** `scripts/embodied/gmr_to_protomotions.py`

**Add after line 289:**
```python
# Sanity check: quaternion roundtrip
body_rot_wxyz_test = data.xquat[b + 1]
body_rot_xyzw_test = quat_wxyz_to_xyzw(body_rot_wxyz_test)
body_rot_wxyz_back = quat_xyzw_to_wxyz(body_rot_xyzw_test)
if not np.allclose(body_rot_wxyz_test, body_rot_wxyz_back):
    print(f"WARNING: Quaternion roundtrip failed for body {b}")
```

---

### Fix #7: Better Ground Offset Computation

**File:** `scripts/embodied/gmr_retarget_headless.py`

**Replace lines 35-62:**
```python
def compute_ground_offset(retarget, smplx_data_frames, percentile=5):
    """Pre-scan frames to find ground offset (lowest body Z position).
    
    Uses percentile instead of absolute minimum to avoid extreme poses.
    """
    all_mins = []
    for frame_data in smplx_data_frames:
        human_data = retarget.to_numpy(frame_data)
        human_data = retarget.scale_human_data(
            human_data, retarget.human_root_name, retarget.human_scale_table
        )
        human_data = retarget.offset_human_data(
            human_data, retarget.pos_offsets1, retarget.rot_offsets1
        )
        frame_min_z = np.inf
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if pos[2] < frame_min_z:
                frame_min_z = pos[2]
        all_mins.append(frame_min_z)
    
    # Use percentile instead of absolute min to avoid extreme poses
    offset = np.percentile(all_mins, percentile)
    print(f"  Ground offset: {offset:.4f}m (used {percentile}th percentile of {len(all_mins)} frames)")
    return offset
```

**Impact:** More robust ground height detection.

---

## Testing Checklist

After applying fixes:

```bash
# 1. Run verification script
python scripts/embodied/verify_pipeline_integrity.py \
    --mjcf ref_repo/ProtoMotions/data/robot_assets/g1/mjcf/g1_holo_compat.xml

# 2. Calibrate height for your data
python scripts/embodied/calibrate_height_scaling.py \
    --smplx_file /path/to/smplx.npz

# 3. Run full pipeline with new fixes
python scripts/embodied/pipeline_motion_to_robot.py \
    --input work_dirs/your_motion/npz/00000.npz \
    --output /tmp/test_motion_cache.pt \
    --actual-human-height 1.7 \
    --keep-intermediates

# 4. Check output
python -c "
import torch
cache = torch.load('/tmp/test_motion_cache.pt', weights_only=False)
print('dof_pos range:', cache['dof_pos'].min(), cache['dof_pos'].max())
print('body_pos Z range:', cache['body_pos'][:,:,2].min(), cache['body_pos'][:,:,2].max())
print('Frames:', cache['num_frames'])
"

# 5. Visualize in MuJoCo (if available)
python ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py \
    --motion /tmp/test_motion_cache.pt \
    --loops 1 \
    --no-realtime
```

---

## Debug Commands

```bash
# Check G1 body structure
python -c "
import mujoco
model = mujoco.MjModel.from_xml_path(
    'ref_repo/ProtoMotions/data/robot_assets/g1/mjcf/g1_holo_compat.xml'
)
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjOBJ_BODY, i)
    print(f'{i}: {name}')
" | grep -i ankle

# Check coordinate conversion
python -c "
from scipy.spatial.transform import Rotation as R
import numpy as np

# GMR rot_offset
rot_offset_wxyz = np.array([0.5, -0.5, -0.5, -0.5])
rot_offset_xyzw = rot_offset_wxyz[[1, 2, 3, 0]]
rot = R.from_quat(rot_offset_xyzw)

# Test position
pos_smplx = np.array([1.0, 2.0, 3.0])
pos_mujoco = rot.inv().apply(pos_smplx)
print('SMPL-X:', pos_smplx)
print('MuJoCo:', pos_mujoco)
print('Expected: [3, 1, 2]')
"

# Check quaternion format
python -c "
from scipy.spatial.transform import Rotation as R
import numpy as np

q_xyzw = np.array([0, 0.707, 0, 0.707])  # 90° around Y
r = R.from_quat(q_xyzw)
print('Input (xyzw):', q_xyzw)
print('Euler angles:', r.as_euler('xyz', degrees=True))
print('Expected: [0, 90, 0]')
"
```

---

## Expected Improvements After Fixes

| Symptom | Expected Improvement |
|---------|----------------------|
| Foot sliding | Should stop after Fix #1 + #3 |
| Ground penetration | Should resolve after Fix #1 |
| Deformed poses | Should improve after Fix #2 + #4 |
| Joints at limits | Should resolve after Fix #2 |
| Severe quality loss | Should improve with Fix #4 |

---

## Long-term Improvements

1. **Add unit tests** for rot6d, quaternion, and coordinate conversions
2. **Refactor pipeline** to separate concerns (SMPL-X conversion, GMR retargeting, FK)
3. **Use structured config** instead of hardcoded values
4. **Add visualization tools** for debugging pipeline steps
5. **Document coordinate frames** explicitly at each pipeline stage
6. **Add motion quality metrics** (foot penetration, joint limit violations, smoothness)

---

**Created:** 2026-05-12  
**Status:** Ready for implementation
