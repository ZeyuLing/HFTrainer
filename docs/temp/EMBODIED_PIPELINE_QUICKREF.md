# Embodied Pipeline - Quick Reference Guide

## 🔴 CRITICAL BUGS (Fix Immediately)

| Bug | File | Line | Symptom | Fix |
|-----|------|------|---------|-----|
| Body index offset wrong | `gmr_to_protomotions.py` | 216-220 | Foot sliding, ground penetration | Use dynamic body lookup via `mj_name2id()` |
| No joint limit clamping | `gmr_retarget_headless.py` | 119 | Joints at mechanical limits | Add `np.clip()` after IK |
| Height auto-detection too naive | `gmr_retarget_headless.py` | 82-89 | Severe pose deformation | Use calibration script |
| FK correction may use wrong frame | `gmr_to_protomotions.py` | 417-453 | Inconsistent ground heights | Verify coordinate frame before/after conversion |

## 🟡 MEDIUM PRIORITY BUGS

- Hardcoded `foot_body_indices = [7, 13]` - verify against actual G1 MJCF
- No validation of rot6d → matrix conversion
- Ground offset computed as absolute minimum (should use percentile)
- Limited logging for debugging
- No joint limit checking

## 📋 Pipeline Flow & Coordinate Systems

```
motion_135 (T, 135)
   ├─ [trans(3) + 22×rot6d(132)]
   └─ Format: row-major rot6d
      ↓
motion135_to_smplx.py
   ├─ rot6d conversion: [0,2,4,1,3,5] reorder (row-major → column-major)
   ├─ Gram-Schmidt orthogonalization
   └─ Output: SMPL-X pose in Y-up frame
      ↓
gmr_retarget_headless.py
   ├─ GMR IK retargeting (SMPL-X → G1)
   ├─ Ground offset pre-scan (all frames → minimum Z)
   ├─ Per-frame offset_to_ground=False (relies on post-hoc correction)
   └─ Output: qpos in MuJoCo Z-up frame (with rot_offset baked in)
      ├─ root_pos: (T, 3) in MuJoCo Z-up
      ├─ root_rot: (T, 4) xyzw (rot_offset applied)
      └─ dof_pos: (T, 29)
         ↓
gmr_to_protomotions.py
   ├─ Convert root_rot: remove rot_offset (quaternion)
   ├─ Convert root_pos: Y-up → Z-up (position)  ⚠️ CHECK THIS
   ├─ FK ground correction (if enabled)
   │  ├─ Load MJCF & run FK
   │  ├─ Find foot min Z
   │  └─ Adjust root_pos Z
   ├─ Resample FPS (30Hz → 50Hz)
   └─ Output: ProtoMotions cache
      ├─ dof_pos: (T', 29)
      ├─ body_pos: (T', 33, 3) in MuJoCo Z-up
      ├─ body_rot: (T', 33, 4) xyzw
      └─ Velocities: finite differences
```

## 🧪 Verification Commands

```bash
# 1. Check body indices
python -c "import mujoco; m=mujoco.MjModel.from_xml_path('ref_repo/ProtoMotions/data/robot_assets/g1/mjcf/g1_holo_compat.xml'); print([mujoco.mj_id2name(m,mujoco.mjOBJ_BODY,i) for i in range(m.nbody)])" | grep -i ankle

# 2. Run verification script
python scripts/embodied/verify_pipeline_integrity.py --mjcf ref_repo/ProtoMotions/data/robot_assets/g1/mjcf/g1_holo_compat.xml

# 3. Calibrate height
python scripts/embodied/calibrate_height_scaling.py --smplx_file /path/to/smplx.npz

# 4. Check output sanity
python -c "
import torch
c=torch.load('output.pt', weights_only=False)
print('DOF range:', c['dof_pos'].min(), c['dof_pos'].max())
print('Height range:', c['body_pos'][:,:,2].min(), c['body_pos'][:,:,2].max())
print('Frames:', c['num_frames'])
"
```

## 📐 Coordinate Systems

| System | X | Y | Z | Up |
|--------|---|---|---|----|
| SMPL-X | Right | Up | Forward | Y |
| MuJoCo | Forward | Left | Up | Z |
| Conversion | Y_smplx | Z_smplx | X_smplx | - |

**Transformation:** `rot_offset.inv()` at `[1.0, 0.0, 0.0]_smplx` → `[0.0, 1.0, 0.0]_mujoco`

## 🎯 Expected Fixes (Timeline)

| Priority | Fix | Time | Impact |
|----------|-----|------|--------|
| P0 | Dynamic body indices | 1h | Major - fixes foot sliding |
| P0 | Joint limit clamping | 2h | Major - fixes mechanical limits |
| P1 | Height calibration tool | 1h | High - fixes deformed poses |
| P1 | Comprehensive logging | 1h | Medium - debugging |
| P2 | Rot6D validation | 2h | Low - unlikely issue |
| P2 | Ground offset percentile | 1h | Low - edge case |

## 🚨 Red Flags in Output

If you see these in the output cache, something is wrong:

```
RED FLAG: dof_pos contains values > 3.14 or < -3.14
RED FLAG: body_pos Z < 0 (ground penetration)
RED FLAG: body_pos Z > 2.0 (unrealistic height)
RED FLAG: sudden jumps in root_pos between frames
RED FLAG: all dof_pos values the same (convergence failure)
```

## 🔧 Quick Debugging

```python
import pickle
import numpy as np

# Load GMR output
with open('gmr.pkl', 'rb') as f:
    data = pickle.load(f)

root_pos = data['root_pos']
dof_pos = data['dof_pos']

print(f"Root Z range: {root_pos[:,2].min():.3f} to {root_pos[:,2].max():.3f}m")
print(f"Expected: ~0.79m for G1 standing")
print(f"DOF min/max: {dof_pos.min():.3f} to {dof_pos.max():.3f}")
print(f"Expected: ±1.5 rad for typical motion")

# Check for issues
if np.any(dof_pos > 3.14):
    print("⚠️  DOF EXCEEDS LIMITS (>π)")
if np.any(dof_pos < -3.14):
    print("⚠️  DOF BELOW LIMITS (<-π)")
if np.any(root_pos[:,2] < 0):
    print("⚠️  GROUND PENETRATION DETECTED")
```

## 📞 Common Issues & Solutions

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Foot sliding | Wrong ground offset or body index | Check body indices, verify FK correction |
| Ground penetration | FK not applied or broken | Enable FK correction, verify it runs |
| Deformed poses | Wrong height or joint limits | Run calibration script, add clamping |
| No motion at all | Height scaling way off | Try different heights (1.6-1.9m range) |
| Jerky motion | FPS conversion issue | Check resampling logic |
| High joint velocities | Bad dof_vel computation | Verify finite diff formula |

## 🔗 Related Files

- **GMR source:** `ref_repo/GMR/general_motion_retargeting/`
- **ProtoMotions:** `ref_repo/ProtoMotions/`
- **MJCF:** `ref_repo/ProtoMotions/data/robot_assets/g1/mjcf/g1_holo_compat.xml`
- **SMPL-X models:** `ref_repo/GMR/assets/body_models/`

## 📖 Documentation

1. Read: `EMBODIED_PIPELINE_BUG_ANALYSIS.md` - Full analysis of all bugs
2. Read: `EMBODIED_PIPELINE_FIXES.md` - Detailed fix implementations
3. Run: `scripts/embodied/verify_pipeline_integrity.py` - Sanity checks
4. Use: `scripts/embodied/calibrate_height_scaling.py` - Height tuning

---

**Last Updated:** 2026-05-12  
**Status:** 🔴 Critical bugs identified, fixes provided
