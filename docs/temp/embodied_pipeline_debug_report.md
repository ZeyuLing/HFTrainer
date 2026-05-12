# Embodied Pipeline: HyMotion -> G1 Robot Debug Report

**Date**: 2026-05-12
**Status**: Pipeline working, 80% tracking success rate on first batch

## Pipeline Architecture

```
HyMotion eval NPZ (motion_135 format)
    ↓ motion135_to_smplx.py
SMPL-X NPZ (pose_body + root_orient + trans)
    ↓ gmr_retarget_headless.py (GMR IK solver)
GMR Robot PKL (root_pos + root_rot + dof_pos)
    ↓ gmr_to_protomotions.py (FK + resampling + velocity)
ProtoMotions cache .pt (33-body state @ 50Hz)
    ↓ test_tracker_mujoco.py (ONNX tracker)
Physical simulation in MuJoCo
```

## Key Scripts

| Script | Function |
|--------|----------|
| `scripts/embodied/pipeline_motion_to_robot.py` | End-to-end orchestrator |
| `scripts/embodied/motion135_to_smplx.py` | motion_135 → SMPL-X NPZ |
| `scripts/embodied/gmr_retarget_headless.py` | SMPL-X → Robot (GMR IK) |
| `scripts/embodied/gmr_to_protomotions.py` | GMR PKL → ProtoMotions cache |
| `scripts/embodied/diagnose_height_scaling.py` | Diagnostic tool |

## Bugs Fixed

### BUG #1: rot6d convention mismatch
- HyMotion outputs row-major rot6d, Gram-Schmidt expects column-major
- Fix: reorder `[0,2,4,1,3,5]` in `motion135_to_smplx.py`

### BUG #2: GMR pelvis quaternion frame conversion
- GMR bakes `rot_offset=[0.5,-0.5,-0.5,-0.5]` (wxyz) into pelvis quaternion
- This is a 120-deg rotation mapping Y-up→Z-up
- Fix: right-multiply by `rot_offset.inv()` in `gmr_to_protomotions.py`

### BUG #3: Y-up translation passthrough
- GMR passes SMPL-X Y-up translation without frame conversion
- Fix: apply `rot_offset.inv()` to position vectors: `[x,y,z]_Yup → [z,x,y]_Zup`

### BUG #4: Pelvis height too low → robot falls (FIXED)
- **Root cause**: GMR's height scaling (scale = base_scale × actual_height/assumed_height = 0.9 × 1.66/1.8 = 0.83) aggressively compresses the SMPL-X pelvis position, AND the IK solver produces crouched DOF angles to match the lowered targets
- **Symptom**: Pelvis at ~0.650m (vs reference 0.796m), robot falls by frame 100
- **Fix**: FK-based ground correction in `gmr_to_protomotions.py`:
  1. After coordinate conversion, run MuJoCo FK with current DOF angles
  2. Find the lowest foot Z from FK body positions
  3. Adjust root_pos Z so the lowest foot is at Z=0
  4. This decouples root height from GMR's scaling — derives it from actual kinematics

## Results

### Tracking Success Rate (10 motions, `uncond_local/E2_B`)

| Motion | Init root_h | Final root_h | Max ref err | Status |
|--------|-------------|--------------|-------------|--------|
| 00000 | 0.776 | 0.787 | 0.6492 | STABLE |
| 00001 | 0.770 | 0.780 | 0.5836 | STABLE |
| 00002 | 0.778 | 0.784 | 0.5948 | STABLE |
| 00003 | 0.780 | 0.273 | 1.1986 | FELL (aggressive motion) |
| 00004 | 0.774 | 0.786 | 0.5953 | STABLE |
| 00005 | 0.550 | 0.079 | 2.4172 | FELL (non-standing pose) |
| 00006 | 0.778 | 0.783 | 0.6991 | STABLE |
| 00007 | 0.780 | 0.790 | 0.6525 | STABLE |
| 00008 | 0.546 | 0.784 | 1.4872 | STABLE |
| 00009 | 0.777 | 0.783 | 0.7634 | STABLE |

**Success rate: 80% (8/10)**

### Failure Analysis
- **00003**: Motion starts standing but involves deep squatting (pelvis drops to 0.44m). High DOF velocities ([-30, 25] rad/s).
- **00005**: Motion starts with low pelvis (0.55m) — non-standing pose throughout.
- Both failures are likely motion quality issues, not pipeline bugs.

### Height Variant Comparison (on motion 00000)

| Variant | Scale | Pelvis Z | root_h range | Max ref err |
|---------|-------|----------|--------------|-------------|
| h=1.66 (auto) + FK | 0.83 | 0.758m | 0.776-0.787 | 0.6492 |
| h=1.8 + FK | 0.90 | 0.764m | 0.779-0.789 | 0.6146 |
| h=2.0 + FK | 1.00 | 0.763m | 0.777-0.785 | 0.6267 |
| Reference standing | - | 0.796m | 0.791-0.796 | 0.4053 |

All three height variants work with FK correction. The auto-detected height (1.66m) is used as default.

## Usage

```bash
# Single motion conversion + validation
python scripts/embodied/pipeline_motion_to_robot.py \
    --input work_dirs/.../npz/00000.npz \
    --output data/embodied_debug/robot_cache.pt \
    --validate --keep-intermediates

# Without validation (faster)
python scripts/embodied/pipeline_motion_to_robot.py \
    --input work_dirs/.../npz/00000.npz \
    --output /tmp/robot_cache.pt

# Custom height override
python scripts/embodied/pipeline_motion_to_robot.py \
    --input work_dirs/.../npz/00000.npz \
    --output /tmp/robot_cache.pt \
    --actual-human-height 1.8
```

## Next Steps

1. **Batch evaluation**: Run on more motions (50+) to get stable success rate statistics
2. **Filter infeasible motions**: Non-standing poses and extreme crouching cannot be tracked — need motion filtering
3. **Quality metrics**: Beyond tracking success, measure foot skating, ground penetration, DOF tracking error
4. **Training integration**: Use successful robot caches as training data for motion policy
