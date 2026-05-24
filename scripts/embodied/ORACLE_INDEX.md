# Physics Correction Oracle — Complete Index

**Last Updated:** 2026-05-18  
**Analysis Completeness:** 100% (all functions, all metrics, all flow)

---

## 📋 Quick Navigation

- **For Quick Reference:** See `oracle_quick_ref.txt`
- **For Detailed Analysis:** See `physics_oracle_report.md`
- **For Implementation:** Use functions from `run_smpl_physics_sim.py` and `motion135_to_smplx.py`

---

## 🎯 The Physics Correction Oracle

A MuJoCo-based PD-tracking physics simulation system that takes HyMotion's `motion_135` format (T, 135) and produces physics-corrected motion with:

- ✅ **Ground contact enforcement** (foot penetration removed)
- ✅ **Reduced foot sliding** (PD tracking + contact constraints)
- ✅ **Jitter removal** (post-simulation smoothing)
- ✅ **Quality metrics** (tracking error, drift, stability)

**Input:** `motion_135` NPZ → (T, 135) [transl(3) + 22×rot6d(6)], Y-up, HyMotion row-major  
**Output:** Physics-corrected trajectory, JSON visualization + stats

---

## 📂 File Locations

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/embodied/run_smpl_physics_sim.py` | 1172 | Main oracle implementation |
| `scripts/embodied/motion135_to_smplx.py` | 130 | rot6d utilities (shared) |
| `scripts/embodied/physics_oracle_report.md` | ~800 | This analysis (full details) |
| `scripts/embodied/oracle_quick_ref.txt` | ~400 | Quick reference card |

---

## 🔑 Core Functions (All Listed Below)

### 1. **Motion Decoding** — `motion_135` → SMPL 72-dim
- **Function:** `decode_motion_135(npz_path: str)`
- **Location:** `run_smpl_physics_sim.py:193–227`
- **Input:** NPZ path
- **Output:** (T, 72) axis-angle Y-up, (T, 3) transl, fps
- **Key Logic:** Split motion_135 → rot6d → rotmat → AA

### 2. **Rot6D Decoding** — rot6d → Rotation Matrix
- **Function:** `rot6d_to_rotmat(rot6d: np.ndarray)`
- **Location:** `motion135_to_smplx.py:26–67` (or `run_smpl_physics_sim.py:173–190`)
- **Input:** (..., 6) HyMotion row-major rot6d
- **Output:** (..., 3, 3) rotation matrix
- **Key Logic:** Reorder [0,2,4,1,3,5] for Gram-Schmidt orthogonalization

### 3. **Coordinate Transform** — Y-up ↔ Z-up
- **Functions:** 
  - `yup_to_zup(smpl_pose, transl)` — SMPL Y-up → MuJoCo Z-up
  - `zup_to_yup(smpl_pose, transl)` — MuJoCo Z-up → SMPL Y-up
- **Location:** `run_smpl_physics_sim.py:245–286` and `289–314`
- **Transform:** Cyclic permutation [x,y,z]_yup → [z,x,y]_zup
- **Critical:** Apply to ALL 24 joints (root + body local frames)

### 4. **Motion135 → QPOS** — SMPL axis-angle → MuJoCo qpos
- **Function:** `smpl_to_qpos(smpl_pose, transl, body_pos_1, model=None)`
- **Location:** `run_smpl_physics_sim.py:321–426`
- **Input:** (T, 72) axis-angle Z-up, (T, 3) transl, body offset
- **Output:** (T, 76) qpos = [root_trans(3), root_quat_wxyz(4), body_euler(69)]
- **Key Logic:** 
  - Root: AA → quat_wxyz
  - Body: AA → intrinsic XYZ Euler, reorder SMPL → MuJoCo
  - Joint limits: Guard axes centered, main axes clamped

### 5. **QPOS → SMPL** — MuJoCo qpos → SMPL axis-angle (Inverse)
- **Function:** `qpos_to_smpl(qpos, body_pos_1)`
- **Location:** `run_smpl_physics_sim.py:429–464`
- **Input:** (T, 76) qpos
- **Output:** (T, 72) axis-angle Z-up, (T, 3) transl
- **Key Logic:** Inverse of smpl_to_qpos()

### 6. **Physics Model Setup** — Configure MuJoCo for PD Tracking
- **Function:** `load_mujoco_model(xml_path: str)`
- **Location:** `run_smpl_physics_sim.py:525–606`
- **Input:** Path to smpl_humanoid.xml
- **Output:** model, data (configured for PD control)
- **Key Overrides:**
  - dof_damping: 80 → 0 (CRITICAL: was overdamping, ζ=20.6!)
  - dof_armature[6:]: 0.02 → 0.1 (critical damping, ζ=1.0)
  - Per-actuator PD gains: 69 actuators (23 bodies × 3 DOF)

### 7. **Ground Contact Calibration** — Foot z offset
- **Function:** `compute_ground_offset(model, data, ref_qpos)`
- **Location:** `run_smpl_physics_sim.py:471–522`
- **Input:** MuJoCo model, data, ref_qpos (T, 76)
- **Output:** ground_offset (float) to subtract from qpos[:, 2]
- **Key Logic:** Find min foot z across all frames, apply to ensure z=0 ground contact

### 8. **Physics Simulation Loop** — Main oracle (PD-tracking + gravity + contact)
- **Function:** `run_physics_sim(model, data, ref_qpos, fps=30)`
- **Location:** `run_smpl_physics_sim.py:609–717`
- **Input:** MuJoCo model/data, (T, 76) ref_qpos
- **Output:** (T', 76) sim_qpos [T' ≤ T], stats dict
- **Algorithm:**
  ```
  For each frame t:
    1. Set root position/quat kinematic (ref_qpos[t, :7])
    2. Compute root velocity from FD (smooth interpolation)
    3. Set body PD targets: ctrl[:] = ref_qpos[t, 7:69]
    4. Step MuJoCo physics (decimation × sub-steps)
    5. Record simulated qpos
    6. Check fall (root_h < 0.15m or NaN)
  ```
- **PD Force:** τ = kp(ctrl - qpos) - kd(qvel)
- **Stats Computed:**
  - joint_tracking_error_rad
  - root_position_drift_m
  - min_root_height_m
  - fall_frame

### 9. **Post-Simulation Smoothing** — Remove PD oscillation
- **Function:** `smooth_simulated_qpos(sim_qpos, ref_qpos, fps=30, window_ms=333.0, blend_alpha=0.5)`
- **Location:** `run_smpl_physics_sim.py:720–795`
- **Input:** (T, 76) simulated qpos, reference qpos
- **Output:** (T, 76) smoothed qpos
- **Key Logic:**
  - Savitzky-Golay filter (333ms window = 10 frames @ 30fps)
  - Blend: α × smoothed_sim + (1-α) × ref
  - Root stays kinematic (not smoothed)
  - Removes 5–10 Hz PD oscillation

### 10. **Euler→AA Jitter Removal** — Smooth conversion artifacts
- **Function:** `smooth_smpl_poses(smpl_pose, fps=30, window_ms=333.0)`
- **Location:** `run_smpl_physics_sim.py:802–890`
- **Input:** (T, 72) SMPL axis-angle Z-up
- **Output:** (T, 72) smoothed axis-angle
- **Key Logic:**
  - Quaternion-space smoothing (AA → quat → savgol → AA)
  - Adaptive windowing (wide window for > 60° rotation joints)
  - Two-pass smoothing for convergence

### 11. **Main Pipeline** — Full end-to-end processing
- **Function:** `process_single_motion(npz_path, xml_path, output_dir, stats_dir=None, fps=30)`
- **Location:** `run_smpl_physics_sim.py:950–1036`
- **Flow:** Decode → YupToZup → SmplToQpos → ComputeGroundOffset → RunPhysics → SmoothQpos → QposToSmpl → SmoothPoses → ZupToYup → ExportJSON
- **Output:** {stem}.json (visualization) + {stem}.json (stats)

---

## ⚙️ Key Configuration Parameters

### PD Gains (Per Body)
```python
# Lines 147–156 of run_smpl_physics_sim.py
L_Hip / R_Hip:        kp=1000, kd=20
L_Knee / R_Knee:      kp=1000, kd=20
L_Ankle / R_Ankle:    kp=800,  kd=18
L_Toe / R_Toe:        kp=400,  kd=13
Torso / Spine / Chest: kp=2000, kd=28
Neck / Head:          kp=200,  kd=9
Shoulders:            kp=800,  kd=18
Elbows:               kp=600,  kd=16
Wrists / Hands:       kp=200,  kd=9
```

**Tracking characteristics:**
- Time constant: τ = kd/kp ≈ 0.02s = 0.6 frames @ 30fps
- Critical damping: ζ = kd/(2√(kp*armature)) = 1.0 (with armature=0.1)

### Coordinate Transform Matrices
```python
# Lines 239–242
_YUP_TO_ZUP = [[0,0,1], [1,0,0], [0,1,0]]
_ZUP_TO_YUP = [[0,1,0], [0,0,1], [1,0,0]]
```

### Joint Reordering
```python
# Lines 133–136
SMPL_2_MUJOCO = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 14, 12, 15, 17, 19, 21, 13, 16, 18, 20, 22]
MUJOCO_2_SMPL = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 13, 18, 12, 14, 19, 15, 20, 16, 21, 17, 22]
```

### Fall Detection
```python
# Line 159
FALL_HEIGHT_THRESHOLD = 0.15  # meters
```

---

## 📊 Physics Quality Metrics

Returned in `stats` dict from `run_physics_sim()`:

| Metric | Type | Meaning |
|--------|------|---------|
| `completed` | bool | True if no fall detected |
| `total_frames` | int | T (input trajectory length) |
| `simulated_frames` | int | T' (output length after fall detection) |
| `fall_frame` | int\|None | Frame index where fall occurred |
| `joint_tracking_error_rad` | float | mean(\|sim_qpos[:, 7:] - ref_qpos[:, 7:]\|) |
| `root_position_drift_m` | float | \|final_root_pos - ref_final_pos\| |
| `min_root_height_m` | float | min(root_h) over all frames |
| `ground_offset_m` | float | Vertical shift applied for foot contact |
| `fps` | int | Control frame rate |
| `decimation` | int | MuJoCo sub-steps per control frame |

---

## 🎓 How to Use (API)

```python
from scripts.embodied.run_smpl_physics_sim import process_single_motion

# Single file
stats = process_single_motion(
    npz_path="/path/to/motion_135.npz",
    xml_path="/path/to/smpl_humanoid.xml",
    output_dir="/output/path",
    stats_dir="/stats/path",
    fps=30  # optional, inferred from NPZ if not provided
)

print(f"Completed: {stats['completed']}")
print(f"Tracking error: {stats['joint_tracking_error_rad']:.4f} rad")
print(f"Root drift: {stats['root_position_drift_m']:.4f} m")

# Outputs:
#  - {stem}.json                    → web visualization JSON
#  - {stem}.json (in stats_dir)     → simulation statistics
```

---

## 🔍 Critical Implementation Details

### Rot6D Decoding (HyMotion Format)

HyMotion stores rot6d in **row-major** layout, but Gram-Schmidt expects **column-major**:
- Row-major: [R00, R01, R10, R11, R20, R21]
- Column-major: [R00, R10, R20, R01, R11, R21]
- Reorder: [0, 2, 4, 1, 3, 5]

Then apply Gram-Schmidt:
1. b1 = normalize(a1)
2. b2 = normalize(a2 - dot(b1, a2) * b1)
3. b3 = cross(b1, b2)
4. Stack [b1, b2, b3] → (3, 3) rotation matrix

### Coordinate System Mismatch Fix

**Problem:** SMPL knee flexion is around local X axis, but without coordinate transform, it maps to MuJoCo X-joint (forward ±5.6°), not Y-joint (lateral flexion).

**Solution:** Apply cyclic permutation to ALL joint axis-angles (root + body):
- Root orientation is in global frame (Y-up → Z-up)
- Body joint axes are in local frames, which in T-pose align with global
- Same transform needed for both

**Why both:** Not just rotation matrices, but the axes themselves rotate.

### Joint Limit Handling

Problem: Euler angle decomposition of large rotations (e.g., deep knee bends) spreads rotation onto "guard" axes (narrow-limit joints like ±5.6°).

Solution: Two-tier approach:
1. **Guard axes** (range < 15°): Set PD target to center of range (usually 0)
   - Loses at most ±5.6° on minor axis (visually imperceptible)
   - Eliminates chatter entirely
2. **Main axes** (range ≥ 15°): Clamp to joint limits as before

### Root Tracking with Smooth Interpolation

The root is a free joint (not actuated). Instead of direct PD control:
1. **Position:** Reset to reference each control frame
2. **Velocity:** Compute from finite differences of reference trajectory
   - Linear: (pos_next - pos_cur) / dt
   - Angular: as_rotvec(inv(R_cur) * R_next) / dt
3. **Effect:** Physics sub-steps interpolate smoothly (not teleport)

### Ground Offset Computation

Motion data may be generated for a different body height than the MuJoCo model. To ensure foot contact:
1. Scan 30 evenly-spaced frames (or all if T ≤ 30)
2. Find minimum foot z across all frames: min(xpos[L_Toe, z], R_Toe, z, L_Ankle, z, R_Ankle, z])
3. Apply: ref_qpos[:, 2] -= ground_offset
4. After simulation: undo by adding back

---

## 🧪 Testing & Validation

See test files in `scripts/embodied/`:
- `test_pd_standing.py` — Basic PD tracking validation
- `test_mujoco_euler.py` — Euler angle conventions
- `test_root_rotation_fix.py` — Coordinate transform verification
- `diagnose_oscillation.py` — PD gain tuning

---

## 📚 Related Documentation

- **HyMotion Format:** See `FORMAT_SPECIFICATION.md`
- **Verification Report:** See `VERIFICATION_SUMMARY.txt`
- **Detailed Code Analysis:** See `detailed_code_analysis.md`

---

## 💡 Key Takeaways

1. **Full pipeline:** motion_135 → SMPL → qpos → physics → qpos → SMPL → JSON
2. **Coordinate awareness:** Y-up ↔ Z-up transforms ALL joints (root + body)
3. **PD control:** Critically damped, body joints only; root tracked kinematically
4. **Quality metrics:** Tracking error + drift + fall detection + ground contact
5. **Two-stage smoothing:** Post-sim smoothing (remove PD oscillation) + AA smoothing (remove Euler jitter)

---

## 📞 Quick Debug Checklist

- [ ] Input motion_135 format correct? (T, 135 = transl(3) + 22×rot6d(6))
- [ ] rot6d reordering applied? ([0, 2, 4, 1, 3, 5])
- [ ] Coordinate transform applied to ALL joints? (not just root)
- [ ] Joint limits enforced in smpl_to_qpos? (guard axes centered, main axes clamped)
- [ ] Ground offset computed? (min foot z across all frames)
- [ ] Fall detection working? (root_h < 0.15m or NaN check)
- [ ] Smoothing applied both to qpos and smpl_pose? (different stages)

---

**End of Index**
