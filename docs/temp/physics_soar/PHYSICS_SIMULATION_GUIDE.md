# Physics Simulation Pipeline Guide

**Status:** Complete Analysis  
**Date:** 2026-05-18  
**Scope:** Motion135 → Physics-Corrected SMPL Motion  
**Key File:** `scripts/embodied/run_smpl_physics_sim.py` (1100+ lines)

---

## Executive Summary

This pipeline converts HyMotion's `motion_135` format (22 joints, 6D rotation) into physically plausible SMPL motion by running MuJoCo physics simulation with PD tracking control. The process corrects:

- ❌ Foot sliding and interpenetration
- ❌ Height inconsistencies (floating above ground)
- ❌ Physically impossible joint configurations
- ✅ Maintains kinematic fidelity of original motion
- ✅ Enforces natural ground contact via physics

**Complete Pipeline:**
```
motion_135 NPZ (T, 135)
  ↓ [1] Decode rot6d → axis-angle
  ↓ [2] Y-up → Z-up coordinate transform
  ↓ [3] SMPL 72D → MuJoCo 76D (qpos)
  ↓ [4] Compute ground offset (feet touch ground)
  ↓ [5] PD-tracking physics simulation
  ↓ [6] Post-sim smoothing (remove PD jitter)
  ↓ [7] Z-up → Y-up reverse transform
  ↓ [8] Export SMPL mesh JSON (optional)
  ↓
SMPL-corrected motion (T, 72) + stats
```

---

## Part 1: Motion Format Conversions

### 1.1 motion_135 Format (HyMotion Internal)

**Structure:** `(T, 135)` array
- **Indices 0-2:** Translation (3D, Y-up world space)
- **Indices 3-134:** 22 joints × 6D rotations (132 values)
  - Root (pelvis): indices 3-8
  - 21 body joints: indices 9-134

**6D Rotation Representation:**
- Uses **row-major layout** (per HyMotion M2M convention): `[R00, R01, R10, R11, R20, R21]`
- Must be reordered to column-major `[R00, R10, R20, R01, R11, R21]` before Gram-Schmidt decoding
- Reorder indices: `[0, 2, 4, 1, 3, 5]`

**Key Function:** `decode_motion_135()` at line 193

```python
def decode_motion_135(npz_path: str):
    """Extract motion_135 from NPZ and decode rot6d."""
    data = np.load(npz_path, allow_pickle=True)
    motion = data['motion_135']  # (T, 135)
    
    # Extract components
    transl = motion[:, :3]                      # (T, 3)
    rot6d = motion[:, 3:].reshape(T, 22, 6)    # (T, 22, 6)
    
    # Decode: rot6d → rotation matrix → axis-angle
    rotmat = rot6d_to_rotmat(rot6d)             # (T, 22, 3, 3)
    axis_angle = rotmat_to_axis_angle(rotmat)   # (T, 22, 3)
    
    # Concatenate: [root_aa(3), body_aa(66)] → (T, 72)
    smpl_pose = np.concatenate([
        axis_angle[:, 0],                       # root
        axis_angle[:, 1:].reshape(T, 66)        # body
    ], axis=1)
    
    return smpl_pose, transl, fps  # (T, 72), (T, 3), int
```

### 1.2 rot6d_to_rotmat: The 6D Rotation Decoder

**Algorithm:** Gram-Schmidt orthogonalization (lines 173-191)

```python
def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) → (..., 3, 3) rotation matrix via Gram-Schmidt."""
    
    # Step 1: Reorder row-major → column-major
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]  # First 3D vector
    a2 = rot6d[..., 3:6] # Second 3D vector
    
    # Step 2: Normalize first column
    b1 = a1 / (||a1|| + 1e-8)
    
    # Step 3: Gram-Schmidt orthogonalize second column
    dot = sum(b1 * a2)
    b2 = a2 - dot * b1
    b2 = b2 / (||b2|| + 1e-8)
    
    # Step 4: Cross product for third column
    b3 = b1 × b2
    
    # Result: R = [b1, b2, b3]
    return stack([b1, b2, b3], axis=-1)
```

**Why Gram-Schmidt?**
- Ensures orthonormality (a true rotation matrix)
- Handles numerical errors in compact 6D representation
- Stable and well-conditioned for backprop (if needed)

### 1.3 Coordinate System Transforms

**Motion Data Coordinate System:** Y-up (standard for motion capture)
**MuJoCo Coordinate System:** Z-up (standard for physics engines)

#### yup_to_zup (lines 245-287)

Converts Y-up SMPL motion to Z-up for MuJoCo:

```python
def yup_to_zup(smpl_pose: np.ndarray, transl: np.ndarray):
    """Y-up (motion capture) → Z-up (physics engine).
    
    Rotation matrix: Rotate around X-axis by -90°
    [1   0    0  ]
    [0 cos -sin]  where cos=0, sin=1 (at -90°)
    [0  sin  cos]
    
    Result:
    X_new = X_old
    Y_new = Z_old  (up direction)
    Z_new = -Y_old
    """
    
    T = smpl_pose.shape[0]
    
    # Root translation: (X, Y, Z) → (X, Z, -Y)
    transl_zup = np.zeros_like(transl)
    transl_zup[:, 0] = transl[:, 0]
    transl_zup[:, 1] = transl[:, 2]
    transl_zup[:, 2] = -transl[:, 1]
    
    # Rotations: Apply R_transform to each axis-angle
    R_yup_to_zup = rotation_matrix_xaxis(-np.pi/2)  # Rot(-90°, X)
    
    for t in range(T):
        for j in range(22):
            aa = smpl_pose[t, j*3:(j+1)*3]
            R = exp_map(aa)
            R_zup = R_yup_to_zup @ R @ R_yup_to_zup.T
            smpl_pose_zup[t, j*3:(j+1)*3] = log_map(R_zup)
    
    return smpl_pose_zup, transl_zup
```

#### zup_to_yup (lines 289-319)

Inverse transform (Z-up → Y-up for final output):

```python
def zup_to_yup(smpl_pose: np.ndarray, transl: np.ndarray):
    """Z-up (physics) → Y-up (motion capture).
    
    Inverse: Rotate around X-axis by +90°
    Same procedure as yup_to_zup but with +90°.
    """
```

---

## Part 2: Motion to MuJoCo State Conversion

### 2.1 SMPL to qpos: smpl_to_qpos() (lines 321-426)

Converts SMPL 72D axis-angle to MuJoCo 76D configuration space.

**SMPL Format (72D):**
- Root: 3D axis-angle
- 23 body joints: 3D axis-angle each (23 × 3 = 69D)

**MuJoCo qpos Format (76D):**
- Root translation: 3D `[x, y, z]`
- Root orientation: 4D quaternion `[w, x, y, z]`
- Body joints: 69D Euler angles in MuJoCo joint order

**Joint Order Mapping:**

SMPL uses standard SMPL order:
```
0: Pelvis (root)
1-2: L_Hip, R_Hip
3-6: Spine1, L_Knee, R_Knee, Spine2
7-8: L_Ankle, R_Ankle
9: Spine3
10-11: L_Foot, R_Foot
12-21: Neck, shoulders, elbows, wrists, hands
```

MuJoCo uses depth-first tree order (from smpl_humanoid.xml):
```
0: Pelvis (free joint — not actuated)
1: L_Hip
2: L_Knee
3: L_Ankle
4: L_Toe (= SMPL L_Foot)
... (right side mirror) ...
```

**Conversion Code:**

```python
def smpl_to_qpos(smpl_pose, transl, body_pos_1, model=None) -> qpos:
    """(T, 72) SMPL pose + (T, 3) trans → (T, 76) qpos."""
    
    T = smpl_pose.shape[0]
    qpos = np.zeros((T, 76), dtype=np.float64)
    
    joint_aa = smpl_pose.reshape(T, 24, 3)
    
    # [1] Root position + body offset
    qpos[:, :3] = transl + body_pos_1  # Account for Pelvis offset in XML
    
    # [2] Root orientation: axis-angle → quaternion (wxyz)
    root_quat_xyzw = Rotation.from_rotvec(joint_aa[:, 0]).as_quat()  # (T, 4)
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]  # Reorder to wxyz
    
    # [3] Body joints: axis-angle → Euler (intrinsic XYZ)
    body_aa = joint_aa[:, 1:].reshape(-1, 3)  # (T*23, 3)
    body_euler = Rotation.from_rotvec(body_aa).as_euler("xyz")  # Intrinsic XYZ
    body_euler = body_euler.reshape(T, 23, 3)
    
    # [4] Reorder: SMPL order → MuJoCo tree order
    body_euler_mj = body_euler[:, SMPL_2_MUJOCO]
    qpos[:, 7:] = body_euler_mj.reshape(T, 69)
    
    # [5] Joint limit enforcement (prevent impossible PD targets)
    if model is not None:
        # Guard axes (range < 15°): center to prevent chatter
        # Main axes (range ≥ 15°): clamp to limits
        ...
    
    return qpos  # (T, 76)
```

### 2.2 Euler Angle Convention

**Why "xyz" (Intrinsic XYZ)?**

MuJoCo XML specifies `<compiler coordinate="local"/>`, which means joint angles are local/intrinsic rotations:

```
R_total = Rx(θx) * Ry(θy) * Rz(θz)  [intrinsic composition]
```

The `as_euler("xyz")` call from scipy outputs angles in order `[θx, θy, θz]`, which directly matches MuJoCo's qpos slots `[X_joint, Y_joint, Z_joint]`.

**⚠️ Common Error:** PHC uses `as_euler("ZYX")` (extrinsic), which outputs `[θz, θy, θx]` — swapping X/Z slots. This works for RL policies (not PD) but breaks direct PD tracking.

### 2.3 qpos_to_smpl: Inverse Conversion (lines 429-464)

Converts simulated MuJoCo qpos back to SMPL 72D:

```python
def qpos_to_smpl(qpos, body_pos_1):
    """(T, 76) qpos → (T, 72) SMPL pose + (T, 3) trans."""
    
    T = qpos.shape[0]
    
    # [1] Root translation: remove body_pos offset
    transl = (qpos[:, :3] - body_pos_1).astype(np.float32)
    
    # [2] Root orientation: quaternion → axis-angle
    root_quat_wxyz = qpos[:, 3:7]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    root_aa = Rotation.from_quat(root_quat_xyzw).as_rotvec()  # (T, 3)
    
    # [3] Body joints: Euler → axis-angle
    body_euler_mj = qpos[:, 7:].reshape(T, 23, 3)  # MuJoCo order
    body_euler_smpl = body_euler_mj[:, MUJOCO_2_SMPL]  # SMPL order
    body_aa = Rotation.from_euler("xyz", body_euler_smpl.reshape(-1, 3)).as_rotvec()
    body_aa = body_aa.reshape(T, 23, 3)
    
    # [4] Concatenate
    smpl_pose = np.zeros((T, 72), dtype=np.float32)
    smpl_pose[:, :3] = root_aa.astype(np.float32)
    smpl_pose[:, 3:] = body_aa.reshape(T, 69)
    
    return smpl_pose, transl
```

### 2.4 Joint Limit Handling: Preventing PD Chatter

**Problem:** Euler decomposition of large rotations (deep knee bends, crouches) spreads rotation across "guard axes" — narrow-range joints meant only for tiny adjustments:

- Knee/ankle X, Z joints: ±5.6° limits (lateral wobble only)
- Euler decomposition of 120° bend: might spread to ±180° on X-axis
- Result: PD controller "chatters" at joint stop (persistently pushes against limit)

**Solution:** Two-tier joint limit enforcement (lines 395-424):

```python
GUARD_AXIS_THRESHOLD = np.radians(15.0)  # Range < 15° is "guard"

for joint in model.joints:
    lo, hi = joint.limit  # Joint angle range
    joint_range = hi - lo
    center = (lo + hi) / 2
    
    if joint_range < GUARD_AXIS_THRESHOLD:
        # Guard axis: ignore Euler output, center instead
        qpos[:, qi] = center  # Usually 0
        # Loss: ≤ 5.6° on minor axis — imperceptible
    else:
        # Main axis: clamp to valid range
        qpos[:, qi] = np.clip(qpos[:, qi], lo, hi)
```

**Effect:**
- ✅ Eliminates PD chatter entirely
- ✅ Loss is imperceptible (max ±5.6° on minor axes)
- ✅ Large rotations (knee bends) still tracked on main axes

---

## Part 3: Ground Offset Computation

### 3.1 compute_ground_offset() (lines 471-523)

**Problem:** Motion data may be generated for taller/shorter body models than MuJoCo SMPL humanoid, causing feet to float above or sink into ground.

**Solution:** Find vertical offset so feet touch ground in frame 0.

```python
def compute_ground_offset(model, data, ref_qpos: np.ndarray) -> float:
    """Compute Z offset to align first frame feet to ground."""
    
    # Set MuJoCo to frame 0
    data.qpos[:] = ref_qpos[0]
    mujoco.mj_forward(model, data)
    
    # Get foot body positions (indices from model)
    L_foot_pos = data.xpos[model.body_name2id("L_Foot")]
    R_foot_pos = data.xpos[model.body_name2id("R_Foot")]
    
    # Minimum foot height (should touch ground at z=0)
    min_foot_z = min(L_foot_pos[2], R_foot_pos[2])
    
    # Required offset to bring feet to ground
    ground_offset = min_foot_z
    
    return ground_offset  # Scalar, typically ≤ 0.2m
```

**Application:** Subtract offset from all frames' Z-translation:

```python
ref_qpos[:, 2] -= ground_offset
```

---

## Part 4: Physics Simulation Loop

### 4.1 run_physics_sim() (lines 609-717)

**Core Algorithm:** PD-tracking with kinematic root

**Root Strategy:**
- MuJoCo cannot apply PD control to free joint (root)
- Solution: Reset root pose (position + orientation) each control step
- Use finite-difference velocities for smooth physics interpolation

**Body Joints:**
- PD actuators track reference angles via MuJoCo's built-in PD controller
- Physics enforces ground contact, prevents penetration
- Gravity pulls body downward realistically

**Pseudocode:**

```python
def run_physics_sim(model, data, ref_qpos, fps=30):
    """Simulate motion with PD body joint tracking + kinematic root."""
    
    T = ref_qpos.shape[0]
    sim_dt = model.opt.timestep  # Usually 0.005s
    ctrl_dt = 1.0 / fps           # Usually 0.033s for 30fps
    decimation = int(ctrl_dt / sim_dt)  # Steps per control frame
    
    # Initialize
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    
    sim_qpos_list = []
    min_root_h = float("inf")
    fall_frame = None
    
    for t in range(T):
        # [1] KINEMATIC ROOT: Reset to reference each frame
        data.qpos[:7] = ref_qpos[t, :7]
        
        # [2] ROOT VELOCITY: Finite difference for smooth interpolation
        if t + 1 < T:
            # Linear: (pos_next - pos_cur) / ctrl_dt
            data.qvel[:3] = (ref_qpos[t+1, :3] - ref_qpos[t, :3]) / ctrl_dt
            
            # Angular: from quaternion difference
            q_cur = Rotation.from_quat(ref_qpos[t, 3:7])     # wxyz format
            q_next = Rotation.from_quat(ref_qpos[t+1, 3:7])
            R_diff = q_cur.inv() * q_next
            data.qvel[3:6] = R_diff.as_rotvec() / ctrl_dt
        else:
            data.qvel[:6] = 0.0
        
        # [3] PD TARGETS: Body joints track reference
        data.ctrl[:] = ref_qpos[t, 7:]  # All 69 joint angles
        
        # [4] PHYSICS STEPS: Sub-step with decimation
        for _ in range(decimation):
            mujoco.mj_step(model, data)  # 1 physics step
        
        # [5] RECORD STATE
        sim_qpos_list.append(data.qpos.copy())
        
        # [6] FALL DETECTION
        root_h = data.qpos[2]
        min_root_h = min(min_root_h, root_h)
        
        if root_h < FALL_HEIGHT_THRESHOLD or np.any(np.isnan(data.qpos)):
            fall_frame = t
            print(f"FALL at frame {t}")
            break
    
    sim_qpos = np.array(sim_qpos_list)
    T_sim = len(sim_qpos)
    
    # [7] COMPUTE STATISTICS
    joint_error = np.mean(np.abs(sim_qpos[:, 7:] - ref_qpos[:T_sim, 7:]))
    root_drift = np.linalg.norm(sim_qpos[-1, :3] - ref_qpos[T_sim-1, :3])
    
    stats = {
        "total_frames": T,
        "simulated_frames": T_sim,
        "fall_frame": fall_frame,
        "completed": fall_frame is None,
        "joint_tracking_error_rad": float(joint_error),
        "root_position_drift_m": float(root_drift),
        "min_root_height_m": float(min_root_h),
    }
    
    return sim_qpos, stats
```

### 4.2 Key Design Decisions

#### Why Kinematic Root?

| Aspect | Kinematic Root | Learned RL Policy |
|--------|---|---|
| Root trajectory | Exact (follows reference) | Learned (may deviate) |
| Body joints | PD-tracked + physics | RL-controlled |
| Foot placement | Reference + physics correction | Learned |
| Stability | ✅ No root drift | ✅ Natural locomotion |
| Use case | **Post-training fine-tuning** | **Embodied generation** |

For post-training physics refinement, we want to keep root trajectory faithful while improving body joint physics compliance.

#### Why Finite-Difference Velocities?

MuJoCo physics sub-steps at 200Hz, but control commands come at 30Hz. Setting qvel to the finite-difference velocity ensures smooth interpolation rather than a "teleport + settle" pattern:

```
Linear interpolation: x(t+δ) = x(t) + v * δ
Matches: data.qvel[:3] = (x_next - x_cur) / Δt
```

#### Fall Detection Threshold

```python
FALL_HEIGHT_THRESHOLD = 0.3  # meters, typically
```

If root height drops below threshold, simulation is aborted. Also detects NaN (numerical explosion).

### 4.3 Simulation Statistics

```python
stats = {
    "total_frames": int,              # Requested frames
    "simulated_frames": int,          # Actually completed (may be < total if fall)
    "fall_frame": int or None,        # Frame where fall detected
    "completed": bool,                # True if no fall
    
    "joint_tracking_error_rad": float, # Mean |sim_angle - ref_angle|
    "root_position_drift_m": float,   # ||sim_pos_end - ref_pos_end||
    "min_root_height_m": float,       # Minimum root Z across simulation
    
    "fps": int,                        # Control frame rate
    "decimation": int,                # Sub-steps per control frame
}
```

**Interpretation:**
- `joint_tracking_error_rad < 0.05`: Good PD tracking (≈ 3° average error)
- `root_position_drift_m < 0.05`: Negligible root drift (kinematic should be ~0)
- `completed`: True indicates full successful simulation
- `min_root_height_m > 0.3`: Plausible (feet not penetrating ground)

---

## Part 5: Post-Simulation Smoothing

### 5.1 smooth_simulated_qpos() (lines 720-795)

**Problem:** PD tracking introduces high-frequency oscillation (jitter) from discrete control loop.

**Solution:** Low-pass filter + blend with kinematic reference.

```python
def smooth_simulated_qpos(sim_qpos, ref_qpos, fps=30,
                         window_ms=333.0, blend_alpha=0.5):
    """Smooth PD oscillations while preserving physics."""
    
    from scipy.signal import savgol_filter
    
    T = min(sim_qpos.shape[0], ref_qpos.shape[0])
    smoothed = sim_qpos[:T].copy()
    
    # Window size: default 333ms = 10 frames @ 30fps
    window_frames = max(3, int(round(window_ms / 1000.0 * fps)))
    if window_frames % 2 == 0:
        window_frames += 1  # Must be odd
    polyorder = min(3, window_frames - 1)
    
    if T < window_frames:
        # Too short: just blend
        smoothed[:T, 7:] = blend_alpha * sim_qpos[:T, 7:] + \
                           (1 - blend_alpha) * ref_qpos[:T, 7:]
        return smoothed
    
    # Body joints only (root stays kinematic)
    sim_body = sim_qpos[:T, 7:]     # (T, 69)
    ref_body = ref_qpos[:T, 7:]
    
    # [1] Savitzky-Golay smooth
    smooth_sim = savgol_filter(sim_body, window_frames, polyorder, axis=0)
    
    # [2] Blend with reference
    smoothed_body = blend_alpha * smooth_sim + (1 - blend_alpha) * ref_body
    smoothed[:T, 7:] = smoothed_body
    
    # [3] Report jerk reduction
    raw_jerk = np.mean(np.abs(np.diff(sim_body, n=3, axis=0))) * (fps**3)
    smooth_jerk = np.mean(np.abs(np.diff(smoothed_body, n=3, axis=0))) * (fps**3)
    jerk_reduction = (1 - smooth_jerk / raw_jerk) * 100 if raw_jerk > 0 else 0
    
    print(f"Jerk: {raw_jerk:.0f} → {smooth_jerk:.0f} (-{jerk_reduction:.0f}%)")
    
    return smoothed
```

### 5.2 Savitzky-Golay Filter

**What it does:** Polynomial curve-fitting with local windows

- Preserves discontinuities better than Gaussian blur
- Smooths within window, doesn't shift edges
- Good for removing oscillation while keeping real motion features

**Parameters:**
- `window_ms=333.0` (default): ~10 frames @ 30fps
  - Larger: more smoothing, more latency
  - Smaller: less smoothing, more jitter remains
- `blend_alpha=0.5` (default): 50% physics, 50% kinematic reference
  - 1.0: all physics (if jitter is acceptable)
  - 0.0: pure kinematic (no physics benefit)

### 5.3 Jerk Metric

**Jerk = 3rd time derivative of position (m/s³)**

```python
jerk = np.diff(smoothed, n=3, axis=0) * (fps ** 3)
mean_jerk = np.mean(np.abs(jerk))
```

High jerk = "jerky" motion (unnatural rapid changes in acceleration)

Typical values:
- Kinematic reference: 50-200 (m/s³)
- Raw physics: 100-500 (jittery PD control)
- After smoothing: 50-150 (clean, natural)

---

## Part 6: Complete End-to-End Pipeline

### 6.1 process_single_motion() (lines 950-1037)

Orchestrates full pipeline:

```python
def process_single_motion(npz_path, xml_path, output_dir, stats_dir=None, fps=30):
    """motion_135 → physics-corrected SMPL → JSON mesh."""
    
    print(f"Processing: {Path(npz_path).stem}")
    
    # [1] DECODE MOTION_135
    smpl_pose, transl, motion_fps = decode_motion_135(npz_path)
    fps = motion_fps or fps
    T = smpl_pose.shape[0]
    print(f"  Decoded: {T} frames @ {fps}fps, duration={T/fps:.1f}s")
    
    # [2] Y-UP → Z-UP
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)
    
    # [3] SMPL 72D → MUJOCO 76D
    model, data = load_mujoco_model(xml_path)
    body_pos_1 = model.body_pos[1].copy()
    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1, model=model)
    
    # [4] GROUND OFFSET
    ground_offset = compute_ground_offset(model, data, ref_qpos)
    if abs(ground_offset) > 0.001:
        ref_qpos[:, 2] -= ground_offset
    
    # [5] PHYSICS SIMULATION
    sim_qpos, stats = run_physics_sim(model, data, ref_qpos, fps)
    stats["ground_offset_m"] = float(ground_offset)
    T_sim = stats["simulated_frames"]
    
    # [6] POST-SIMULATION SMOOTHING
    sim_qpos_smooth = smooth_simulated_qpos(sim_qpos, ref_qpos[:T_sim], fps)
    
    # [7] Z-UP → Y-UP
    smpl_pose_final, transl_final = zup_to_yup(
        qpos_to_smpl(sim_qpos_smooth, body_pos_1)[0],
        qpos_to_smpl(sim_qpos_smooth, body_pos_1)[1]
    )
    
    # [8] EXPORT (OPTIONAL)
    smpl_to_mesh_json(smpl_pose_final, transl_final, output_dir, fps)
    
    # [9] SAVE STATS
    if stats_dir:
        json.dump(stats, open(f"{stats_dir}/{stem}.json", "w"))
    
    return stats
```

### 6.2 Usage Example

```bash
# Single file
python3 scripts/embodied/run_smpl_physics_sim.py \
    --npz-file output/walk_forward.npz \
    --output-dir output/smpl_mesh_physics \
    --xml-path ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml

# Batch with stats
python3 scripts/embodied/run_smpl_physics_sim.py \
    --npz-dir output/npz \
    --output-dir output/smpl_mesh_physics \
    --xml-path ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --stats-dir output/sim_stats \
    --filter-flat-ground
```

---

## Part 7: Integration with motion135_to_smplx.py

The companion script `scripts/embodied/motion135_to_smplx.py` provides a simpler conversion (without physics) for external tools:

**Use motion135_to_smplx.py when:**
- Need SMPL-X NPZ format for external tools (GMR, retargeting)
- Physics not required yet
- Fast conversion without MuJoCo dependency

**Use run_smpl_physics_sim.py when:**
- Need physically plausible motion
- Want to remove foot sliding
- Generating embodied motion with physics constraints

**Data Flow:**

```
HyMotion motion_135 output
  ├─→ motion135_to_smplx.py ─→ SMPL-X NPZ (clean data)
  │
  └─→ run_smpl_physics_sim.py
      ├─→ motion135_to_smplx.py (internal step 1)
      ├─→ coord transform (Y-up → Z-up)
      ├─→ MuJoCo simulation
      ├─→ smoothing
      └─→ export JSON for visualization
```

---

## Part 8: Hyperparameters and Tuning

### 8.1 Physics Simulation Parameters

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `fps` | 30 | 20-60 | Control frequency. Higher = finer control, more expensive. |
| `decimation` | auto | 4-10 | Sub-steps per control frame. Auto-computed from timestep. |
| `FALL_HEIGHT_THRESHOLD` | 0.3 m | 0.1-0.5 | Abort if root drops below (m). Lower = stricter. |

### 8.2 Post-Smoothing Parameters

| Parameter | Default | Effect |
|---|---|---|
| `window_ms` | 333 ms | Smoothing window size. Larger = more smoothing. |
| `blend_alpha` | 0.5 | Ratio of physics to kinematic. 1.0 = all physics. |

**Tuning guide:**
- Too much jitter? Increase `blend_alpha` toward 1.0
- Losing physics detail? Increase `blend_alpha` toward 1.0
- Motion feels fake? Decrease `blend_alpha` toward 0.5

### 8.3 Joint Limit Enforcement

| Parameter | Default | Purpose |
|---|---|---|
| `GUARD_AXIS_THRESHOLD` | 15° | Range below which axis is "guard" (centered). |

Guard axes are automatically centered to prevent PD chatter. Fine-tuning rarely needed.

---

## Part 9: Common Issues and Solutions

### Issue 1: Feet Floating Above Ground

**Symptom:** Root height too high, feet not touching ground.

**Cause:** Motion data generated for different body model (taller/shorter).

**Solution:** `compute_ground_offset()` automatically applies correction. Check output:
```
Ground offset: 0.145m (subtracting from all frames)
```

### Issue 2: PD Chatter / Jitter

**Symptom:** Smooth output, but noisy joint angles (high-frequency oscillation).

**Cause:** Discrete PD control loop.

**Solution:** Increase `blend_alpha` in `smooth_simulated_qpos()`:
```python
smooth_qpos = smooth_simulated_qpos(sim_qpos, ref_qpos, blend_alpha=0.7)
```

Or increase Butterworth window:
```python
smooth_qpos = smooth_simulated_qpos(sim_qpos, ref_qpos, window_ms=500.0)
```

### Issue 3: Motion Falls / Explodes

**Symptom:** Simulation aborts early with `FALL at frame X`.

**Cause:** 
- Unphysical input (joint limits violated, bad rotations)
- PD gains too aggressive
- Reference motion too fast / jerky

**Solutions:**
- Validate reference qpos before simulation
- Lower PD gains in MuJoCo XML (if available)
- Pre-smooth reference motion before simulation

### Issue 4: Slow Performance

**Symptom:** Simulation takes minutes per motion.

**Cause:** Too many sub-steps (high decimation from low timestep).

**Solutions:**
- Check MuJoCo XML: `<option timestep="0.005"/>` (standard)
- Increase timestep if physics instability not observed: `0.01` (2x faster)
- Profile: Print sim_dt and decimation

```python
print(f"sim_dt={sim_dt:.5f}s, decimation={decimation}")
# If decimation > 10, consider larger timestep
```

---

## Part 10: Integration with SOAR Training

Physics simulation can feed into SOAR post-training as the **correction target**:

```python
# Standard SOAR correction target (from clean motion)
v_corr_clean = (x1_clean - z_re) / (1 - t_prime)

# Physics-enhanced correction target
# Replace with simulated motion at same timestep
v_corr_physics = (x1_simulated - z_re) / (1 - t_prime)
```

This combines:
- ✅ SOAR's exposure bias correction
- ✅ Physics-enforced constraints (no penetration, gravity)
- ✅ Naturally plausible motion

See `SOAR_PHYSICS_INTEGRATION_ANALYSIS.md` for details.

---

## Summary: Key Takeaways

1. **Format chain:** motion_135 (6D rot) → axis-angle → SMPL (Y-up) → qpos (Z-up) → MuJoCo → smooth → final SMPL (Y-up)

2. **Coordinate systems matter:** Y-up (motion capture) ≠ Z-up (physics engine). Must convert properly.

3. **Euler angle convention:** Use "xyz" (intrinsic) for direct PD tracking, not "ZYX" (extrinsic).

4. **Joint limits prevent chatter:** Guard axes (< 15° range) are centered; main axes are clamped.

5. **Kinematic root + PD body:** Root trajectory stays faithful (no drift), body joints get physics corrections.

6. **Post-processing smooths jitter:** Savitzky-Golay filter removes PD oscillation. Blending with kinematic reference bounds quality.

7. **Statistics tell you success:** joint_tracking_error should be < 0.05 rad; root_drift should be ~0.

8. **Scaling matters:** More frames = more compute. 1-minute motion @ 30fps = 1800 frames = ~60s sim @ decimation=6.

9. **Integration ready:** Physics-corrected motion feeds back into SOAR or other post-training methods for embodied refinement.

