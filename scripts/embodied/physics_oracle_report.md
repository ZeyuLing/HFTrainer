# Physics Correction Oracle Analysis
## File Paths & Complete Function Inventory

---

## 1. MOTION DECODING: `motion_135` → SMPL 72-dim axis-angle

### File: `scripts/embodied/run_smpl_physics_sim.py`
**Lines: 193–227**

#### Function Signature:
```python
def decode_motion_135(npz_path: str):
    """Decode motion_135 NPZ to SMPL 72-dim axis-angle pose + translation (Y-up).
    
    motion_135 format: (T, 135) = transl(3) + 22 x rot6d(6).
    SMPL expects 24 joints; joints 22-23 (L_Hand, R_Hand) are zero-padded.
    
    Returns:
        smpl_pose: (T, 72) axis-angle in SMPL joint order, Y-up
        transl:    (T, 3)  translation, Y-up
        fps:       int
    """
```

#### Key Logic:
```python
# [1] Load NPZ
data = np.load(npz_path, allow_pickle=True)
motion = data["motion_135"]  # (T, 135)
fps = int(data.get("fps", 30))
T = motion.shape[0]

# [2] Split motion_135 layout: [transl(3) | 22×rot6d(6)]
transl = motion[:, :3]                        # (T, 3)
rot6d = motion[:, 3:].reshape(T, 22, 6)       # (T, 22, 6) — HyMotion row-major layout

# [3] rot6d → rotation matrix → axis-angle
rotmat = rot6d_to_rotmat(rot6d)                # (T, 22, 3, 3)
aa = sRot.from_matrix(
    rotmat.reshape(-1, 3, 3)
).as_rotvec().reshape(T, 22, 3)                # (T, 22, 3)

# [4] Extract root & body, pad hand joints
root_orient = aa[:, 0, :]                      # (T, 3) — joint 0 (Pelvis)
body_pose = aa[:, 1:22, :].reshape(T, -1)      # (T, 63) — joints 1-21

# [5] Build full 72-dim SMPL pose (0-indexed: 24 joints × 3)
smpl_pose = np.zeros((T, 72), dtype=np.float32)
smpl_pose[:, :3] = root_orient
smpl_pose[:, 3:66] = body_pose
# smpl_pose[:, 66:72] = 0  (L_Hand, R_Hand — implicit)

return smpl_pose, transl.astype(np.float32), fps
```

---

## 2. ROT6D CONVERSION UTILITIES

### File: `scripts/embodied/motion135_to_smplx.py`
**Lines: 26–55**

#### Function Signature:
```python
def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix.
    
    HyMotion outputs rot6d in row-major layout: [R00,R01, R10,R11, R20,R21]
    Gram-Schmidt expects column-major layout: [R00,R10,R20, R01,R11,R21]
    We reorder [0,2,4,1,3,5] to convert row-major → column-major before decoding.
    
    Args:
        rot6d: (..., 6) array of 6D rotation representations (row-major)
    Returns:
        rotmat: (..., 3, 3) array of rotation matrices
    """
```

#### Key Logic:
```python
# [1] Row-major → column-major reorder: [0,2,4,1,3,5]
rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
a1 = rot6d[..., :3]
a2 = rot6d[..., 3:6]

# [2] Gram-Schmidt orthogonalization to build R from two column vectors
# First column: normalize a1
b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)

# Second column: orthogonalize a2 against b1
dot = np.sum(b1 * a2, axis=-1, keepdims=True)
b2 = a2 - dot * b1
b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)

# Third column: cross product for determinant = +1
b3 = np.cross(b1, b2)

# Stack columns into rotation matrix
rotmat = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)
return rotmat
```

**Also in `run_smpl_physics_sim.py` lines 173–190:**
Identical implementation with same logic.

#### Second Conversion (rotmat → axis-angle):
```python
def rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle representation."""
    from scipy.spatial.transform import Rotation as R
    
    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    rot = R.from_matrix(rotmat_flat)
    aa_flat = rot.as_rotvec()
    return aa_flat.reshape(*orig_shape, 3)
```

---

## 3. COORDINATE TRANSFORMS: Y-up ↔ Z-up

### File: `scripts/embodied/run_smpl_physics_sim.py`
**Lines: 234–314**

#### Transform Matrices (Global):
```python
# SMPL Y-up → MuJoCo Z-up (cyclic permutation)
# SMPL axes: X=left, Y=up, Z=forward
# MuJoCo axes: X=forward, Y=left, Z=up
# Mapping: [x,y,z]_yup → [z,x,y]_zup
_YUP_TO_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

# Inverse: [x,y,z]_zup → [y,z,x]_yup
_ZUP_TO_YUP = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
```

#### Function Signatures:
```python
def yup_to_zup(smpl_pose: np.ndarray, transl: np.ndarray):
    """Transform SMPL pose + translation from Y-up to Z-up.
    
    All joints (root + body) need coordinate transform because:
    - Root orientation is in global frame (Y-up → Z-up)
    - Body joint axis-angles are in LOCAL body frames
    - In T-pose, local frames align with global → same transform needed
    
    CRITICAL: Without this transform, SMPL knee flexion (local X axis)
    maps to MuJoCo X-joint (forward ±5.6°), not Y-joint (lateral flexion).
    The transform fixes this for PD position control.
    
    Returns:
        out_pose:   (T, 72) SMPL axis-angle, Z-up
        out_transl: (T, 3)  translation, Z-up
    """

def zup_to_yup(smpl_pose: np.ndarray, transl: np.ndarray):
    """Transform SMPL pose + translation from Z-up to Y-up.
    
    Inverse of yup_to_zup. Used after MuJoCo simulation to convert back
    to original HyMotion format for export.
    
    Returns:
        out_pose:   (T, 72) SMPL axis-angle, Y-up
        out_transl: (T, 3)  translation, Y-up
    """
```

#### Key Logic:
```python
def yup_to_zup(smpl_pose: np.ndarray, transl: np.ndarray):
    T = smpl_pose.shape[0]
    
    # Translation: [x,y,z]_yup → [z,x,y]_zup
    out_transl = (transl.astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32)
    
    # Transform ALL joint axis-angles: root (0:3) + body (3:72)
    # Each joint's axis-angle is a 3D vector; cyclic permutation of axes
    pose_72 = smpl_pose[:, :72].astype(np.float64)  # (T, 72)
    pose_72_3d = pose_72.reshape(T * 24, 3)  # (T*24, 3) — 24 joints
    pose_72_zup = (pose_72_3d @ _YUP_TO_ZUP.T).reshape(T, 72)
    out_pose = smpl_pose.copy()
    out_pose[:, :72] = pose_72_zup.astype(np.float32)
    
    return out_pose, out_transl
```

---

## 4. SMPL → MuJoCo QPOS: `motion135_to_qpos()`

### File: `scripts/embodied/run_smpl_physics_sim.py`
**Lines: 321–426**

#### Function Signature:
```python
def smpl_to_qpos(smpl_pose: np.ndarray, transl: np.ndarray,
                 body_pos_1: np.ndarray,
                 model=None) -> np.ndarray:
    """Convert SMPL 72-dim axis-angle pose to MuJoCo 76-dim qpos.
    
    qpos layout: [root_trans(3), root_quat_wxyz(4), joint_euler_xyz(69)]
    
    Euler convention for PD tracking:
      MuJoCo smpl_humanoid.xml has <compiler coordinate="local"/> with
      intrinsic XYZ: R = Rx(θx) * Ry(θy) * Rz(θz).
      We use as_euler("xyz") which outputs [x, y, z] angles directly
      matching qpos slots [X_joint, Y_joint, Z_joint].
    
    Joint limit handling (when model is provided):
      Euler angle decomposition of large rotations can spread rotation
      onto "guard" axes (narrow-limit joints like ±5.6°). Two-tier fix:
        1. Guard axes (range < 15°): Set PD target to center of range
        2. Main axes (range ≥ 15°): Clamp to joint limits
    
    Args:
        smpl_pose:  (T, 72) axis-angle, Z-up root / local body joints, SMPL order
        transl:     (T, 3)  translation, Z-up
        body_pos_1: (3,)    Pelvis body position offset from MuJoCo XML
        model:      optional MuJoCo model (for joint limit clamping)
    Returns:
        qpos: (T, 76) float64
    """
```

#### Key Logic:
```python
T = smpl_pose.shape[0]
qpos = np.zeros((T, 76), dtype=np.float64)

joint_aa = smpl_pose.reshape(T, 24, 3)  # (T, 24, 3) — 24 SMPL joints

# [1] Root translation: add Pelvis body_pos offset
qpos[:, :3] = transl.astype(np.float64) + body_pos_1

# [2] Root orientation: axis-angle → quaternion wxyz
root_quat_xyzw = sRot.from_rotvec(joint_aa[:, 0]).as_quat()  # (T, 4) xyzw
qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]               # → wxyz

# [3] Body joints (1-23): axis-angle → intrinsic XYZ Euler angles
# as_euler("xyz") = intrinsic XYZ, output order [x, y, z]
# This matches qpos slots [X_joint, Y_joint, Z_joint] for PD tracking.
body_aa = joint_aa[:, 1:].reshape(-1, 3)               # (T*23, 3)
body_euler = sRot.from_rotvec(body_aa).as_euler("xyz")  # (T*23, 3) = [x, y, z]
body_euler = body_euler.reshape(T, 23, 3)               # (T, 23, 3) SMPL order

# [4] Reorder from SMPL joint order to MuJoCo tree order
body_euler_mj = body_euler[:, SMPL_2_MUJOCO]           # (T, 23, 3)
qpos[:, 7:] = body_euler_mj.reshape(T, 69)

# [5] Joint limit handling (optional, with model)
# Problem: Euler decomposition of large rotations spreads onto guard axes
# Solution: Center guard axes (< 15° range), clamp main axes
if model is not None:
    GUARD_AXIS_THRESHOLD = np.radians(15.0)
    for jid in range(model.njnt):
        if model.jnt_type[jid] != 3:  continue  # hinge only
        if not model.jnt_limited[jid]: continue
        qi = model.jnt_qposadr[jid]
        lo, hi = model.jnt_range[jid]
        joint_range = hi - lo
        center = (lo + hi) / 2.0
        
        if joint_range < GUARD_AXIS_THRESHOLD:
            qpos[:, qi] = center  # Guard: center to avoid chatter
        else:
            qpos[:, qi] = np.clip(qpos[:, qi], lo, hi)  # Main: clamp

return qpos
```

#### Joint Reordering Arrays (Global Constants):
```python
# Lines 133–136: Build SMPL → MuJoCo and MuJoCo → SMPL reorder indices
SMPL_2_MUJOCO, MUJOCO_2_SMPL = _build_reorder_indices()
# SMPL_2_MUJOCO = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 14, 12, 15, 17, 19, 21, 13, 16, 18, 20, 22]
# MUJOCO_2_SMPL = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 13, 18, 12, 14, 19, 15, 20, 16, 21, 17, 22]
```

---

## 5. MuJoCo QPOS → SMPL: `qpos_to_motion135()`

### File: `scripts/embodied/run_smpl_physics_sim.py`
**Lines: 429–464**

#### Function Signature:
```python
def qpos_to_smpl(qpos: np.ndarray, body_pos_1: np.ndarray):
    """Convert MuJoCo 76-dim qpos to SMPL 72-dim axis-angle pose.
    
    Inverse of smpl_to_qpos(). Uses "xyz" (intrinsic XYZ) Euler convention.
    qpos slots [X_joint, Y_joint, Z_joint] → from_euler("xyz") reads [x, y, z].
    
    Args:
        qpos:       (T, 76) float64
        body_pos_1: (3,)    Pelvis body position offset from MuJoCo XML
    Returns:
        smpl_pose: (T, 72) axis-angle, Z-up root / local body joints, SMPL order
        transl:    (T, 3)  translation, Z-up
    """
```

#### Key Logic:
```python
T = qpos.shape[0]

# [1] Root translation: undo Pelvis body_pos offset
transl = (qpos[:, :3] - body_pos_1).astype(np.float32)

# [2] Root orientation: quaternion wxyz → axis-angle
root_quat_wxyz = qpos[:, 3:7]
root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]   # → xyzw
root_aa = sRot.from_quat(root_quat_xyzw).as_rotvec()  # (T, 3)

# [3] Body joints: intrinsic XYZ Euler → axis-angle
# qpos stores [x, y, z] per body → from_euler("xyz") interprets correctly
body_euler_mj = qpos[:, 7:].reshape(T, 23, 3)         # MuJoCo tree order
body_euler_smpl = body_euler_mj[:, MUJOCO_2_SMPL]      # → SMPL order
body_aa = sRot.from_euler(
    "xyz", body_euler_smpl.reshape(-1, 3)
).as_rotvec().reshape(T, 23, 3)

# [4] Assemble full 72-dim SMPL pose
smpl_pose = np.zeros((T, 72), dtype=np.float32)
smpl_pose[:, :3] = root_aa.astype(np.float32)
smpl_pose[:, 3:] = body_aa.reshape(T, 69).astype(np.float32)

return smpl_pose, transl
```

---

## 6. PD TRACKING PHYSICS SIMULATION LOOP

### File: `scripts/embodied/run_smpl_physics_sim.py`
**Lines: 609–717**

#### Function Signature:
```python
def run_physics_sim(model, data, ref_qpos: np.ndarray, fps: int = 30):
    """Run PD-tracking physics simulation with kinematic root tracking.
    
    Root joint strategy: The SMPL humanoid's root is a free joint with NO
    actuator — MuJoCo cannot apply PD control to it. Instead, reset the root
    pose (position + orientation) to kinematic reference each control frame.
    This gives:
      - Kinematic root trajectory (same as reference)
      - Physics-enforced body joints (PD tracking + contact + gravity)
      - Natural foot-ground interaction (no penetration, reduced sliding)
    
    Root velocity (qvel[:6]) is set to finite-difference of the reference
    trajectory so physics sub-steps produce smooth interpolation between
    control frames (not discontinuous teleport).
    
    Args:
        model:    MuJoCo model (configured with PD actuators)
        data:     MuJoCo data
        ref_qpos: (T, 76) reference qpos trajectory
        fps:      control frame rate
    Returns:
        sim_qpos: (T', 76) simulated qpos (T' <= T, shorter if fall detected)
        stats:    dict with simulation statistics
    """
```

#### Key Logic (Main Loop):
```python
T = ref_qpos.shape[0]
sim_dt = model.opt.timestep
ctrl_dt = 1.0 / fps
decimation = max(1, int(round(ctrl_dt / sim_dt)))

print(f"sim_dt={sim_dt:.5f}s, ctrl_dt={ctrl_dt:.4f}s, decimation={decimation}")

# Initialize with first frame
data.qpos[:] = ref_qpos[0]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

sim_qpos_list = []
fall_frame = None
min_root_h = float("inf")

# ========== MAIN CONTROL LOOP ==========
for t in range(T):
    # ---- [1] Root tracking: reset root to reference ----
    # Set root position and quaternion from reference
    data.qpos[:7] = ref_qpos[t, :7]
    
    # [2] Compute root velocity from finite differences
    # so physics sub-steps interpolate smoothly (not teleport).
    if t + 1 < T:
        # Linear velocity: (pos_next - pos_cur) / ctrl_dt
        data.qvel[:3] = (ref_qpos[t + 1, :3] - ref_qpos[t, :3]) / ctrl_dt
        
        # Angular velocity: from quaternion difference
        # MuJoCo uses wxyz quaternion
        q_cur = ref_qpos[t, 3:7][[1, 2, 3, 0]]   # wxyz -> xyzw
        q_next = ref_qpos[t + 1, 3:7][[1, 2, 3, 0]]
        R_diff = sRot.from_quat(q_cur).inv() * sRot.from_quat(q_next)
        data.qvel[3:6] = R_diff.as_rotvec() / ctrl_dt
    else:
        data.qvel[:6] = 0.0
    
    # ---- [3] PD targets for body joints ----
    # Body joints are ACTUATED via PD; control = target position
    data.ctrl[:] = ref_qpos[t, 7:]  # (69,) → body joint Euler targets
    
    # ---- [4] Step physics (decimation sub-steps per control frame) ----
    for _ in range(decimation):
        mujoco.mj_step(model, data)
    
    # ---- [5] Record simulated qpos ----
    sim_qpos_list.append(data.qpos.copy())
    
    # ---- [6] Track root height ----
    root_h = float(data.qpos[2])
    min_root_h = min(min_root_h, root_h)
    
    # ---- [7] Fall detection ----
    if root_h < FALL_HEIGHT_THRESHOLD or np.any(np.isnan(data.qpos)):
        fall_frame = t
        reason = "NaN" if np.any(np.isnan(data.qpos)) else f"root_h={root_h:.3f}m"
        print(f"FALL at frame {t}/{T}: {reason}")
        break

# ========== POST-SIMULATION STATS ==========
sim_qpos = np.array(sim_qpos_list)
T_sim = len(sim_qpos)

# Compute tracking error (mean absolute joint angle error in radians)
joint_error = float(np.mean(np.abs(sim_qpos[:, 7:] - ref_qpos[:T_sim, 7:])))

# Compute root position drift
root_drift = float(np.linalg.norm(
    sim_qpos[-1, :3] - ref_qpos[min(T_sim - 1, T - 1), :3]
))

stats = {
    "total_frames": int(T),
    "simulated_frames": int(T_sim),
    "fall_frame": int(fall_frame) if fall_frame is not None else None,
    "completed": fall_frame is None,
    "joint_tracking_error_rad": joint_error,
    "root_position_drift_m": root_drift,
    "min_root_height_m": float(min_root_h),
    "ground_offset_m": 0.0,
    "fps": fps,
    "decimation": decimation,
}

return sim_qpos, stats
```

#### PD Gains Configuration (Global Constants):
```python
# Lines 147–156
PD_GAINS_PER_BODY = {
    "L_Hip": (1000, 20),    "L_Knee": (1000, 20),   "L_Ankle": (800, 18),  "L_Toe": (400, 13),
    "R_Hip": (1000, 20),    "R_Knee": (1000, 20),   "R_Ankle": (800, 18),  "R_Toe": (400, 13),
    "Torso": (2000, 28),    "Spine": (2000, 28),     "Chest": (2000, 28),
    "Neck": (200, 9),       "Head": (200, 9),
    "L_Thorax": (800, 18),  "L_Shoulder": (800, 18), "L_Elbow": (600, 16),
    "L_Wrist": (200, 9),    "L_Hand": (200, 9),
    "R_Thorax": (800, 18),  "R_Shoulder": (800, 18), "R_Elbow": (600, 16),
    "R_Wrist": (200, 9),    "R_Hand": (200, 9),
}
# Format: (kp, kd) per body
# Critically damped: ζ = kd/(2√(kp*armature)) = 1.0 for tracking τ ≈ 0.02s
```

#### Model Configuration:
```python
def load_mujoco_model(xml_path: str):
    """Load and configure MuJoCo SMPL humanoid model for PD-tracking physics sim.
    
    Strategy: Zero ALL passive dynamics so only PD actuators drive motion.
    Keep armature for numerical stability.
    
    XML overrides:
      - dof_damping = 80 → 0  (CRITICAL FIX: was overdamping, ζ=20.6!)
      - jnt_stiffness = 800 → 0
      - dof_frictionloss → 0
      - dof_armature = 0.02 → 0.1 for stability
    
    PD actuators: force_i = kp*(ctrl_i - qpos_i) - kd*qvel_i
    """
    # [1] Load XML
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # [2] Zero passive dynamics
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0      # CRITICAL: was 80
    model.dof_frictionloss[:] = 0.0
    
    # [3] Increase armature for numerical stability
    model.dof_armature[6:] = 0.1    # body joints only
    
    # [4] Configure per-actuator PD gains
    # Actuators: 23 bodies × 3 DOF/body = 69 actuators
    for i in range(model.nu):
        kp = stiffness[i]
        kd = damping[i]
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biastype[i] = 1
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kd
        model.actuator_ctrllimited[i] = 0
        model.actuator_gear[i, :] = np.array([1, 0, 0, 0, 0, 0])  # reset from 500
```

---

## 7. PHYSICS QUALITY METRICS

### File: `scripts/embodied/run_smpl_physics_sim.py`

#### Fall Detection Threshold:
```python
# Line 159
FALL_HEIGHT_THRESHOLD = 0.15  # meters — low enough for deep crouches
```

#### Ground Contact / Offset Handling:
```python
# Lines 471–522
def compute_ground_offset(model, data, ref_qpos: np.ndarray) -> float:
    """Compute vertical offset so feet touch the ground (z=0).
    
    Motion data may be generated for taller/shorter body than MuJoCo model,
    causing feet to float above/penetrate ground even when FK is correct.
    
    Strategy: Scan ALL frames to find global minimum foot z across sequence.
    More robust than frame 0 alone because:
    - Jump motions: frame 0 may be airborne → huge wrong offset
    - Walk motions: lowest foot contact varies over time
    
    Returns:
        ground_offset: float, value to subtract from all qpos[:, 2]
    """
    foot_names = ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]
    foot_bids = []
    for name in foot_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            foot_bids.append(bid)
    
    if not foot_bids:
        return 0.0
    
    T = ref_qpos.shape[0]
    # Sample frames: for efficiency, check at most 30 evenly-spaced frames
    if T <= 30:
        frame_indices = range(T)
    else:
        frame_indices = np.linspace(0, T - 1, 30, dtype=int)
    
    global_min_foot_z = float("inf")
    for t in frame_indices:
        data.qpos[:] = ref_qpos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for bid in foot_bids:
            global_min_foot_z = min(global_min_foot_z, data.xpos[bid, 2])
    
    if global_min_foot_z == float("inf"):
        return 0.0
    
    return float(global_min_foot_z)
```

#### Simulation Statistics Dictionary:
```python
# Lines 704–715
stats = {
    "total_frames": int(T),
    "simulated_frames": int(T_sim),
    "fall_frame": int(fall_frame) if fall_frame is not None else None,
    "completed": fall_frame is None,
    "joint_tracking_error_rad": joint_error,  # Mean |sim - ref| on body joints
    "root_position_drift_m": root_drift,       # |final_root - ref_final|
    "min_root_height_m": float(min_root_h),   # Minimum root z during sim
    "ground_offset_m": float(ground_offset),   # Vertical shift for ground contact
    "fps": fps,
    "decimation": decimation,
}
```

#### Joint Tracking Error Computation:
```python
# Line 697
joint_error = float(np.mean(np.abs(sim_qpos[:, 7:] - ref_qpos[:T_sim, 7:])))
# Mean absolute Euler angle error in radians across all body joints
```

---

## 8. POST-SIMULATION SMOOTHING

### File: `scripts/embodied/run_smpl_physics_sim.py`
**Lines: 720–795**

#### Function Signature:
```python
def smooth_simulated_qpos(sim_qpos: np.ndarray, ref_qpos: np.ndarray,
                          fps: int = 30, window_ms: float = 333.0,
                          blend_alpha: float = 0.5) -> np.ndarray:
    """Post-simulation smoothing to remove PD oscillation artifacts.
    
    Physics sim adds ground-truth contact and prevents penetration, but PD
    tracking introduces high-frequency oscillation (jerk) from discrete control.
    This filter removes oscillation while keeping physics-corrected trajectory.
    
    Strategy: Butterworth low-pass filter (Savitzky-Goyal) on body joints.
    Cutoff frequency removes PD oscillation (5–10 Hz) while preserving motion
    content (< 5 Hz for human motion).
    
    After filtering, blend between filtered sim and kinematic reference:
      result = blend_alpha * filtered_sim + (1 - blend_alpha) * ref
    
    Root (pos + quat) is NOT smoothed — already kinematic.
    
    Args:
        sim_qpos:    (T, 76) simulated qpos from PD tracking
        ref_qpos:    (T, 76) reference qpos (kinematic)
        fps:         control frame rate
        window_ms:   smoothing window in milliseconds (default 333ms)
        blend_alpha: how much physics to keep (1.0 = all physics, 0.0 = all kinematic)
    Returns:
        smoothed_qpos: (T, 76) smoothed qpos
    """
```

#### Key Logic:
```python
T = min(sim_qpos.shape[0], ref_qpos.shape[0])
smoothed = sim_qpos[:T].copy()

# Compute window length in frames (must be odd, >= 3)
window_frames = max(3, int(round(window_ms / 1000.0 * fps)))
if window_frames % 2 == 0:
    window_frames += 1
polyorder = min(3, window_frames - 1)

if T < window_frames:
    # Too short to smooth — blend raw sim with ref
    smoothed[:T, 7:] = blend_alpha * sim_qpos[:T, 7:] + (1 - blend_alpha) * ref_qpos[:T, 7:]
    return smoothed

# Body joints only (indices 7:76), root stays kinematic
sim_body = sim_qpos[:T, 7:].copy()     # (T, 69)
ref_body = ref_qpos[:T, 7:]             # (T, 69)

# Step 1: Savitzky-Golay smooth the simulated body joints directly
smooth_sim = savgol_filter(sim_body, window_frames, polyorder, axis=0)

# Step 2: Blend between smoothed sim and kinematic reference
# This bounds output quality — even if smoothing isn't perfect,
# blend with reference ensures we don't add more jerk than original.
smoothed_body = blend_alpha * smooth_sim + (1 - blend_alpha) * ref_body

smoothed[:T, 7:] = smoothed_body

# Stats: jerk comparison
raw_jerk = np.mean(np.abs(np.diff(sim_body, n=3, axis=0))) * (fps ** 3)
smooth_jerk = np.mean(np.abs(np.diff(smoothed_body, n=3, axis=0))) * (fps ** 3)
ref_jerk = np.mean(np.abs(np.diff(ref_body, n=3, axis=0))) * (fps ** 3)
```

---

## 9. AXIS-ANGLE POSE SMOOTHING (Euler→AA Jitter Fix)

### File: `scripts/embodied/run_smpl_physics_sim.py`
**Lines: 802–890**

#### Function Signature:
```python
def smooth_smpl_poses(smpl_pose: np.ndarray, fps: int = 30,
                      window_ms: float = 333.0) -> np.ndarray:
    """Smooth SMPL axis-angle poses to remove Euler↔AA conversion jitter.
    
    The qpos→SMPL conversion (Euler angles → rotation matrix → axis-angle)
    can amplify small numerical differences into large axis-angle jumps near
    gimbal lock / near-zero rotation angles (where the axis is ill-defined).
    
    We smooth each joint's rotation in quaternion space using SLERP-based
    Savitzky-Golay filtering:
      1. Convert AA → quaternion
      2. Apply SavGol on quat components (with sign flipping for continuity)
      3. Convert back to AA
    
    Uses adaptive windowing: large-rotation joints get wider windows because
    Euler→AA conversion amplifies jitter more when rotations are large.
    
    Multi-pass: applies smoothing twice — first pass removes bulk jitter,
    second pass catches residual oscillation.
    
    Args:
        smpl_pose:   (T, 72) SMPL axis-angle, Z-up
        fps:         frame rate
        window_ms:   base smoothing window in ms (adaptive: large joints get 2x)
    Returns:
        smoothed:    (T, 72) smoothed SMPL axis-angle
    """
```

---

## 10. COMPLETE PIPELINE FLOW

### File: `scripts/embodied/run_smpl_physics_sim.py`
**Lines: 950–1036**

#### Main Processing Function:
```python
def process_single_motion(npz_path: str, xml_path: str, output_dir: str,
                          stats_dir: str = None, fps: int = 30) -> dict:
    """Full pipeline: motion_135 NPZ -> physics sim -> SMPL mesh JSON.
    
    Returns:
        stats dict with simulation results
    """
```

#### Complete Flow (7 stages):
```
[1] decode_motion_135(npz_path)
    → smpl_pose (T, 72) Y-up, transl (T, 3) Y-up, fps
    
[2] yup_to_zup(smpl_pose, transl)
    → smpl_pose_zup (T, 72) Z-up, transl_zup (T, 3) Z-up
    
[3] smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1, model)
    → ref_qpos (T, 76)
    
[3.5] compute_ground_offset(model, data, ref_qpos)
    → ground_offset (float)
    → ref_qpos[:, 2] -= ground_offset (adjust for foot contact)
    
[4] run_physics_sim(model, data, ref_qpos, fps)
    → sim_qpos (T', 76), stats (dict)
    
[4.5] smooth_simulated_qpos(sim_qpos, ref_qpos, fps)
    → sim_qpos smoothed (T', 76)
    
[5] qpos_to_smpl(sim_qpos, body_pos_1)
    → smpl_pose_sim (T', 72) Z-up, transl_sim (T', 3) Z-up
    
[5.3] smooth_smpl_poses(smpl_pose_sim, fps)
    → smpl_pose_sim smoothed (T', 72)
    
[5.5] Undo ground offset: transl_sim[:, 2] += ground_offset
    
[6] zup_to_yup(smpl_pose_sim, transl_sim)
    → smpl_pose_yup (T', 72) Y-up, transl_yup (T', 3) Y-up
    
[7] smpl_to_mesh_json(smpl_pose_yup, transl_yup, fps)
    → result (dict) in web visualizer format
    
Output: JSON file + stats JSON
```

---

## 11. USAGE AS "PHYSICS CORRECTION ORACLE"

### Input / Output Contract:
```python
# INPUT: motion_135 (T, 135)
#   [transl(3) | 22×rot6d(6)]
#   Format: HyMotion row-major rot6d, Y-up coordinates

# OUTPUT: physics-corrected motion_135 (T', 135)
#   Same layout, T' <= T (shorter if fall detected)
#   Physics-enforced:
#     - Foot contact (no penetration)
#     - Reduced foot sliding (PD tracking + ground constraints)
#     - Removed jitter (post-simulation smoothing)

# PROCESSING CHAIN:
process_single_motion(
    npz_path="/path/to/motion_135.npz",
    xml_path="/path/to/smpl_humanoid.xml",
    output_dir="/output/path",
    stats_dir="/stats/path",
    fps=30  # inferred from motion_135 NPZ if not provided
)
# Outputs:
#   - {stem}.json: mesh JSON for web visualization
#   - {stem}.json (in stats_dir): simulation statistics
```

### Key Physics Parameters:
- **PD Gains**: (kp, kd) per body (lines 147–156)
- **Stability**: Critically damped (ζ=1.0) for tracking τ ≈ 0.02s = 0.6 frames
- **Armature**: 0.1 (increased from XML 0.02 for stability)
- **Joint Limits**: Guard axes centered, main axes clamped
- **Ground Offset**: Computed per-motion to ensure foot contact
- **Post-Smoothing**: Savitzky-Golay 333ms window on body joints

### Quality Metrics Output:
```python
stats = {
    "completed": bool,              # True if no fall
    "total_frames": int(T),
    "simulated_frames": int(T_sim),
    "fall_frame": int or None,
    "joint_tracking_error_rad": float,  # Mean |sim - ref| on body joints
    "root_position_drift_m": float,     # |final_root - ref_final|
    "min_root_height_m": float,         # Minimum root z during sim
    "ground_offset_m": float,            # Vertical shift applied
    "fps": int,
    "decimation": int,                   # Mujoco substeps per control frame
}
```

---

## SUMMARY TABLE

| Component | File | Lines | Function | Key Output |
|-----------|------|-------|----------|-----------|
| Motion decode | `run_smpl_physics_sim.py` | 193–227 | `decode_motion_135()` | (T,72) AA + (T,3) transl, Y-up |
| Rot6d utils | `motion135_to_smplx.py` | 26–67 | `rot6d_to_rotmat()`, `rotmat_to_axis_angle()` | rot6d → R → AA |
| Y↔Z transform | `run_smpl_physics_sim.py` | 234–314 | `yup_to_zup()`, `zup_to_yup()` | Cyclic permutation of axes |
| SMPL→qpos | `run_smpl_physics_sim.py` | 321–426 | `smpl_to_qpos()` | (T,76) qpos with joint limit handling |
| qpos→SMPL | `run_smpl_physics_sim.py` | 429–464 | `qpos_to_smpl()` | (T,72) AA + (T,3) transl, Z-up |
| Physics sim loop | `run_smpl_physics_sim.py` | 609–717 | `run_physics_sim()` | (T',76) simulated qpos + stats |
| Ground offset | `run_smpl_physics_sim.py` | 471–522 | `compute_ground_offset()` | float, Z adjustment for foot contact |
| Qpos smoothing | `run_smpl_physics_sim.py` | 720–795 | `smooth_simulated_qpos()` | (T,76) smoothed qpos |
| AA smoothing | `run_smpl_physics_sim.py` | 802–890 | `smooth_smpl_poses()` | (T,72) jitter-reduced AA |
| Model config | `run_smpl_physics_sim.py` | 525–606 | `load_mujoco_model()` | model + data with PD gains |
| Main pipeline | `run_smpl_physics_sim.py` | 950–1036 | `process_single_motion()` | JSON + stats for full pipeline |

