#!/usr/bin/env python3
"""Run MuJoCo physics simulation on SMPL humanoid to enforce physical constraints.

Fixes foot sliding, ground penetration, and jitter by running PD-tracking simulation
on a full SMPL humanoid in MuJoCo, letting the physics engine enforce ground contact
and physical plausibility.

Pipeline:
  motion_135 NPZ (T, 135) -- Y-up, HyMotion format
    | [1] Decode rot6d -> axis-angle (reuse from motion135_to_smplx.py)
    | [2] Y-up -> Z-up coordinate transform
    | [3] SMPL axis-angle (72-dim) -> MuJoCo qpos (76-dim)
    | [4] MuJoCo PD simulation
    | [5] Export simulated qpos -> SMPL axis-angle
    | [6] Z-up -> Y-up reverse transform
    | [7] Generate SMPL mesh JSON for website

Usage:
    # Single file
    python3 scripts/embodied/run_smpl_physics_sim.py \
        --npz-file output/embodied_t2m_v4/data/npz/walk_forward.npz \
        --output-dir output/embodied_t2m_v4/data/smpl_mesh_physics \
        --xml-path ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml

    # Batch (with flat-ground filter)
    python3 scripts/embodied/run_smpl_physics_sim.py \
        --npz-dir output/embodied_t2m_v4/data/npz \
        --output-dir output/embodied_t2m_v4/data/smpl_mesh_physics \
        --xml-path ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
        --meta-dir output/embodied_t2m_v4/data/meta \
        --stats-dir output/embodied_t2m_v4/data/sim_stats \
        --filter-flat-ground
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as sRot


# ===========================================================================
#  Constants
# ===========================================================================

# SMPL joint names (24 joints, standard SMPL order)
SMPL_JOINT_NAMES = [
    "Pelvis",       # 0  (root)
    "L_Hip",        # 1
    "R_Hip",        # 2
    "Spine1",       # 3
    "L_Knee",       # 4
    "R_Knee",       # 5
    "Spine2",       # 6
    "L_Ankle",      # 7
    "R_Ankle",      # 8
    "Spine3",       # 9
    "L_Foot",       # 10
    "R_Foot",       # 11
    "Neck",         # 12
    "L_Collar",     # 13
    "R_Collar",     # 14
    "Head",         # 15
    "L_Shoulder",   # 16
    "R_Shoulder",   # 17
    "L_Elbow",      # 18
    "R_Elbow",      # 19
    "L_Wrist",      # 20
    "R_Wrist",      # 21
    "L_Hand",       # 22
    "R_Hand",       # 23
]

# MuJoCo body names in depth-first tree order (from smpl_humanoid.xml)
MUJOCO_BODY_NAMES = [
    "Pelvis",       # 0  (root — free joint, not actuated)
    "L_Hip",        # 1
    "L_Knee",       # 2
    "L_Ankle",      # 3
    "L_Toe",        # 4   (= SMPL L_Foot)
    "R_Hip",        # 5
    "R_Knee",       # 6
    "R_Ankle",      # 7
    "R_Toe",        # 8   (= SMPL R_Foot)
    "Torso",        # 9   (= SMPL Spine1)
    "Spine",        # 10  (= SMPL Spine2)
    "Chest",        # 11  (= SMPL Spine3)
    "Neck",         # 12
    "Head",         # 13
    "L_Thorax",     # 14  (= SMPL L_Collar)
    "L_Shoulder",   # 15
    "L_Elbow",      # 16
    "L_Wrist",      # 17
    "L_Hand",       # 18
    "R_Thorax",     # 19  (= SMPL R_Collar)
    "R_Shoulder",   # 20
    "R_Elbow",      # 21
    "R_Wrist",      # 22
    "R_Hand",       # 23
]

# Name mapping: MuJoCo XML name -> SMPL bone name (only for mismatched names)
_MUJOCO_TO_SMPL_NAME = {
    "Torso": "Spine1", "Spine": "Spine2", "Chest": "Spine3",
    "L_Toe": "L_Foot", "R_Toe": "R_Foot",
    "L_Thorax": "L_Collar", "R_Thorax": "R_Collar",
}


def _build_reorder_indices():
    """Build smpl_2_mujoco and mujoco_2_smpl reorder arrays.

    smpl_2_mujoco[i] = SMPL non-root joint index for the i-th MuJoCo non-root body.
    mujoco_2_smpl[i] = MuJoCo non-root body index for the i-th SMPL non-root joint.
    """
    smpl_names = SMPL_JOINT_NAMES[1:]      # 23 non-root joints
    mj_names = MUJOCO_BODY_NAMES[1:]       # 23 non-root bodies

    # For each MuJoCo body, find its index in the SMPL list
    s2m = []
    for mj_name in mj_names:
        smpl_name = _MUJOCO_TO_SMPL_NAME.get(mj_name, mj_name)
        s2m.append(smpl_names.index(smpl_name))

    # Inverse mapping
    m2s = [0] * 23
    for mj_idx, smpl_idx in enumerate(s2m):
        m2s[smpl_idx] = mj_idx

    return s2m, m2s


SMPL_2_MUJOCO, MUJOCO_2_SMPL = _build_reorder_indices()
# Verified:
# SMPL_2_MUJOCO = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 14, 12, 15, 17, 19, 21, 13, 16, 18, 20, 22]
# MUJOCO_2_SMPL = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 13, 18, 12, 14, 19, 15, 20, 16, 21, 17, 22]

# PD gains per MuJoCo body (kp, kd)
# With dof_damping zeroed, effective kd = PD kd only.
# Stability requires: ω = √(kp/armature) < 1/dt → kp < armature/dt² ≈ 649
# (armature=0.02, dt=0.00555s). Keeping original PHC gains (kp≤1000) is safe
# because armature is also increased to 0.1 for the highest-gain joints.
#
# Tracking: τ = kd/kp. For kp=500,kd=50: τ=0.1s=3frames. Acceptable for walking.
# Critical damping: ζ = kd/(2√(kp*armature)). With armature=0.1:
#   kp=1000,kd=20 → ζ=1.0 (critically damped), τ=0.02s (0.6 frames) — FAST
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

# Fall detection
FALL_HEIGHT_THRESHOLD = 0.15  # meters — low enough for deep crouches

# Flat-ground case filtering
EXCLUDE_KEYWORDS = [
    "stair", "stairs", "step up", "step down", "climb", "box", "platform",
    "jump on", "jump off", "ledge", "obstacle", "hurdle", "ladder",
    "上楼", "下楼", "台阶", "箱子", "跳上", "跳下", "攀爬",
]


# ===========================================================================
#  Motion Decoding (from motion_135 NPZ)
# ===========================================================================

def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Convert HyMotion row-major rot6d (..., 6) to rotation matrix (..., 3, 3).

    HyMotion stores rot6d in row-major layout: [R00,R01, R10,R11, R20,R21].
    Gram-Schmidt expects column-major: [R00,R10,R20, R01,R11,R21].
    Reorder [0,2,4,1,3,5] to convert row-major -> column-major before decoding.
    """
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)


def decode_motion_135(npz_path: str):
    """Decode motion_135 NPZ to SMPL 72-dim axis-angle pose + translation (Y-up).

    motion_135 format: (T, 135) = transl(3) + 22 x rot6d(6).
    SMPL expects 24 joints; joints 22-23 (L_Hand, R_Hand) are zero-padded.

    Returns:
        smpl_pose: (T, 72) axis-angle in SMPL joint order, Y-up
        transl:    (T, 3)  translation, Y-up
        fps:       int
    """
    data = np.load(npz_path, allow_pickle=True)
    motion = data["motion_135"]  # (T, 135)
    fps = int(data.get("fps", 30))
    T = motion.shape[0]

    transl = motion[:, :3]                        # (T, 3)
    rot6d = motion[:, 3:].reshape(T, 22, 6)       # (T, 22, 6)

    # rot6d -> rotation matrix -> axis-angle
    rotmat = rot6d_to_rotmat(rot6d)                # (T, 22, 3, 3)
    aa = sRot.from_matrix(
        rotmat.reshape(-1, 3, 3)
    ).as_rotvec().reshape(T, 22, 3)                # (T, 22, 3)

    root_orient = aa[:, 0, :]                      # (T, 3) — joint 0
    body_pose = aa[:, 1:22, :].reshape(T, -1)      # (T, 63) — joints 1-21

    # Build full 72-dim SMPL pose (pad joints 22-23 with zeros)
    smpl_pose = np.zeros((T, 72), dtype=np.float32)
    smpl_pose[:, :3] = root_orient
    smpl_pose[:, 3:66] = body_pose
    # smpl_pose[:, 66:72] = 0  (L_Hand, R_Hand)

    return smpl_pose, transl.astype(np.float32), fps


# ===========================================================================
#  Coordinate Transforms (Y-up <-> Z-up)
# ===========================================================================

# Cyclic permutation: SMPL Y-up -> MuJoCo Z-up.
#   SMPL axes:   X=left, Y=up, Z=forward  (standard SMPL, L_Hip at +X)
#   MuJoCo axes: X=forward, Y=left, Z=up  (from smpl_humanoid.xml body offsets)
# Mapping:  SMPL_X(left)->MJ_Y(left), SMPL_Y(up)->MJ_Z(up), SMPL_Z(fwd)->MJ_X(fwd)
# [x,y,z]_yup -> [z, x, y]_zup  i.e.  x_zup=z_yup, y_zup=x_yup, z_zup=y_yup
_YUP_TO_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
# Inverse cyclic permutation: MuJoCo Z-up -> SMPL Y-up.
# [x,y,z]_zup -> [y, z, x]_yup  i.e.  x_yup=y_zup, y_yup=z_zup, z_yup=x_zup
_ZUP_TO_YUP = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)


def yup_to_zup(smpl_pose: np.ndarray, transl: np.ndarray):
    """Transform SMPL pose + translation from Y-up to Z-up.

    SMPL Y-up: X=left, Y=up, Z=forward
    MuJoCo Z-up: X=forward, Y=left, Z=up

    Cyclic permutation: [x,y,z]_yup → [z,x,y]_zup
      SMPL X(left) → MuJoCo Y(left)
      SMPL Y(up) → MuJoCo Z(up)
      SMPL Z(fwd) → MuJoCo X(fwd)

    ALL joints (root + body) need the coordinate transform because:

    - Root orientation is in the global frame (Y-up → Z-up).
    - Body joint axis-angles are in LOCAL body frames. In the T-pose (rest pose),
      all body local frames align with the global frame. SMPL body local frames
      are Y-up; MuJoCo body local frames are Z-up. They differ by the same cyclic
      permutation as the global frames.

    Concretely: SMPL knee flexion is around local X (lateral). Without transform,
    this maps to MuJoCo's X-joint (forward axis, ±5.6° guard). With transform,
    it correctly maps to MuJoCo's Y-joint (lateral axis, [0°,180°] flexion).

    Note: PHC's smpl_to_qpose() does NOT apply this transform because PHC uses
    a learned RL policy (not PD position control). The policy adapts to whatever
    axis mapping exists. For PD tracking (ctrl[:] = ref_qpos[7:]), the Euler
    angle slots MUST match the physical joint axes.
    """
    T = smpl_pose.shape[0]

    # Translation: [x,y,z]_yup → [z,x,y]_zup
    out_transl = (transl.astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32)

    # Transform ALL joint axis-angles: root (0:3) + body (3:72)
    # Each joint's axis-angle is a 3D vector; cyclic permutation of axes
    pose_72 = smpl_pose[:, :72].astype(np.float64)  # (T, 72)
    pose_72_3d = pose_72.reshape(T * 24, 3)  # (T*24, 3)
    pose_72_zup = (pose_72_3d @ _YUP_TO_ZUP.T).reshape(T, 72)
    out_pose = smpl_pose.copy()
    out_pose[:, :72] = pose_72_zup.astype(np.float32)

    return out_pose, out_transl


def zup_to_yup(smpl_pose: np.ndarray, transl: np.ndarray):
    """Transform SMPL pose + translation from Z-up to Y-up.

    MuJoCo Z-up: X=forward, Y=left, Z=up
    SMPL Y-up: X=left, Y=up, Z=forward

    Inverse cyclic permutation: [x,y,z]_zup → [y,z,x]_yup
      MuJoCo X(fwd) → SMPL Z(fwd)
      MuJoCo Y(left) → SMPL X(left)
      MuJoCo Z(up) → SMPL Y(up)

    Inverse of yup_to_zup. Transforms ALL joints (root + body).
    """
    T = smpl_pose.shape[0]

    # Translation: [x,y,z]_zup → [y,z,x]_yup
    out_transl = (transl.astype(np.float64) @ _ZUP_TO_YUP.T).astype(np.float32)

    # Transform ALL joint axis-angles back: root (0:3) + body (3:72)
    pose_72 = smpl_pose[:, :72].astype(np.float64)  # (T, 72)
    pose_72_3d = pose_72.reshape(T * 24, 3)  # (T*24, 3)
    pose_72_yup = (pose_72_3d @ _ZUP_TO_YUP.T).reshape(T, 72)
    out_pose = smpl_pose.copy()
    out_pose[:, :72] = pose_72_yup.astype(np.float32)

    return out_pose, out_transl


# ===========================================================================
#  SMPL <-> MuJoCo qpos Conversion
# ===========================================================================

def smpl_to_qpos(smpl_pose: np.ndarray, transl: np.ndarray,
                 body_pos_1: np.ndarray,
                 model=None) -> np.ndarray:
    """Convert SMPL 72-dim axis-angle pose to MuJoCo 76-dim qpos.

    qpos layout: [root_trans(3), root_quat_wxyz(4), joint_euler_xyz(69)]

    Euler convention for PD tracking:
      MuJoCo smpl_humanoid.xml has <compiler coordinate="local"/> with hinge joints
      ordered X, Y, Z per body. With local coordinates, the composition is intrinsic
      XYZ: R = Rx(theta_x) * Ry(theta_y) * Rz(theta_z).

      We use as_euler("xyz") (intrinsic XYZ) which outputs [x, y, z] angles,
      directly matching qpos slots [X_joint, Y_joint, Z_joint].

      NOTE: PHC reference uses as_euler("ZYX") (extrinsic ZYX = same decomposition),
      but that outputs [z, y, x] — swapping X/Z in qpos slots. PHC doesn't care because
      it uses a learned RL policy (not PD position control). For direct PD tracking
      (ctrl[:] = ref_qpos[7:]), the slot ordering MUST match the physical joint axes.

    Joint limit handling (when model is provided):
      Euler angle decomposition of large rotations (e.g., deep knee bends) can spread
      rotation onto "guard" axes — e.g., knee X/Z have ±5.6° limits for tiny lateral
      wobble, but Euler decomposition may put ±180° there. Two-tier fix:
        1. Guard axes (range < 15°): Set PD target to center of range. The PD won't
           fight against joint stops, eliminating chatter. Loss: ≤5.6° on minor axis.
        2. Main axes (range ≥ 15°): Clamp to joint limits to prevent impossible targets.

    Args:
        smpl_pose:  (T, 72) axis-angle, Z-up root / local body joints, SMPL joint order
        transl:     (T, 3)  translation, Z-up
        body_pos_1: (3,)    Pelvis body position offset from MuJoCo XML
        model:      optional MuJoCo model (for joint limit clamping)
    Returns:
        qpos: (T, 76) float64
    """
    T = smpl_pose.shape[0]
    qpos = np.zeros((T, 76), dtype=np.float64)

    joint_aa = smpl_pose.reshape(T, 24, 3)  # (T, 24, 3)

    # Root translation: add Pelvis body_pos offset (per smpl_to_qpose convention)
    qpos[:, :3] = transl.astype(np.float64) + body_pos_1

    # Root orientation: axis-angle -> quaternion wxyz
    root_quat_xyzw = sRot.from_rotvec(joint_aa[:, 0]).as_quat()  # (T, 4) xyzw
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]               # -> wxyz

    # Body joints (1-23): axis-angle -> intrinsic XYZ Euler angles
    # as_euler("xyz") = intrinsic XYZ, output order [x, y, z]
    # This matches qpos slots [X_joint, Y_joint, Z_joint] for PD tracking.
    body_aa = joint_aa[:, 1:].reshape(-1, 3)               # (T*23, 3)
    body_euler = sRot.from_rotvec(body_aa).as_euler("xyz")  # (T*23, 3) = [x, y, z]
    body_euler = body_euler.reshape(T, 23, 3)               # (T, 23, 3) SMPL order

    # Reorder from SMPL joint order to MuJoCo tree order
    body_euler_mj = body_euler[:, SMPL_2_MUJOCO]           # (T, 23, 3)
    qpos[:, 7:] = body_euler_mj.reshape(T, 69)

    # Joint limit handling (prevents impossible PD targets + guard-axis chatter)
    #
    # Problem: Euler decomposition of large rotations (deep knee bends, crouches)
    # spreads rotation onto "guard" axes — narrow-limit joints (±5.6°) that exist
    # for tiny lateral wobble. These get values like ±180° which clamp to the limit
    # boundary. The PD controller then "chatters" against the joint stop because
    # it persistently targets the limit boundary.
    #
    # Solution: Two-tier approach:
    #   1. Guard axes (range < 15°): Set PD target to center of range (usually 0).
    #      These axes have negligible rotational contribution (max ±5.6°), so
    #      centering loses at most ~5.6° on a minor axis — visually imperceptible.
    #      This eliminates chatter entirely.
    #   2. Main axes (range ≥ 15°): Clamp to joint limits as before.
    #
    GUARD_AXIS_THRESHOLD = np.radians(15.0)  # joints with range < 15° are "guard"
    if model is not None:
        n_clamped = 0
        n_centered = 0
        for jid in range(model.njnt):
            if model.jnt_type[jid] != 3:  # only hinge joints
                continue
            if not model.jnt_limited[jid]:
                continue
            qi = model.jnt_qposadr[jid]
            lo, hi = model.jnt_range[jid]
            joint_range = hi - lo
            center = (lo + hi) / 2.0

            if joint_range < GUARD_AXIS_THRESHOLD:
                # Guard axis: center the target to avoid chatter at joint stops
                n_not_center = int(np.sum(qpos[:, qi] != center))
                qpos[:, qi] = center
                if n_not_center > 0:
                    n_centered += n_not_center
            else:
                # Main axis: clamp to joint limits
                before = qpos[:, qi].copy()
                qpos[:, qi] = np.clip(qpos[:, qi], lo, hi)
                n_violations = int(np.sum(before != qpos[:, qi]))
                if n_violations > 0:
                    n_clamped += n_violations
        if n_clamped > 0 or n_centered > 0:
            print(f"  Joint limit fix: {n_clamped} clamped (main axes), "
                  f"{n_centered} centered (guard axes) across {T} frames")

    return qpos


def qpos_to_smpl(qpos: np.ndarray, body_pos_1: np.ndarray):
    """Convert MuJoCo 76-dim qpos to SMPL 72-dim axis-angle pose.

    Inverse of smpl_to_qpos(). Uses "xyz" (intrinsic XYZ) Euler convention.
    qpos slots [X_joint, Y_joint, Z_joint] → from_euler("xyz") reads [x, y, z].

    Args:
        qpos:       (T, 76) float64
        body_pos_1: (3,)    Pelvis body position offset from MuJoCo XML
    Returns:
        smpl_pose: (T, 72) axis-angle, Z-up root / local body joints, SMPL joint order
        transl:    (T, 3)  translation, Z-up
    """
    T = qpos.shape[0]

    # Root translation: undo Pelvis body_pos offset
    transl = (qpos[:, :3] - body_pos_1).astype(np.float32)

    # Root orientation: quaternion wxyz -> axis-angle
    root_quat_wxyz = qpos[:, 3:7]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]   # -> xyzw
    root_aa = sRot.from_quat(root_quat_xyzw).as_rotvec()  # (T, 3)

    # Body joints: intrinsic XYZ Euler -> axis-angle
    # qpos stores [x, y, z] per body → from_euler("xyz") interprets correctly
    body_euler_mj = qpos[:, 7:].reshape(T, 23, 3)         # MuJoCo tree order
    body_euler_smpl = body_euler_mj[:, MUJOCO_2_SMPL]      # -> SMPL order
    body_aa = sRot.from_euler(
        "xyz", body_euler_smpl.reshape(-1, 3)
    ).as_rotvec().reshape(T, 23, 3)

    smpl_pose = np.zeros((T, 72), dtype=np.float32)
    smpl_pose[:, :3] = root_aa.astype(np.float32)
    smpl_pose[:, 3:] = body_aa.reshape(T, 69).astype(np.float32)

    return smpl_pose, transl


# ===========================================================================
#  MuJoCo Physics Simulation
# ===========================================================================

def compute_ground_offset(model, data, ref_qpos: np.ndarray) -> float:
    """Compute vertical offset so feet touch the ground (z=0).

    The motion data may have been generated for a taller/shorter body than the
    MuJoCo model. This causes feet to float above or penetrate below the ground
    even when FK is correct.

    We scan ALL frames to find the global minimum foot z across the sequence.
    This is more robust than using only frame 0, because:
    - For jump motions, frame 0 may be airborne → huge (wrong) offset
    - For walk motions, the lowest foot contact varies over time

    The global minimum foot z represents the true "ground contact" height.

    Args:
        model:    MuJoCo model
        data:     MuJoCo data
        ref_qpos: (T, 76) full reference qpos sequence
    Returns:
        ground_offset: float, value to subtract from all qpos[:, 2]
    """
    import mujoco

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


def load_mujoco_model(xml_path: str):
    """Load and configure MuJoCo SMPL humanoid model for PD-tracking physics sim.

    Strategy: Zero ALL passive dynamics (stiffness, damping, friction) so that
    only the PD actuators drive joint motion. Keep armature for numerical stability.

    XML defaults we OVERRIDE:
      - dof_damping = 80  → 0  (was causing extreme overdamping: effective ζ=20.6!)
      - jnt_stiffness = 800 → 0  (pulls toward T-pose, fights PD tracking)
      - dof_frictionloss → 0

    XML defaults we KEEP:
      - dof_armature = 0.02  (adds effective inertia, helps numerical stability)

    Total joint force = PD_actuator only:
      = kp*(target - qpos) - kd*qvel

    Returns:
        model: mujoco.MjModel
        data:  mujoco.MjData
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Zero ALL passive dynamics — let PD actuators be the sole force source.
    # Previously dof_damping was kept at 80, which combined with PD kd created
    # total effective damping of kd+80, making joints extremely sluggish
    # (ζ=20.6, τ=0.26s=7.8 frames to converge — could not track motion).
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0      # CRITICAL FIX: was 80, caused overdamping
    model.dof_frictionloss[:] = 0.0

    # Increase armature for numerical stability with higher PD gains.
    # Stability: ω = √(kp/armature), need dt < 2/ω → armature > kp*dt²/4
    # For kp=2000, dt=0.00555: armature > 2000*0.00555²/4 = 0.0154 → 0.1 is safe
    # Also achieves near-critical damping: ζ = kd/(2√(kp*armature))
    # For kp=1000,kd=20,armature=0.1: ζ = 20/(2*√100) = 1.0 (perfect!)
    model.dof_armature[6:] = 0.1    # body joints only (skip root free joint DOFs 0-5)

    # Build per-actuator stiffness/damping arrays
    # Actuator order follows joint order = 3 hinge joints (x,y,z) per body in tree order
    mj_bodies = MUJOCO_BODY_NAMES[1:]  # 23 non-root bodies
    stiffness = []
    damping = []
    for body_name in mj_bodies:
        kp, kd = PD_GAINS_PER_BODY[body_name]
        for _ in range(3):  # 3 DOF per body
            stiffness.append(float(kp))
            damping.append(float(kd))

    assert model.nu == len(stiffness), (
        f"Actuator count mismatch: model.nu={model.nu}, expected={len(stiffness)}"
    )

    # Configure implicit PD actuators
    # force_i = gear * (gainprm[0] * ctrl + biasprm[0] + biasprm[1]*qpos + biasprm[2]*qvel)
    # With gear=1, gainprm=kp, biasprm=[0,-kp,-kd]:
    #   force_i = kp * (ctrl_i - qpos_i) - kd * qvel_i
    #
    # IMPORTANT: The XML default has gear="500" which multiplies actuator output.
    # We must reset gear to 1 since our PD gains already define the correct force magnitudes.
    for i in range(model.nu):
        kp = stiffness[i]
        kd = damping[i]
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biastype[i] = 1       # affine bias
        model.actuator_biasprm[i, 0] = 0.0   # intercept
        model.actuator_biasprm[i, 1] = -kp   # coeff on qpos
        model.actuator_biasprm[i, 2] = -kd   # coeff on qvel
        model.actuator_ctrllimited[i] = 0     # no control limits
        model.actuator_gear[i, :] = np.array([1, 0, 0, 0, 0, 0])  # reset gear from 500 to 1

    print(f"  MuJoCo model loaded: {model.nbody} bodies, {model.nu} actuators, "
          f"nq={model.nq}, nv={model.nv}, dt={model.opt.timestep:.5f}s")
    print(f"  Passive: dof_damping={model.dof_damping[6]:.0f} (zeroed), "
          f"dof_armature={model.dof_armature[6]:.3f} (set to 0.1)")
    print(f"  PD example (L_Hip_x): kp={model.actuator_gainprm[0,0]:.0f}, "
          f"kd={-model.actuator_biasprm[0,2]:.0f}")

    return model, data


def run_physics_sim(model, data, ref_qpos: np.ndarray, fps: int = 30,
                    root_mode: str = "free"):
    """Run PD-tracking physics simulation with physically-grounded root motion.

    Root joint strategy (root_mode="free"):
        The root joint is FREE — no kinematic tracking. Only body joints are
        PD-controlled via actuators. Root translation and rotation emerge
        entirely from physics: ground contact forces, gravity, friction, and
        the torques transmitted through the kinematic chain from PD-controlled
        joints.

        This matches the approach used in DeepMimic, PHC, UHC, etc. where root
        motion is a CONSEQUENCE of the body's interaction with the environment,
        not an externally-imposed trajectory.

        A weak "virtual spring" on root orientation (applied via xfrc_applied)
        prevents catastrophic early-frame toppling while being too weak to
        override physics-driven motion. This decays over the first ~0.5s.

    Root joint strategy (root_mode="kinematic", legacy):
        Root is reset to reference each frame. Only for comparison/debugging.

    Args:
        model:    MuJoCo model (configured with PD actuators)
        data:     MuJoCo data
        ref_qpos: (T, 76) reference qpos trajectory
        fps:      control frame rate
        root_mode: "free" (physically-grounded) or "kinematic" (legacy)
    Returns:
        sim_qpos: (T', 76) simulated qpos (T' <= T, shorter if fall detected)
        stats:    dict with simulation statistics
    """
    import mujoco
    from scipy.spatial.transform import Rotation as sRot

    T = ref_qpos.shape[0]
    sim_dt = model.opt.timestep
    ctrl_dt = 1.0 / fps
    decimation = max(1, int(round(ctrl_dt / sim_dt)))

    print(f"  sim_dt={sim_dt:.5f}s, ctrl_dt={ctrl_dt:.4f}s, decimation={decimation}, "
          f"root_mode={root_mode}")

    # Initialize with first frame — root state set ONLY here
    data.qpos[:] = ref_qpos[0]
    # Give initial root velocity from reference trajectory (frames 0→1)
    data.qvel[:] = 0.0
    if T > 1:
        # Initial linear velocity
        data.qvel[:3] = (ref_qpos[1, :3] - ref_qpos[0, :3]) / ctrl_dt
        # Initial angular velocity
        q_cur = ref_qpos[0, 3:7][[1, 2, 3, 0]]   # wxyz -> xyzw
        q_next = ref_qpos[1, 3:7][[1, 2, 3, 0]]
        R_diff = sRot.from_quat(q_cur).inv() * sRot.from_quat(q_next)
        data.qvel[3:6] = R_diff.as_rotvec() / ctrl_dt
    mujoco.mj_forward(model, data)

    sim_qpos_list = []
    fall_frame = None
    min_root_h = float("inf")

    # ─────────────────────────────────────────────────────────────────────
    # Assistive force parameters (free mode)
    # ─────────────────────────────────────────────────────────────────────
    # Unlike kinematic forcing (which teleports the root), assistive forces
    # apply a PHYSICAL spring+damper at the root body. This is the standard
    # approach in biomechanics (OpenSim residual actuators) and physics-based
    # motion imitation without RL (DeepMimic ablations, MuJoCo tutorials).
    #
    # The root can still:
    #   - Sag under gravity (spring is not infinitely stiff)
    #   - Bounce on ground contact (forces propagate through physics)
    #   - Deviate from reference (observable in output)
    #   - Interact naturally with contact forces
    #
    # The spring strength is set so that:
    #   - Position: Body mass ~40kg, need ~400N to counteract gravity.
    #     kp=2000 N/m → 0.2m error produces 400N (gravity compensation).
    #     This allows ~5-10cm of sag which is visibly physical.
    #   - Orientation: kp=800 Nm/rad → enough to prevent toppling,
    #     weak enough to allow natural body lean (5-10 deg deviation).
    #
    # Estimated body mass from SMPL humanoid: ~40-80 kg depending on shape.
    # mg ≈ 600N. Spring at rest = 0 force. To support weight, CoM must
    # sag by mg/kp ≈ 600/2000 = 0.3m. This is too much for position.
    # Solution: Use a stronger position spring but allow HORIZONTAL drift.
    # ─────────────────────────────────────────────────────────────────────

    # Estimate body mass from model
    total_mass = float(sum(model.body_mass))
    gravity_force = total_mass * 9.81  # N

    # ─── Assistive force design ───
    # The goal is NOT to support full body weight (that would be kinematic tracking).
    # The goal is gentle guidance that:
    #   1. Prevents catastrophic toppling in the first few frames
    #   2. Provides a weak orientation reference (like balance vestibular sense)
    #   3. Allows the humanoid's dynamics to dominate (ground contact, gravity, inertia)
    #
    # Key insight from experiments:
    #   - Pure PD (no force) is stable for ~20-25 frames, then slowly topples
    #   - The toppling is due to imperfect PD tracking accumulating lean
    #   - We need just enough force to counteract this slow lean, NOT to track reference
    #
    # Design: Very weak springs that apply <10% of body weight in force.
    # A 74kg body has mg≈726N. We want max ~50-70N from the spring.
    # With typical position error of 0.1-0.2m: kp*0.15 ≈ 50N → kp ≈ 330 N/m
    #
    # Orientation spring: main anti-toppling mechanism. Provides a restoring
    # torque when the body leans. At 10 deg lean: torque = kp*0.17 ≈ 50 Nm.
    # This is comparable to ankle torque from a human maintaining balance.
    ROOT_POS_KP_XY = 200.0    # N/m — weak horizontal (allows drift, physical movement)
    ROOT_POS_KP_Z = 1200.0    # N/m — moderate-strong vertical (fights height loss)
                               # At 5cm error: 1200*0.05 = 60N (~8% of body weight)
                               # Prevents collapse while still allowing natural sag
    ROOT_POS_KD = 60.0         # N*s/m — light damping (prevents oscillation)

    # Orientation spring — primary balance mechanism
    ROOT_ORI_KP = 300.0    # Nm/rad — at 10deg lean: 300*0.17 = 51 Nm (moderate)
    ROOT_ORI_KD = 30.0     # Nm*s/rad — gentle orientation damping

    # Decay: keep constant assistance (no decay). The forces are weak enough
    # that they don't override physics-driven motion, and removing them causes
    # late falls. A constant weak spring is the standard approach in OpenSim
    # "residual actuators" for motion reconstruction.
    # NO decay — constant weak assistance throughout the motion.
    ASSIST_RAMPUP_FRAMES = int(fps * 0.2)  # 0.2s ramp-up from zero (avoids impulse)

    print(f"  Assistive forces: mass={total_mass:.1f}kg, mg={gravity_force:.0f}N")
    print(f"    pos_kp_xy={ROOT_POS_KP_XY:.0f}, pos_kp_z={ROOT_POS_KP_Z:.0f}, "
          f"pos_kd={ROOT_POS_KD:.0f}")
    print(f"    ori_kp={ROOT_ORI_KP:.0f}, ori_kd={ROOT_ORI_KD:.0f}, "
          f"rampup={ASSIST_RAMPUP_FRAMES}f")

    for t in range(T):
        if root_mode == "kinematic":
            # Legacy mode: force root to reference (for comparison only)
            data.qpos[:7] = ref_qpos[t, :7]
            if t + 1 < T:
                data.qvel[:3] = (ref_qpos[t + 1, :3] - ref_qpos[t, :3]) / ctrl_dt
                q_cur = ref_qpos[t, 3:7][[1, 2, 3, 0]]
                q_next = ref_qpos[t + 1, 3:7][[1, 2, 3, 0]]
                R_diff = sRot.from_quat(q_cur).inv() * sRot.from_quat(q_next)
                data.qvel[3:6] = R_diff.as_rotvec() / ctrl_dt
            else:
                data.qvel[:6] = 0.0
        else:
            # ── FREE ROOT MODE with weak assistive spring forces ──
            # Ramp-up from 0 to full over first few frames (avoids initial impulse)
            if t < ASSIST_RAMPUP_FRAMES:
                scale = t / max(ASSIST_RAMPUP_FRAMES, 1)
            else:
                scale = 1.0

            # ── Position spring force ──
            pos_error = ref_qpos[t, :3] - data.qpos[:3]  # reference - current
            force_x = scale * (ROOT_POS_KP_XY * pos_error[0] - ROOT_POS_KD * data.qvel[0])
            force_y = scale * (ROOT_POS_KP_XY * pos_error[1] - ROOT_POS_KD * data.qvel[1])
            force_z = scale * (ROOT_POS_KP_Z * pos_error[2] - ROOT_POS_KD * data.qvel[2])

            # ── Orientation spring torque ──
            q_sim = data.qpos[3:7][[1, 2, 3, 0]]   # wxyz -> xyzw
            q_ref = ref_qpos[t, 3:7][[1, 2, 3, 0]]
            R_err = sRot.from_quat(q_sim).inv() * sRot.from_quat(q_ref)
            ori_error = R_err.as_rotvec()  # (3,) axis-angle error
            torque = scale * (ROOT_ORI_KP * ori_error - ROOT_ORI_KD * data.qvel[3:6])

            # Apply to root body (body index 1 = Pelvis)
            # xfrc_applied: (nbody, 6) = [fx, fy, fz, tx, ty, tz]
            data.xfrc_applied[1, 0] = force_x
            data.xfrc_applied[1, 1] = force_y
            data.xfrc_applied[1, 2] = force_z
            data.xfrc_applied[1, 3:6] = torque

        # ---- PD targets for body joints ----
        data.ctrl[:] = ref_qpos[t, 7:]

        # Step physics (decimation sub-steps per control frame)
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        # Clear external forces after stepping (so they don't accumulate)
        if root_mode == "free":
            data.xfrc_applied[1, :] = 0.0

        sim_qpos_list.append(data.qpos.copy())

        # Track root height
        root_h = float(data.qpos[2])
        min_root_h = min(min_root_h, root_h)

        # Fall detection — with free root, falls are expected for difficult motions
        if root_h < FALL_HEIGHT_THRESHOLD or np.any(np.isnan(data.qpos)):
            fall_frame = t
            reason = "NaN" if np.any(np.isnan(data.qpos)) else f"root_h={root_h:.3f}m"
            print(f"  FALL at frame {t}/{T}: {reason}")
            break

    sim_qpos = np.array(sim_qpos_list)
    T_sim = len(sim_qpos)

    # Compute tracking error (mean absolute joint angle error in radians)
    joint_error = float(np.mean(np.abs(sim_qpos[:, 7:] - ref_qpos[:T_sim, 7:])))

    # Compute root position drift (with free root, this will be significant)
    root_drift = float(np.linalg.norm(
        sim_qpos[-1, :3] - ref_qpos[min(T_sim - 1, T - 1), :3]
    ))

    # Compute mean root translation difference per frame
    root_pos_sim = sim_qpos[:, :3]
    root_pos_ref = ref_qpos[:T_sim, :3]
    mean_root_deviation = float(np.mean(np.linalg.norm(
        root_pos_sim - root_pos_ref, axis=1
    )))

    stats = {
        "total_frames": int(T),
        "simulated_frames": int(T_sim),
        "fall_frame": int(fall_frame) if fall_frame is not None else None,
        "completed": fall_frame is None,
        "joint_tracking_error_rad": joint_error,
        "root_position_drift_m": root_drift,
        "mean_root_deviation_m": mean_root_deviation,
        "min_root_height_m": float(min_root_h),
        "ground_offset_m": 0.0,  # will be set by caller if needed
        "fps": fps,
        "decimation": decimation,
        "root_mode": root_mode,
    }

    return sim_qpos, stats


def smooth_simulated_qpos(sim_qpos: np.ndarray, ref_qpos: np.ndarray,
                          fps: int = 30, window_ms: float = 333.0,
                          blend_alpha: float = 0.5) -> np.ndarray:
    """Post-simulation smoothing to remove PD oscillation artifacts.

    Physics sim adds ground-truth contact and prevents penetration, but PD
    tracking introduces high-frequency oscillation (jerk) from the discrete
    control loop. This filter removes the oscillation while keeping the
    physics-corrected trajectory.

    Strategy: Butterworth low-pass filter on the simulated body joint angles.
    The cutoff frequency is set to remove PD oscillation (typically 5-10 Hz)
    while preserving motion content (typically < 5 Hz for human motion).

    After filtering, blend between filtered sim and kinematic reference:
      result = blend_alpha * filtered_sim + (1 - blend_alpha) * ref

    For free-root mode: use blend_alpha=1.0 (keep full physics output).
    For kinematic-root mode: use blend_alpha=0.5 (blend with reference).

    The root (pos + quat, indices 0:7) is preserved from sim_qpos as-is.
    In free-root mode this means the root comes from physics simulation.
    In kinematic mode this means it's the reference trajectory.

    Args:
        sim_qpos:    (T, 76) simulated qpos from PD tracking
        ref_qpos:    (T, 76) reference qpos (kinematic, pre-simulation)
        fps:         control frame rate
        window_ms:   smoothing window in milliseconds (for Savitzky-Golay, default 167ms)
        blend_alpha: how much of the physics sim to keep (1.0 = all physics, 0.0 = all kinematic)
    Returns:
        smoothed_qpos: (T, 76) smoothed qpos
    """
    from scipy.signal import savgol_filter

    T = min(sim_qpos.shape[0], ref_qpos.shape[0])
    smoothed = sim_qpos[:T].copy()

    # Compute window length in frames (must be odd, >= 3)
    window_frames = max(3, int(round(window_ms / 1000.0 * fps)))
    if window_frames % 2 == 0:
        window_frames += 1
    # Savitzky-Golay polynomial order
    polyorder = min(3, window_frames - 1)

    if T < window_frames:
        # Too short to smooth — just blend raw sim with ref
        smoothed[:T, 7:] = blend_alpha * sim_qpos[:T, 7:] + (1 - blend_alpha) * ref_qpos[:T, 7:]
        return smoothed

    # Body joints only (indices 7:76), root preserved from sim_qpos as-is
    # (In free-root mode, root comes from physics; in kinematic mode, from reference)
    sim_body = sim_qpos[:T, 7:].copy()     # (T, 69)
    ref_body = ref_qpos[:T, 7:]             # (T, 69)

    # Step 1: Savitzky-Golay smooth the simulated body joints directly
    smooth_sim = savgol_filter(sim_body, window_frames, polyorder, axis=0)

    # Step 2: Blend between smoothed sim and kinematic reference
    # This bounds the output quality — even if smoothing isn't perfect,
    # the blend with reference ensures we don't add more jerk than the original.
    smoothed_body = blend_alpha * smooth_sim + (1 - blend_alpha) * ref_body

    smoothed[:T, 7:] = smoothed_body

    # Stats
    raw_jerk = np.mean(np.abs(np.diff(sim_body, n=3, axis=0))) * (fps ** 3)
    smooth_jerk = np.mean(np.abs(np.diff(smoothed_body, n=3, axis=0))) * (fps ** 3)
    ref_jerk = np.mean(np.abs(np.diff(ref_body, n=3, axis=0))) * (fps ** 3)
    if raw_jerk > 0:
        jerk_reduction = (1 - smooth_jerk / raw_jerk) * 100
        ratio = smooth_jerk / max(ref_jerk, 1e-6)
        print(f"  Post-sim smoothing: window={window_frames}f ({window_ms:.0f}ms), "
              f"blend={blend_alpha:.2f}, jerk: {raw_jerk:.0f}→{smooth_jerk:.0f} "
              f"(-{jerk_reduction:.0f}%), ratio vs kin: {ratio:.2f}x")
    else:
        print(f"  Post-sim smoothing: window={window_frames}f ({window_ms:.0f}ms)")

    return smoothed


# ===========================================================================
#  SMPL Mesh JSON Export (for website visualization)
# ===========================================================================

def smooth_smpl_poses(smpl_pose: np.ndarray, fps: int = 30,
                      window_ms: float = 333.0) -> np.ndarray:
    """Smooth SMPL axis-angle poses to remove Euler↔AA conversion jitter.

    The qpos→SMPL conversion (Euler angles → rotation matrix → axis-angle)
    can amplify small numerical differences into large axis-angle jumps near
    gimbal lock / near-zero rotation angles (where the axis is ill-defined).

    We smooth each joint's rotation in quaternion space using SLERP-based
    Savitzky-Golay filtering: convert AA → quaternion, apply SavGol on
    quaternion components (with sign flipping for continuity), convert back.

    Uses adaptive windowing: joints with large rotation ranges (hips, knees
    during crouching) get wider windows because the Euler→AA conversion
    amplifies jitter more when rotation magnitudes are large.

    Multi-pass: applies smoothing twice — first pass removes the bulk of the
    jitter, second pass catches residual oscillation from the first pass's
    boundary effects.

    Args:
        smpl_pose:   (T, 72) SMPL axis-angle, Z-up
        fps:         frame rate
        window_ms:   base smoothing window in ms (adaptive: large joints get 2x)
    Returns:
        smoothed:    (T, 72) smoothed SMPL axis-angle
    """
    from scipy.signal import savgol_filter

    T = smpl_pose.shape[0]
    if T < 5:
        return smpl_pose

    def _compute_window(wms):
        wf = max(5, int(round(wms / 1000.0 * fps)))
        if wf % 2 == 0:
            wf += 1
        wf = min(wf, T if T % 2 == 1 else T - 1)
        return wf

    base_window = _compute_window(window_ms)
    wide_window = _compute_window(window_ms * 2)  # 2x for high-flexion joints

    def _smooth_quat_seq(aa_seq, window_frames):
        """Smooth a (T, 3) axis-angle sequence in quaternion space."""
        polyorder = min(3, window_frames - 1)
        quats = sRot.from_rotvec(aa_seq).as_quat()  # (T, 4)
        # Ensure quaternion continuity
        for t in range(1, T):
            if np.dot(quats[t], quats[t - 1]) < 0:
                quats[t] = -quats[t]
        quats_smooth = savgol_filter(quats, window_frames, polyorder, axis=0)
        norms = np.linalg.norm(quats_smooth, axis=1, keepdims=True)
        quats_smooth = quats_smooth / np.maximum(norms, 1e-8)
        return sRot.from_quat(quats_smooth).as_rotvec().astype(np.float32)

    # Two-pass smoothing for better convergence
    current = smpl_pose.copy()
    for pass_idx in range(2):
        smoothed = current.copy()

        for j in range(1, 24):  # body joints
            start = j * 3
            end = start + 3
            aa_seq = current[:, start:end]

            # Adaptive window: measure rotation range for this joint
            angles = np.linalg.norm(aa_seq, axis=1)
            angle_range = float(angles.max() - angles.min())
            # Joints with > 60° range of motion get wider window
            wf = wide_window if angle_range > np.radians(60) else base_window

            smoothed[:, start:end] = _smooth_quat_seq(aa_seq, wf)

        # Root orientation: always use base window
        smoothed[:, :3] = _smooth_quat_seq(current[:, :3], base_window)

        current = smoothed

    # Stats: jerk comparison
    raw_jerk = np.mean(np.abs(np.diff(smpl_pose[:, 3:], n=3, axis=0)))
    smooth_jerk = np.mean(np.abs(np.diff(current[:, 3:], n=3, axis=0)))
    if raw_jerk > 0:
        reduction = (1 - smooth_jerk / raw_jerk) * 100
        print(f"  SMPL pose smoothing: base_window={base_window}f, "
              f"wide_window={wide_window}f, 2-pass, "
              f"AA jerk reduction={reduction:.0f}%")

    return current


def smpl_to_mesh_json(smpl_pose: np.ndarray, transl: np.ndarray,
                      fps: int, smpl_type: str = "smplh") -> dict:
    """Convert SMPL pose to mesh JSON matching the web visualizer format.

    Output format:
      {"type": "frames", "fps": N, "frames": [[{Rh, Th, poses, ...}], ...]}

    Same format as batch_npz_to_smpl_mesh_json.py produces.
    """
    T = smpl_pose.shape[0]
    root_orient = smpl_pose[:, :3]                    # (T, 3)
    body_pose = smpl_pose[:, 3:66].reshape(T, 21, 3)  # (T, 21, 3)

    # Build full poses array
    if smpl_type == "smplh":
        poses_dim = 156   # 52 joints x 3 (root + 21 body + 30 hand)
    elif smpl_type == "smplx":
        poses_dim = 165   # 55 joints x 3
    else:
        poses_dim = 72    # 24 joints x 3

    poses = np.zeros((T, poses_dim), dtype=np.float32)
    poses[:, :3] = root_orient
    poses[:, 3:66] = body_pose.reshape(T, 63)

    shapes = [[0.0] * 16]

    frames = []
    for t in range(T):
        frames.append([{
            "id": 0,
            "gender": "neutral",
            "smpl_type": smpl_type,
            "Rh": [root_orient[t].tolist()],
            "Th": [transl[t].tolist()],
            "poses": [poses[t].tolist()],
            "shapes": shapes,
            "mocap_framerate": fps,
        }])

    return {"type": "frames", "fps": fps, "frames": frames}


# ===========================================================================
#  Case Filtering
# ===========================================================================

def is_flat_ground(prompt: str) -> bool:
    """Check if a motion prompt is suitable for flat-ground physics simulation."""
    prompt_lower = prompt.lower()
    return not any(kw in prompt_lower for kw in EXCLUDE_KEYWORDS)


# ===========================================================================
#  Main Pipeline
# ===========================================================================

def process_single_motion(npz_path: str, xml_path: str, output_dir: str,
                          stats_dir: str = None, fps: int = 30,
                          root_mode: str = "free") -> dict:
    """Full pipeline: motion_135 NPZ -> physics sim -> SMPL mesh JSON.

    Returns:
        stats dict with simulation results
    """
    stem = Path(npz_path).stem
    output_json = Path(output_dir) / f"{stem}.json"

    print(f"\n{'=' * 60}")
    print(f"Processing: {stem}")
    print(f"{'=' * 60}")

    # [1] Decode motion_135
    smpl_pose, transl, motion_fps = decode_motion_135(npz_path)
    fps = motion_fps or fps
    T = smpl_pose.shape[0]
    print(f"  Decoded: {T} frames @ {fps}fps, duration={T/fps:.1f}s")

    # [2] Y-up -> Z-up
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    # [3] SMPL -> MuJoCo qpos
    model, data = load_mujoco_model(xml_path)
    body_pos_1 = model.body_pos[1].copy()
    print(f"  body_pos[1] (Pelvis offset): {body_pos_1}")
    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1, model=model)
    print(f"  ref_qpos shape: {ref_qpos.shape}, "
          f"root_h range: [{ref_qpos[:, 2].min():.3f}, {ref_qpos[:, 2].max():.3f}]")

    # [3.5] Ground offset: adjust translation so feet touch ground in frame 0.
    # Motion data may be generated for a taller body than the MuJoCo model,
    # causing feet to float above ground.
    ground_offset = compute_ground_offset(model, data, ref_qpos)
    if abs(ground_offset) > 0.001:
        print(f"  Ground offset: {ground_offset:.4f}m (subtracting from all frames)")
        ref_qpos[:, 2] -= ground_offset
        print(f"  Adjusted root_h range: [{ref_qpos[:, 2].min():.3f}, {ref_qpos[:, 2].max():.3f}]")

    # [4] Physics simulation
    sim_qpos, stats = run_physics_sim(model, data, ref_qpos, fps,
                                      root_mode=root_mode)
    stats["ground_offset_m"] = float(ground_offset)
    T_sim = stats["simulated_frames"]
    status = "COMPLETED" if stats["completed"] else f"FELL at frame {stats['fall_frame']}"
    print(f"  Simulation: {T_sim}/{T} frames — {status}")
    print(f"  Tracking error: {stats['joint_tracking_error_rad']:.4f} rad, "
          f"root drift: {stats['root_position_drift_m']:.4f} m")

    # [4.5] Post-simulation smoothing: remove PD oscillation artifacts
    # For free-root mode, use full physics output (no blending with reference)
    # because the root is physics-driven and blending would fight that.
    blend_alpha = 1.0 if root_mode == "free" else 0.5
    sim_qpos = smooth_simulated_qpos(sim_qpos, ref_qpos, fps,
                                     blend_alpha=blend_alpha)

    # [5] Export: MuJoCo qpos -> SMPL axis-angle
    smpl_pose_sim, transl_sim = qpos_to_smpl(sim_qpos, body_pos_1)

    # [5.3] Smooth SMPL axis-angle poses to remove Euler→AA conversion artifacts.
    # The qpos→SMPL conversion (Euler to axis-angle) can amplify jitter near
    # gimbal lock. We smooth in rotation-matrix space (per-joint) to avoid this.
    smpl_pose_sim = smooth_smpl_poses(smpl_pose_sim, fps)

    # [5.5] Undo ground offset so exported heights match original motion data.
    # The ground offset was applied to bring feet to ground in MuJoCo coords;
    # we reverse it to preserve the original absolute height (the physics only
    # corrected the body joint angles, not the intended global position).
    if abs(ground_offset) > 0.001:
        transl_sim[:, 2] += ground_offset

    # [6] Z-up -> Y-up
    smpl_pose_yup, transl_yup = zup_to_yup(smpl_pose_sim, transl_sim)

    # [7] Generate mesh JSON
    result = smpl_to_mesh_json(smpl_pose_yup, transl_yup, fps)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"  Saved: {output_json} ({output_json.stat().st_size / 1024:.1f} KB)")

    # Save per-motion stats
    if stats_dir:
        Path(stats_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(stats_dir) / f"{stem}.json", "w") as f:
            json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="SMPL MuJoCo physics simulation — fix foot sliding, "
                    "ground penetration, jitter"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz-dir", type=str,
                       help="Directory of motion_135 NPZ files")
    group.add_argument("--npz-file", type=str,
                       help="Single NPZ file to process")

    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for physics-simulated mesh JSONs")
    parser.add_argument("--xml-path", type=str, required=True,
                        help="Path to smpl_humanoid.xml MuJoCo model")
    parser.add_argument("--meta-dir", type=str, default=None,
                        help="Metadata JSON dir (for case filtering)")
    parser.add_argument("--stats-dir", type=str, default=None,
                        help="Directory to save per-motion sim stats")
    parser.add_argument("--filter-flat-ground", action="store_true",
                        help="Only process flat-ground suitable motions")
    parser.add_argument("--fps", type=int, default=30,
                        help="Control frame rate (default: 30)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip already processed files")
    parser.add_argument("--root-mode", type=str, default="free",
                        choices=["free", "kinematic"],
                        help="Root joint mode: 'free' (physically-grounded, "
                             "root emerges from contact forces) or 'kinematic' "
                             "(legacy: root forced to reference). Default: free")

    args = parser.parse_args()

    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    # Collect NPZ files
    if args.npz_file:
        npz_files = [Path(args.npz_file)]
    else:
        npz_dir = Path(args.npz_dir)
        npz_files = sorted(f for f in npz_dir.iterdir() if f.suffix == ".npz")

    print(f"Found {len(npz_files)} NPZ files")

    # --- Flat-ground filter ---
    if args.filter_flat_ground and args.meta_dir:
        meta_dir = Path(args.meta_dir)
        filtered = []
        excluded = 0
        for npz_path in npz_files:
            meta_path = meta_dir / f"{npz_path.stem}.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                prompt = meta.get("prompt", "") or meta.get("text", "")
                if is_flat_ground(prompt):
                    filtered.append(npz_path)
                else:
                    excluded += 1
            else:
                filtered.append(npz_path)  # no metadata => include
        npz_files = filtered
        print(f"Flat-ground filter: {len(npz_files)} kept, {excluded} excluded")

    if not npz_files:
        print("No files to process.")
        return

    # --- Process all ---
    success = 0
    failed = 0
    skipped = 0
    fell = 0
    all_stats = {}

    for npz_path in npz_files:
        if args.skip_existing:
            out_json = Path(args.output_dir) / f"{npz_path.stem}.json"
            if out_json.exists():
                skipped += 1
                continue

        try:
            stats = process_single_motion(
                str(npz_path), str(xml_path), args.output_dir,
                args.stats_dir, args.fps, args.root_mode,
            )
            all_stats[npz_path.stem] = stats
            success += 1
            if not stats["completed"]:
                fell += 1
        except Exception as e:
            import traceback
            print(f"\n  FAILED: {npz_path.stem}: {e}")
            traceback.print_exc()
            failed += 1

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total: {len(npz_files)}, Success: {success}, Failed: {failed}, "
          f"Skipped: {skipped}")
    print(f"Completed without fall: {success - fell}/{success} "
          f"({100 * (success - fell) / max(success, 1):.0f}%)")

    if all_stats:
        errors = [s["joint_tracking_error_rad"] for s in all_stats.values()]
        print(f"Mean tracking error: {np.mean(errors):.4f} rad")
        completed = [s for s in all_stats.values() if s["completed"]]
        if completed:
            print(f"Mean root drift (completed): "
                  f"{np.mean([s['root_position_drift_m'] for s in completed]):.4f} m")

    # Save aggregate summary
    if args.stats_dir:
        Path(args.stats_dir).mkdir(parents=True, exist_ok=True)
        summary = {
            "total": len(npz_files),
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "fell": fell,
            "completed_rate": (success - fell) / max(success, 1),
            "mean_tracking_error_rad": float(np.mean(errors)) if all_stats else None,
            "per_motion": all_stats,
        }
        summary_path = Path(args.stats_dir) / "_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nStats saved: {summary_path}")


if __name__ == "__main__":
    main()
