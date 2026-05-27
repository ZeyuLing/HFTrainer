#!/usr/bin/env python3
"""Run SMPL RL tracker (ONNX policy) in MuJoCo physics simulation.

Uses the trained SMPL motion tracking RL policy (exported as ONNX) to run
closed-loop physics simulation on SMPL humanoid. Unlike PD-only tracking,
the RL policy learns to balance, maintain ground contact, and produce
physically plausible motion.

Pipeline:
  motion_135 NPZ (T, 135) -- Y-up, HyMotion format
    | [1] Decode rot6d -> axis-angle
    | [2] Y-up -> Z-up coordinate transform
    | [3] SMPL axis-angle -> MuJoCo qpos
    | [4] Pre-compute reference max-coords via mj_forward (FK)
    | [5] Run RL policy in closed-loop MuJoCo simulation
    | [6] Export: qpos -> SMPL -> Z-up -> Y-up -> mesh JSON

Usage:
    # Single file
    python3 scripts/embodied/run_smpl_rl_tracker.py \\
        --npz-file output/embodied_t2m_v4/data/npz/walk_forward.npz \\
        --output-dir output/embodied_t2m_v4/data/smpl_mesh_physics

    # Batch mode
    python3 scripts/embodied/run_smpl_rl_tracker.py \\
        --npz-dir output/embodied_t2m_v4/data/npz \\
        --output-dir output/embodied_t2m_v4/data/smpl_mesh_physics \\
        --meta-dir output/embodied_t2m_v4/data/meta \\
        --filter-flat-ground
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent  # scripts/embodied/../../ = hf_trainer/

_DEFAULT_ONNX = str(
    _REPO_ROOT
    / "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker"
    / "smpl/compiled_models/unified_pipeline.onnx"
)
_DEFAULT_YAML = str(
    _REPO_ROOT
    / "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker"
    / "smpl/compiled_models/unified_pipeline.yaml"
)
_DEFAULT_MJCF = str(
    _REPO_ROOT
    / "ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml"
)


# ===========================================================================
#  Constants
# ===========================================================================

# SMPL joint names (24 joints, standard SMPL order)
SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

# MuJoCo body names in depth-first tree order (from smpl_humanoid.xml)
# NOTE: MuJoCo model has 25 bodies (index 0 = world). Body indices 1-24 = SMPL bodies.
MUJOCO_BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

# Name mapping: MuJoCo XML name -> SMPL bone name (only for mismatched names)
_MUJOCO_TO_SMPL_NAME = {
    "Torso": "Spine1", "Spine": "Spine2", "Chest": "Spine3",
    "L_Toe": "L_Foot", "R_Toe": "R_Foot",
    "L_Thorax": "L_Collar", "R_Thorax": "R_Collar",
}


def _build_reorder_indices():
    """Build smpl_2_mujoco and mujoco_2_smpl reorder arrays."""
    smpl_names = SMPL_JOINT_NAMES[1:]   # 23 non-root joints
    mj_names = MUJOCO_BODY_NAMES[1:]    # 23 non-root bodies
    s2m = []
    for mj_name in mj_names:
        smpl_name = _MUJOCO_TO_SMPL_NAME.get(mj_name, mj_name)
        s2m.append(smpl_names.index(smpl_name))
    m2s = [0] * 23
    for mj_idx, smpl_idx in enumerate(s2m):
        m2s[smpl_idx] = mj_idx
    return s2m, m2s


SMPL_2_MUJOCO, MUJOCO_2_SMPL = _build_reorder_indices()

# Fall detection
FALL_HEIGHT_THRESHOLD = 0.3

# Flat-ground case filtering
EXCLUDE_KEYWORDS = [
    "stair", "stairs", "step up", "step down", "climb", "box", "platform",
    "jump on", "jump off", "ledge", "obstacle", "hurdle", "ladder",
]

# Coordinate transforms: Y-up (SMPL) <-> Z-up (MuJoCo)
# [x,y,z]_yup -> [z,x,y]_zup
_YUP_TO_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
# [x,y,z]_zup -> [y,z,x]_yup
_ZUP_TO_YUP = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)


# ===========================================================================
#  Quaternion utilities (xyzw convention, matching ProtoMotions)
# ===========================================================================

def mujoco_wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion wxyz -> ProtoMotions xyzw."""
    return wxyz[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    """Convert ProtoMotions xyzw -> MuJoCo wxyz."""
    return xyzw[..., [3, 0, 1, 2]]


def quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1).astype(np.float32)


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    """Conjugate of xyzw quaternion."""
    result = q.copy()
    result[..., :3] *= -1.0
    return result


def extract_yaw_quat_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from orientation (xyzw)."""
    x, y, z, w = q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def compute_yaw_offset_np(robot_quat_xyzw, motion_quat_xyzw):
    """Compute yaw heading offset: offset = yaw(robot) * yaw(motion)^-1."""
    robot_yaw = extract_yaw_quat_np(robot_quat_xyzw)
    motion_yaw = extract_yaw_quat_np(motion_quat_xyzw)
    return quat_mul_np(robot_yaw, quat_conjugate_np(motion_yaw))


def apply_heading_offset_np(offset_quat_xyzw, body_rots_xyzw):
    """Apply heading offset to body rotations."""
    original_shape = body_rots_xyzw.shape
    flat = body_rots_xyzw.reshape(-1, 4)
    offset_broadcast = np.broadcast_to(offset_quat_xyzw, flat.shape)
    aligned = quat_mul_np(offset_broadcast, flat)
    return aligned.reshape(original_shape)


def quat_rotate_inverse_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by the INVERSE of quaternion q (xyzw convention).

    Equivalent to expressing vector v in the body frame defined by q.
    Used to convert world-frame angular velocity to body-local frame.

    Matches ProtoMotions `_quat_rotate_inverse_np` in deployment/state_utils.py
    and protomotions/simulator/mujoco/simulator.py.

    Args:
        q_xyzw: Unit quaternion [x, y, z, w], shape [4,].
        v: Vector in world frame, shape [3,].

    Returns:
        Vector in body frame, shape [3,].
    """
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def nlerp(q1, q2, alpha):
    """Normalized linear quaternion interpolation."""
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)
    q2_adj = np.where(dot < 0, -q2, q2)
    q = (1.0 - alpha) * q1 + alpha * q2_adj
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


# ===========================================================================
#  Motion Decoding (from motion_135 NPZ)
# ===========================================================================

def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Convert HyMotion row-major rot6d (..., 6) to rotation matrix (..., 3, 3)."""
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
    """Decode motion_135 NPZ to SMPL 72-dim axis-angle + translation (Y-up)."""
    data = np.load(npz_path, allow_pickle=True)
    motion = data["motion_135"]
    fps = int(data.get("fps", 30))
    T = motion.shape[0]

    transl = motion[:, :3]
    rot6d = motion[:, 3:].reshape(T, 22, 6)
    rotmat = rot6d_to_rotmat(rot6d)
    aa = sRot.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 22, 3)

    smpl_pose = np.zeros((T, 72), dtype=np.float32)
    smpl_pose[:, :3] = aa[:, 0, :]
    smpl_pose[:, 3:66] = aa[:, 1:22, :].reshape(T, -1)
    return smpl_pose, transl.astype(np.float32), fps


# ===========================================================================
#  Coordinate Transforms
# ===========================================================================

def yup_to_zup(smpl_pose, transl):
    """Transform SMPL pose+translation from Y-up to Z-up."""
    T = smpl_pose.shape[0]
    out_transl = (transl.astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32)
    pose_72 = smpl_pose[:, :72].astype(np.float64)
    pose_72_zup = (pose_72.reshape(T * 24, 3) @ _YUP_TO_ZUP.T).reshape(T, 72)
    out_pose = smpl_pose.copy()
    out_pose[:, :72] = pose_72_zup.astype(np.float32)
    return out_pose, out_transl


def zup_to_yup(smpl_pose, transl):
    """Transform SMPL pose+translation from Z-up to Y-up."""
    T = smpl_pose.shape[0]
    out_transl = (transl.astype(np.float64) @ _ZUP_TO_YUP.T).astype(np.float32)
    pose_72 = smpl_pose[:, :72].astype(np.float64)
    pose_72_yup = (pose_72.reshape(T * 24, 3) @ _ZUP_TO_YUP.T).reshape(T, 72)
    out_pose = smpl_pose.copy()
    out_pose[:, :72] = pose_72_yup.astype(np.float32)
    return out_pose, out_transl


# ===========================================================================
#  SMPL <-> MuJoCo qpos Conversion
# ===========================================================================

def smpl_to_qpos(smpl_pose, transl, body_pos_1):
    """Convert SMPL 72-dim axis-angle to MuJoCo 76-dim qpos.

    Euler convention: "ZYX" (intrinsic) to match smpl_mujoco.py reference code
    that the RL policy was trained with. Note: MuJoCo's actual hinge chain is
    intrinsic "XYZ" (R = Rx(a) @ Ry(b) @ Rz(c)), so "ZYX" returns [rZ, rY, rX]
    which effectively swaps axes when stored as [q_x, q_y, q_z]. This is
    mathematically incorrect but must be consistent with training convention.
    """
    T = smpl_pose.shape[0]
    qpos = np.zeros((T, 76), dtype=np.float64)
    joint_aa = smpl_pose.reshape(T, 24, 3)

    qpos[:, :3] = transl.astype(np.float64) + body_pos_1
    root_quat_xyzw = sRot.from_rotvec(joint_aa[:, 0]).as_quat()
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]  # -> wxyz

    body_aa = joint_aa[:, 1:].reshape(-1, 3)
    body_euler = sRot.from_rotvec(body_aa).as_euler("ZYX")
    body_euler = body_euler.reshape(T, 23, 3)
    body_euler_mj = body_euler[:, SMPL_2_MUJOCO]
    qpos[:, 7:] = body_euler_mj.reshape(T, 69)
    return qpos


def qpos_to_smpl(qpos, body_pos_1):
    """Convert MuJoCo 76-dim qpos to SMPL 72-dim axis-angle.

    Uses "ZYX" Euler convention to match smpl_to_qpos() and the reference
    smpl_mujoco.py code that the RL policy was trained with.
    """
    T = qpos.shape[0]
    transl = (qpos[:, :3] - body_pos_1).astype(np.float32)
    root_quat_wxyz = qpos[:, 3:7]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    root_aa = sRot.from_quat(root_quat_xyzw).as_rotvec()

    body_euler_mj = qpos[:, 7:].reshape(T, 23, 3)
    body_euler_smpl = body_euler_mj[:, MUJOCO_2_SMPL]
    body_aa = sRot.from_euler(
        "ZYX", body_euler_smpl.reshape(-1, 3)
    ).as_rotvec().reshape(T, 23, 3)

    smpl_pose = np.zeros((T, 72), dtype=np.float32)
    smpl_pose[:, :3] = root_aa.astype(np.float32)
    smpl_pose[:, 3:] = body_aa.reshape(T, 69).astype(np.float32)
    return smpl_pose, transl


# ===========================================================================
#  SMPL Mesh JSON Export
# ===========================================================================

def smpl_to_mesh_json(smpl_pose, transl, fps, smpl_type="smplh"):
    """Convert SMPL pose to mesh JSON for web visualizer."""
    T = smpl_pose.shape[0]
    root_orient = smpl_pose[:, :3]
    body_pose = smpl_pose[:, 3:66].reshape(T, 21, 3)

    poses_dim = 156 if smpl_type == "smplh" else 72
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
#  MuJoCo Model Loading (for RL tracker simulation)
# ===========================================================================

def _patch_mjcf_xml(xml_path: Path) -> str:
    """Patch MJCF for standalone MuJoCo: strip sensors, add ground/light."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    for sensor_elem in root.findall("sensor"):
        root.remove(sensor_elem)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        has_ground = any(
            g.get("type", "").lower() == "plane"
            for g in worldbody.findall("geom")
        )
        if not has_ground:
            ground = ET.SubElement(worldbody, "geom")
            ground.set("name", "floor")
            ground.set("type", "plane")
            ground.set("size", "0 0 0.05")
            ground.set("rgba", "0.7 0.7 0.7 1")
        if not worldbody.findall("light"):
            light = ET.SubElement(worldbody, "light")
            light.set("pos", "2 0 5.0")
            light.set("dir", "0 0 -1")
            light.set("diffuse", "0.4 0.4 0.4")
            light.set("specular", "0.1 0.1 0.1")
            light.set("directional", "true")

    return ET.tostring(root, encoding="unicode")


def load_mujoco_model(mjcf_path: str, stiffness: list, damping: list,
                      physics_dt: float):
    """Load MuJoCo model configured for RL tracker simulation."""
    import tempfile

    mjcf_file = Path(mjcf_path)
    if not mjcf_file.exists():
        raise FileNotFoundError(f"MJCF not found: {mjcf_file}")

    log.info(f"Loading MuJoCo model: {mjcf_file}")
    patched_xml = _patch_mjcf_xml(mjcf_file)

    asset_dir = str(mjcf_file.parent)
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

    # Override physics timestep to match training
    model.opt.timestep = physics_dt
    log.info(f"  Physics timestep: {physics_dt}s ({1.0/physics_dt:.0f}Hz)")

    # ---- Solver tuning for stability with SMPL humanoid ----
    # EMPIRICAL FINDING (test_physics_configs.py A/B test, 6 configs):
    #   - Config D (Euler + margin=0.02, no solref/solimp override) survived 164 steps
    #   - Config A (IMPLICITFAST + solref/solimp + margin) fell at step 30
    #   - Config B (Euler, all MuJoCo defaults) fell at step 26
    #   - Config E (IMPLICITFAST + margin, no solref/solimp) fell at step 30
    #
    # Key findings:
    #   1. IMPLICITFAST HURTS stability (D=164 vs E=30, only difference is integrator)
    #   2. Custom solref/solimp HURTS stability (D=164 vs A=30, only difference is contact)
    #   3. margin=0.02 HELPS stability (D=164 vs B=26, only difference is margin)
    #
    # Therefore: Keep Euler integrator (default), keep MuJoCo default solref/solimp,
    # only set margin=0.02 to simulate IsaacGym's contact_offset.

    # Integrator: Leave as Euler (MuJoCo default) — empirically best for RL tracking
    # DO NOT use IMPLICITFAST — it causes falls at step 30 vs 164 with Euler
    log.info("  Integrator: Euler (MuJoCo default, best for RL tracking per A/B test)")

    # Contact parameters: Leave as MuJoCo defaults (solref=[0.02, 1], solimp=[0.9, 0.95, 0.001, 0.5, 2])
    # DO NOT override solref/solimp — custom values cause instability
    log.info("  Contact params: MuJoCo defaults (do NOT override solref/solimp)")

    # Contact margin: DO NOT set margin=0.02!
    #
    # DIAGNOSIS (2026-05-26): margin=0.02 causes a CATAPULT EFFECT.
    # MuJoCo's `margin` is NOT equivalent to IsaacGym's `contact_offset`:
    #   - IsaacGym contact_offset=0.02: soft spring detection zone, gentle forces
    #   - MuJoCo margin=0.02: HARD constraint boundary, massive impulsive forces
    #
    # With margin=0.02 at correct root height:
    #   Initial Fz = 5166 N (20× gravity!) → robot catapulted upward 1.6cm
    # With margin=0 at same pose:
    #   Initial Fz = 256 N (gravity support) → robot STABLE
    #
    # ProtoMotions' own MuJoCo simulator (simulator/mujoco/simulator.py) does NOT
    # set any margin — it relies on MuJoCo defaults (margin=0 from XML defaults).
    # The SMPL MJCF has no geom-level margin attribute either.
    #
    # Previous A/B test showing margin=0.02 "helped" was likely an artifact of
    # wrong initial conditions (incorrect ground offset or penetrating pose).
    # With correct ground offset, margin=0 is stable and physically correct.
    margin_override = os.environ.get("PHYSFLOW_GEOM_MARGIN")
    if margin_override is not None:
        margin_value = float(margin_override)
        model.geom_margin[:] = margin_value
        log.info(f"  Contact margin override: {margin_value}")
    else:
        log.info(
            "  Contact margin: MJCF values "
            f"[{model.geom_margin.min():.4f}, {model.geom_margin.max():.4f}]"
        )

    # Zero passive forces (match training conditions)
    # ProtoMotions deployment script (test_tracker_mujoco.py) zeros all three.
    # IsaacGym/Newton don't model Coulomb joint friction, so the policy
    # wasn't trained with it — zeroing frictionloss is required.
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0
    log.info("  Zeroed passive stiffness, damping, and frictionloss")

    # Configure implicit PD actuators with training gains
    num_actuators = model.nu
    assert num_actuators == len(stiffness) == len(damping), (
        f"Actuator count mismatch: nu={num_actuators}, "
        f"stiffness={len(stiffness)}, damping={len(damping)}"
    )

    # GEAR RATIO: Override to gear=1.0 for correct PD semantics.
    #
    # The SMPL XML has gear=500, but with gear=500 the MuJoCo PD formula becomes:
    #   actuator_force = kp*(ctrl - 500*q) - kd*(500*qdot)
    # This always saturates forcerange → bang-bang control → crash at step 2.
    #
    # With gear=1, the formula is the standard PD:
    #   actuator_force = kp*(ctrl - q) - kd*qdot
    # This matches IsaacGym's PD semantics (where the policy was trained).
    #
    # ProtoMotions' MuJoCo code comment (simulator.py:540-546) also assumes gear=1:
    #   "force = kp * (ctrl - q) - kd * qd"
    # The fact that they don't override gear=500 is likely a latent bug in their
    # MuJoCo backend (which is CPU-only, num_envs=1, for lightweight testing only).
    #
    # Empirically confirmed: gear=500 crashes at step 2, gear=1 survives to step 62+.
    model.actuator_gear[:, 0] = 1.0
    log.info(f"  Gear override: 1.0 (standard PD, matches IsaacGym semantics)")

    # CRITICAL: Zero passive joint stiffness and damping.
    #
    # The SMPL MJCF has per-joint stiffness=800, damping=80 which produce passive
    # spring-damper forces: F_passive = -stiffness*(q - springref) - damping*qdot
    # These are ALWAYS applied by MuJoCo, independent of actuators.
    #
    # With actuator PD also applying F_actuator = kp*(ctrl - q) - kd*qdot (kp=800, kd=80),
    # the effective gains are DOUBLED: total_kp=1600, total_kd=160.
    #
    # The RL policy was trained in IsaacGym where ONLY the PD actuator exists (no passive
    # spring). ProtoMotions' MuJoCo simulator explicitly zeros these out:
    #   simulator.py:386-392 (_zero_passive_forces):
    #     "We manage PD control ourselves, so passive forces would double-count."
    #     self.model.jnt_stiffness[:] = 0.0
    #     self.model.dof_damping[:] = 0.0
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    log.info(f"  Zeroed passive forces: {model.njnt} joints stiffness, {model.nv} DOFs damping")

    for i in range(num_actuators):
        kp = stiffness[i]
        kd = damping[i]
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biastype[i] = 1
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kd
        model.actuator_ctrllimited[i] = 0
        # Force limiting: ENABLED to match ProtoMotions training environment.
        # ProtoMotions _configure_actuators_for_pd() (simulator/mujoco/simulator.py:549-555)
        # sets forcelimited=1 with forcerange=[-effort, effort] where effort=500 for SMPL.
        # Without force limiting, PD errors of ~1 rad produce unbounded torques
        # (kp=800 → 800 N·m), destabilizing the robot.
        model.actuator_forcerange[i, 0] = -500.0
        model.actuator_forcerange[i, 1] = 500.0
        model.actuator_forcelimited[i] = 1

    # Disable body-body self-collision to prevent instability from initial
    # interpenetrations in T2M-generated poses. Keep floor contact.
    #
    # SMPL XML has contype=7 (bits 0,1,2) and conaffinity=1 (bit 0) for body geoms.
    # Floor geom typically has contype=1, conaffinity=7 (or similar complement).
    # Contact happens when (geomA.contype & geomB.conaffinity) != 0 OR
    #                       (geomB.contype & geomA.conaffinity) != 0.
    #
    # Strategy: Set body geoms to contype=1, conaffinity=0.
    #   - Body vs Floor: floor.contype(1) & body.conaffinity(0)=0 BUT
    #     body.contype(1) & floor.conaffinity(X)=1 if floor.conaffinity has bit 0 → CONTACT ✓
    #   - Body vs Body: body.contype(1) & body.conaffinity(0)=0 → NO CONTACT ✓
    #
    # Actually simpler: just set all body geoms conaffinity=0 to disable self-collision.
    # Floor contact still works because floor's conaffinity matches body's contype.
    self_collision_mode = os.environ.get(
        "PHYSFLOW_SELF_COLLISIONS", "false"
    ).strip().lower()
    if self_collision_mode not in {"true", "false"}:
        raise ValueError(
            "PHYSFLOW_SELF_COLLISIONS must be 'true' or 'false', "
            f"got {self_collision_mode!r}"
        )
    if self_collision_mode == "false":
        floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for geom_id in range(model.ngeom):
            if geom_id == floor_geom_id:
                continue  # Don't modify floor
            # Keep contype (body can still contact floor via floor's conaffinity)
            # Zero conaffinity so other bodies' contype doesn't match -> no self-collision
            model.geom_conaffinity[geom_id] = 0
        log.info("  Disabled body-body self-collision (set body geom conaffinity=0)")
    else:
        log.info("  Body-body self-collision: enabled from MJCF")

    log.info(f"  {num_actuators} actuators, {model.nbody} bodies, "
             f"nq={model.nq}, nv={model.nv}")
    return model, data


# ===========================================================================
#  Virtual Floor Spring (emulate IsaacGym contact_offset=0.02)
# ===========================================================================

# IsaacGym contact_offset=0.02m: contacts are detected and the solver provides
# support when geom surfaces are within 20mm of the floor, EVEN if not
# geometrically penetrating. The RL policy was TRAINED with this behavior.
#
# MuJoCo only generates contact forces upon geometric penetration (dist < 0).
# This means feet hovering 1-2mm above floor get NO support → the policy's
# learned ankle torques (which assume floor pushback) lift feet further → fall.
#
# Solution: Apply a virtual spring force via xfrc_applied to each foot body
# when its lowest geom surface is within contact_offset of the floor. This
# provides the same continuous support zone the policy expects.
#
# Key design choices:
#   - Only upward (Fz > 0) forces — floor can't pull
#   - Stiffness calibrated so total force ≈ body_weight at equilibrium
#   - Damping prevents oscillation
#   - Also apply friction (lateral force opposing velocity) for stability
#   - Forces applied to BODY COM via xfrc_applied (wrench in world frame)

VIRTUAL_FLOOR_CONTACT_OFFSET = 0.02  # meters — match IsaacGym training
VIRTUAL_FLOOR_STIFFNESS = 9000.0      # N/m — calibrated: 4 bodies × 0.02m × 9000 = 720N ≈ gravity
VIRTUAL_FLOOR_DAMPING = 150.0         # Ns/m — underdamped to allow natural settling
VIRTUAL_FLOOR_FRICTION_COEFF = 1.0    # Coulomb friction coefficient


def _identify_foot_bodies(model):
    """Identify foot body IDs and their associated geom info for virtual spring.

    Returns list of dicts: [{body_id, body_name, geom_ids, ...}]
    """
    foot_body_names = ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]
    foot_bodies = []

    for bname in foot_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            log.warning(f"  Virtual floor spring: body '{bname}' not found, skipping")
            continue

        # Find all geoms belonging to this body
        geom_ids = []
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid:
                geom_ids.append(gid)

        foot_bodies.append({
            "body_id": bid,
            "body_name": bname,
            "geom_ids": geom_ids,
        })

    log.info(f"  Virtual floor spring: {len(foot_bodies)} foot bodies identified: "
             f"{[fb['body_name'] for fb in foot_bodies]}")
    return foot_bodies


def _compute_foot_bottom_z(model, data, foot_body):
    """Compute the lowest Z coordinate across all geoms of a foot body.

    Uses exact geom geometry (capsule/sphere/box) + current orientation.
    """
    min_z = float("inf")
    for gid in foot_body["geom_ids"]:
        gtype = int(model.geom_type[gid])
        gsize = model.geom_size[gid]
        gxpos = data.geom_xpos[gid]
        gxmat = data.geom_xmat[gid].reshape(3, 3)

        if gtype == 5:  # capsule: center ± half_len along local Z + radius
            radius = gsize[0]
            half_len = gsize[1]
            # Z-extent = |projection of capsule axis onto world Z| * half_len + radius
            z_extent = abs(gxmat[2, 2]) * half_len + radius
            bottom_z = gxpos[2] - z_extent
        elif gtype == 3:  # sphere
            bottom_z = gxpos[2] - gsize[0]
        elif gtype == 6:  # box
            half_extents = gsize[:3]
            z_extent = (abs(gxmat[2, 0]) * half_extents[0] +
                        abs(gxmat[2, 1]) * half_extents[1] +
                        abs(gxmat[2, 2]) * half_extents[2])
            bottom_z = gxpos[2] - z_extent
        else:
            bottom_z = gxpos[2]

        min_z = min(min_z, bottom_z)

    return min_z


def _apply_virtual_floor_spring(model, data, foot_bodies):
    """Apply virtual floor spring forces to foot bodies within contact_offset.

    Emulates IsaacGym contact_offset=0.02m by providing upward support force
    when foot geom surfaces are within 20mm of the floor (Z=0 plane).

    Forces are applied via data.xfrc_applied (6D wrench at body COM in world frame).
    Must be called BEFORE each mj_step (MuJoCo resets xfrc_applied between steps
    only if you're using mj_step1/mj_step2 separately, but with mj_step it
    persists — we clear and reapply each substep to be safe).

    Args:
        model: MjModel
        data: MjData (modified in-place: xfrc_applied updated)
        foot_bodies: list from _identify_foot_bodies()
    """
    for fb in foot_bodies:
        bid = fb["body_id"]
        bottom_z = _compute_foot_bottom_z(model, data, fb)

        # Distance from floor (floor at Z=0)
        dist = bottom_z  # positive = above floor, negative = penetrating

        if 0 < dist < VIRTUAL_FLOOR_CONTACT_OFFSET:
            # Only in the GAP zone: foot is above floor but within contact_offset.
            # When dist <= 0, MuJoCo's hard contacts already provide full support.
            # When dist >= contact_offset, foot is too high for virtual support.
            depth = VIRTUAL_FLOOR_CONTACT_OFFSET - dist

            # Spring force (upward, proportional to penetration into zone)
            f_spring = VIRTUAL_FLOOR_STIFFNESS * depth

            # Damping force (opposes vertical velocity of body)
            # body velocity in world frame: data.cvel[bid] is (6,) = [ang(3), lin(3)]
            # But cvel is in body-attached frame. Use subtree_linvel instead.
            # Actually, for simple Z-velocity, use the body's COM velocity.
            # data.cvel stores spatial velocity; linear part is [3:6].
            # However, cvel is in the frame at body COM with world orientation
            # for MuJoCo 3.x. Safest: use finite difference or qvel projection.
            # Simple approach: use the body's vertical velocity from subtree_linvel
            body_vz = data.cvel[bid, 5]  # linear Z velocity (world frame)

            f_damping = -VIRTUAL_FLOOR_DAMPING * body_vz

            # Total normal force (only upward — floor can't pull)
            f_normal = max(0.0, f_spring + f_damping)

            # Apply normal force (Z-axis, world frame)
            data.xfrc_applied[bid, 2] += f_normal

            # Virtual friction: oppose lateral velocity proportional to normal force
            if f_normal > 0 and VIRTUAL_FLOOR_FRICTION_COEFF > 0:
                body_vx = data.cvel[bid, 3]  # linear X velocity
                body_vy = data.cvel[bid, 4]  # linear Y velocity
                v_lateral = np.sqrt(body_vx**2 + body_vy**2)
                if v_lateral > 1e-6:
                    # Coulomb friction: F_friction = mu * F_normal, opposing velocity
                    f_friction = VIRTUAL_FLOOR_FRICTION_COEFF * f_normal
                    # Cap friction at what would stop the body (prevent oscillation)
                    # Simple velocity-dependent scaling for smoothness
                    scale = min(1.0, v_lateral / 0.01)  # ramp up over 1cm/s
                    data.xfrc_applied[bid, 0] += -body_vx / v_lateral * f_friction * scale
                    data.xfrc_applied[bid, 1] += -body_vy / v_lateral * f_friction * scale


# ===========================================================================
#  Quaternion utilities for reference precomputation
# ===========================================================================

def _quat_mul_wxyz(q1, q2):
    """Multiply two quaternions in wxyz format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ===========================================================================
#  Reference Max-Coords Precomputation
# ===========================================================================

def precompute_reference_maxcoords(model, data, ref_qpos, dt_ref):
    """Run MuJoCo FK on each reference frame to get max-coords.

    Returns dict with body_pos, body_rot (xyzw), body_vel, body_ang_vel,
    dof_pos — all at the reference motion's native frame rate.
    """
    T = ref_qpos.shape[0]
    num_bodies = 24  # SMPL bodies (MuJoCo indices 1-24)

    # CRITICAL: Use float64 for intermediate computation to avoid precision loss
    # in finite-difference velocity/angular velocity calculations.
    # Float32 accumulates ~6e-6 error per frame which compounds over the sim,
    # causing the policy to receive corrupted observations and fall at step ~67.
    # The arrays are cast to float32 only at ONNX inference time (see extract_sim_state).
    body_pos = np.zeros((T, num_bodies, 3))  # float64
    body_rot = np.zeros((T, num_bodies, 4))  # float64, xyzw
    dof_pos = np.zeros((T, model.nu), dtype=np.float32)

    log.info(f"  Pre-computing reference FK for {T} frames...")
    for t in range(T):
        data.qpos[:] = ref_qpos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        body_pos[t] = data.xpos[1:num_bodies + 1].copy()  # float64
        body_rot_wxyz = data.xquat[1:num_bodies + 1].copy()
        body_rot[t] = mujoco_wxyz_to_xyzw(body_rot_wxyz)  # float64
        dof_pos[t] = data.qpos[7:].copy().astype(np.float32)

    # Compute velocities via BACKWARD finite differences (matching test_physics_configs.py)
    # CRITICAL: test_physics_configs uses body_vel[f] = (pos[f] - pos[f-1]) / dt
    # which means velocity at frame f represents "how we arrived at frame f".
    # This is what the RL policy expects (trained with this convention in ProtoMotions).
    # Previous code used FORWARD difference body_vel[t] = (pos[t+1] - pos[t]) / dt
    # which gives velocity at frame t representing "where we're going from t" — WRONG.
    body_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)

    for f in range(1, T):
        body_vel[f] = (body_pos[f] - body_pos[f - 1]) / dt_ref

        # Angular velocity: simplified quaternion-based (matching test_physics_configs.py)
        # Uses 2 * vec(dq) / dt approximation, NOT scipy as_rotvec()
        # The RL policy was trained with this convention.
        for j in range(num_bodies):
            q0 = body_rot[f - 1, j]  # xyzw
            q1 = body_rot[f, j]      # xyzw
            # Convert to wxyz for quaternion multiplication
            q0_w = np.array([q0[3], q0[0], q0[1], q0[2]])
            q1_w = np.array([q1[3], q1[0], q1[1], q1[2]])
            # dq = q1 * q0_inv (in wxyz)
            q0_inv = np.array([q0_w[0], -q0_w[1], -q0_w[2], -q0_w[3]])
            dq = _quat_mul_wxyz(q1_w, q0_inv)
            # Ensure shortest path
            if dq[0] < 0:
                dq = -dq
            # ang_vel ≈ 2 * vec(dq) / dt
            body_ang_vel[f, j] = 2.0 * dq[1:4] / dt_ref  # float64

    # Frame 0: copy from frame 1 (matching test_physics_configs.py)
    if T > 1:
        body_vel[0] = body_vel[1]
        body_ang_vel[0] = body_ang_vel[1]

    log.info(f"  Reference FK complete. body_pos range: "
             f"z=[{body_pos[:, 0, 2].min():.3f}, {body_pos[:, 0, 2].max():.3f}]")

    return {
        "body_pos": body_pos,
        "body_rot": body_rot,
        "body_vel": body_vel,
        "body_ang_vel": body_ang_vel,
        "dof_pos": dof_pos,
    }


def get_reference_at_time(ref_data, time_sec, dt_ref, total_frames):
    """Get interpolated reference state at a given simulation time.

    Interpolates between reference frames (which are at dt_ref intervals)
    for the simulation control loop (which runs at a different frequency).
    """
    frame_f = time_sec / dt_ref
    frame_lo = int(np.floor(frame_f))
    frame_lo = min(frame_lo, total_frames - 1)
    frame_hi = min(frame_lo + 1, total_frames - 1)
    alpha = frame_f - int(np.floor(frame_f))

    if frame_lo == frame_hi or alpha < 1e-6:
        return {k: v[frame_lo].copy().astype(np.float32) for k, v in ref_data.items()}

    result = {}
    for key in ["body_pos", "body_vel", "body_ang_vel", "dof_pos"]:
        result[key] = ((1.0 - alpha) * ref_data[key][frame_lo] +
                       alpha * ref_data[key][frame_hi]).astype(np.float32)

    # Quaternion interpolation via nlerp
    result["body_rot"] = nlerp(
        ref_data["body_rot"][frame_lo],
        ref_data["body_rot"][frame_hi],
        alpha,
    ).astype(np.float32)

    return result


# ===========================================================================
#  COM velocity correction (MuJoCo frame-origin → IsaacGym COM semantics)
# ===========================================================================
# MuJoCo's data.cvel reports velocity at the body FRAME ORIGIN, but
# IsaacGym (where the RL policy was trained) reports velocity at the
# center-of-mass (COM). We correct using:
#   v_COM = v_frame + ω × r_COM_world
# where r_COM_world = data.xmat[body] @ model.body_ipos[body]
#
# IMPORTANT: We use model.body_ipos (MuJoCo's built-in inertial frame offset,
# i.e. the TRUE COM in local body coordinates) — NOT averaged geom positions.
# Using averaged geom positions is an approximation that introduces error and
# causes the RL policy to output incorrect torques, leading to falls.
# ===========================================================================


# ===========================================================================
#  Extract current simulation state (max-coords)
# ===========================================================================

def extract_sim_state(model, data, num_bodies=24, body_com_offsets=None):
    """Extract current rigid body state from MuJoCo simulation.

    Returns dict with body_pos, body_rot (xyzw), body_vel, body_ang_vel
    in world frame.

    Uses model.body_ipos (true COM offset in local frame) and data.xmat
    (body rotation matrix) to compute COM velocity correction. This matches
    the ProtoMotions/IsaacGym convention where body velocities are reported
    at the center of mass.

    The body_com_offsets parameter is IGNORED (kept for API compatibility).
    We always use model.body_ipos which is the correct COM offset.
    """
    # Use float64 for intermediate computation to avoid precision loss that
    # compounds over simulation steps. Cast to float32 only at ONNX input time.
    # (Matching test_physics_configs.py behavior — float32 here caused falls at step 62-67)
    body_pos = np.zeros((num_bodies, 3))
    body_rot = np.zeros((num_bodies, 4))  # xyzw
    body_vel = np.zeros((num_bodies, 3))
    body_ang_vel = np.zeros((num_bodies, 3))

    for j in range(num_bodies):
        bid = j + 1  # Skip world body at index 0

        # Position and rotation
        body_pos[j] = data.xpos[bid].copy()
        quat_wxyz = data.xquat[bid].copy()
        # Convert wxyz → xyzw
        body_rot[j] = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

        # COM velocity correction: v_COM = v_frame + ω × r_COM_world
        cvel = data.cvel[bid]  # (6,) = [ang_vel(3), lin_vel(3)]
        ang_vel = cvel[:3].copy()
        lin_vel = cvel[3:].copy()

        # COM offset in world frame using body_ipos (true COM) and xmat
        com_local = model.body_ipos[bid]
        xmat = data.xmat[bid].reshape(3, 3)
        com_world = xmat @ com_local

        # v_COM = v_frame + ω × r_COM_world
        lin_vel_com = lin_vel + np.cross(ang_vel, com_world)

        body_vel[j] = lin_vel_com
        body_ang_vel[j] = ang_vel

    return {
        "body_pos": body_pos,
        "body_rot": body_rot,
        "body_vel": body_vel,
        "body_ang_vel": body_ang_vel,
    }


# ===========================================================================
#  Main RL Tracker Simulation
# ===========================================================================

def run_rl_tracker(
    ref_qpos: np.ndarray,
    motion_fps: int,
    onnx_path: str,
    mjcf_path: str,
    yaml_meta: dict,
) -> tuple[np.ndarray, dict]:
    """Run RL policy in closed-loop MuJoCo simulation.

    Args:
        ref_qpos:    (T_ref, 76) reference MuJoCo qpos trajectory
        motion_fps:  FPS of the reference motion
        onnx_path:   Path to SMPL ONNX tracker model
        mjcf_path:   Path to SMPL humanoid MJCF XML
        yaml_meta:   Parsed YAML metadata for the ONNX model

    Returns:
        sim_qpos:  (T_sim, 76) simulated qpos trajectory
        stats:     dict with simulation statistics
    """
    import onnxruntime as ort

    # ------------------------------------------------------------------
    # Parse YAML metadata
    # ------------------------------------------------------------------
    robot_meta = yaml_meta["robot"]
    timing = yaml_meta["timing"]
    motion_meta = yaml_meta["motion"]
    control = yaml_meta["control"]
    runtime = yaml_meta["_runtime"]
    history_action_mode = os.environ.get("PHYSFLOW_HISTORY_ACTION_MODE", "raw").strip().lower()
    if history_action_mode not in {"raw", "processed"}:
        raise ValueError(
            "PHYSFLOW_HISTORY_ACTION_MODE must be 'raw' or 'processed', "
            f"got {history_action_mode!r}"
        )
    virtual_floor_enabled = os.environ.get(
        "PHYSFLOW_VIRTUAL_FLOOR", "true"
    ).strip().lower()
    if virtual_floor_enabled not in {"true", "false"}:
        raise ValueError(
            "PHYSFLOW_VIRTUAL_FLOOR must be 'true' or 'false', "
            f"got {virtual_floor_enabled!r}"
        )

    anchor_body_index = robot_meta["anchor_body_index"]  # 0 for SMPL
    num_bodies = robot_meta["num_bodies"]                # 24
    num_dofs = robot_meta["num_dofs"]                    # 69
    control_dt = timing["control_dt"]                    # 0.02
    decimation = timing["decimation"]                    # 20
    physics_dt = timing["physics_dt"]                    # 0.001
    future_step_indices = motion_meta["future_step_indices"]  # [1]
    future_dt_seconds = motion_meta["future_dt_seconds"]      # [0.02]
    stiffness = control["stiffness"]                     # 69 values
    damping_ctrl = control["damping"]                    # 69 values
    onnx_name_to_key = runtime["onnx_name_to_in_key"]
    log.info(f"  Historical action feedback mode: {history_action_mode}")
    log.info(f"  Virtual floor spring: {virtual_floor_enabled}")

    T_ref = ref_qpos.shape[0]
    dt_ref = 1.0 / motion_fps
    motion_duration = T_ref * dt_ref
    T_sim = int(motion_duration / control_dt)

    log.info(f"  Reference: {T_ref} frames @ {motion_fps}fps ({motion_duration:.1f}s)")
    log.info(f"  Simulation: {T_sim} steps @ {1.0/control_dt:.0f}Hz control, "
             f"{1.0/physics_dt:.0f}Hz physics, decimation={decimation}")

    # ------------------------------------------------------------------
    # Load ONNX session
    # ------------------------------------------------------------------
    log.info(f"  Loading ONNX: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual_in_names = [inp.name for inp in session.get_inputs()]
    actual_out_names = [out.name for out in session.get_outputs()]
    log.info(f"  ONNX inputs: {actual_in_names}")
    log.info(f"  ONNX outputs: {actual_out_names}")

    # ------------------------------------------------------------------
    # Load MuJoCo model
    # ------------------------------------------------------------------
    model, data = load_mujoco_model(mjcf_path, stiffness, damping_ctrl, physics_dt)
    body_pos_1 = model.body_pos[1].copy()

    # COM velocity correction now uses model.body_ipos directly inside
    # extract_sim_state() — no precomputed offsets needed.
    # (model.body_ipos is the TRUE COM offset in local body frame)
    log.info(f"  Using model.body_ipos for COM velocity correction")

    # ------------------------------------------------------------------
    # Frame-0 initial height: place feet on ground for MuJoCo hard contacts
    # ------------------------------------------------------------------
    # In ProtoMotions training (IsaacGym), ref_respawn_offset=0.05m lifts the
    # robot above ground, but IsaacGym's contact_offset=0.02m still engages
    # floor contacts within 1-2 control steps. The policy expects ground
    # support from nearly the start.
    #
    # In MuJoCo with hard contacts, geoms must physically overlap for contact.
    # With ref_respawn_offset=0.05m, foot geoms are ~4.5cm above floor →
    # no contact for ~5 control steps → robot freefalls → too much downward
    # velocity by the time contact engages → collapses.
    #
    # Solution: use ref_respawn_offset=0.0 for MuJoCo. This places the lowest
    # body origin at 0.015m above ground (from fix_height), which means foot
    # geom surfaces (~2cm below body origin) are at or slightly below ground
    # → immediate contact → ground support from step 0.
    #
    # The policy handles this fine because it was trained with contact_offset
    # making contacts "sticky" — having contact immediately is better than
    # having a 4.5cm gap with no support.
    REF_RESPAWN_OFFSET = 0.0  # No offset for MuJoCo (hard contacts need geom overlap)
    ref_qpos[:, 2] += REF_RESPAWN_OFFSET
    log.info(f"  Applied ref_respawn_offset = +{REF_RESPAWN_OFFSET}m (MuJoCo: no offset for hard contacts)")
    log.info(f"  After offset: root_h range = [{ref_qpos[:, 2].min():.3f}, {ref_qpos[:, 2].max():.3f}]")

    # ------------------------------------------------------------------
    # Pre-compute reference max-coords via FK (with corrected heights)
    # ------------------------------------------------------------------
    ref_data = precompute_reference_maxcoords(model, data, ref_qpos, dt_ref)

    # ------------------------------------------------------------------
    # Set initial pose from reference (fix_height applied, no respawn offset)
    # ------------------------------------------------------------------
    # Robot starts with feet on/slightly-in the floor (MuJoCo hard contacts):
    # - fix_height set min body origin Z = 0.015m
    # - Foot geom surfaces ~2cm below body origin → slight floor penetration
    # - MuJoCo resolves penetration via constraint force → immediate support
    # - ZERO initial velocity (matching ProtoMotions test_tracker_mujoco.py set_initial_pose)
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0  # ProtoMotions deployment zeros all velocities at init
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------
    # Initial velocity: ZERO (matching ProtoMotions deployment)
    # ------------------------------------------------------------------
    # ProtoMotions test_tracker_mujoco.py `set_initial_pose()` (lines 373-392)
    # explicitly sets `data.qvel[:] = 0.0` — it does NOT initialize from reference.
    # This is critical because:
    #   1. Non-zero initial velocity creates momentum that can break contact equilibrium
    #   2. The RL policy handles velocity ramp-up through its learned balance behavior
    #   3. Initial mismatch between qvel and actual motion state is small (1 frame)
    #      and the policy naturally compensates within a few control steps
    # Previous approach set velocities from reference finite differences, which
    # caused large initial PD errors → excessive torques → contact instability.
    log.info(f"  Initial velocity: ZERO (matching ProtoMotions deployment)")

    # NOTE: Do NOT set data.ctrl before the simulation loop.
    # test_physics_configs (which survives 148 steps) does NOT pre-set ctrl.
    # Setting ctrl = ref_qpos[0, 7:] + calling mj_forward() before the loop
    # causes an extra actuator force computation that shifts the initial state,
    # reducing survival by ~31 steps (from 148 to 117).
    # The RL policy will set ctrl on its first inference step.

    # Check initial contact state for diagnostics
    log.info(f"  Initial root height: {data.qpos[2]:.4f}m")
    log.info(f"  Initial contacts: {data.ncon}")
    for ci in range(min(data.ncon, 5)):
        c = data.contact[ci]
        g1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
        g2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
        log.info(f"    contact[{ci}]: {g1_name}<->{g2_name}, dist={c.dist:.4f}")

    # DEBUG: Check actual foot geom positions in simulation model
    log.info("  DEBUG foot geom Z positions after mj_forward:")
    for gid in range(model.ngeom):
        body_id = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"b{body_id}"
        if bname in ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe"):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
            gtype = int(model.geom_type[gid])
            gsize = model.geom_size[gid]
            gxpos = data.geom_xpos[gid]
            gxmat = data.geom_xmat[gid].reshape(3, 3)
            # Compute lowest point
            if gtype == 5:  # capsule
                radius = gsize[0]
                half_len = gsize[1]
                z_extent = abs(gxmat[2, 2]) * half_len + radius
                bottom_z = gxpos[2] - z_extent
            elif gtype == 3:  # sphere
                bottom_z = gxpos[2] - gsize[0]
            elif gtype == 6:  # box
                half_extents = gsize[:3]
                z_extent = (abs(gxmat[2, 0]) * half_extents[0] +
                            abs(gxmat[2, 1]) * half_extents[1] +
                            abs(gxmat[2, 2]) * half_extents[2])
                bottom_z = gxpos[2] - z_extent
            else:
                bottom_z = gxpos[2]
            log.info(f"    {gname} (body={bname}): type={gtype}, pos_z={gxpos[2]:.5f}, "
                     f"size={gsize[:3]}, bottom_z={bottom_z:.5f}")

    # ------------------------------------------------------------------
    # Compute heading offset (identity in simulation, but keep for correctness)
    # ------------------------------------------------------------------
    sim_state_0 = extract_sim_state(model, data, num_bodies)
    robot_anchor_rot = sim_state_0["body_rot"][anchor_body_index]
    motion_anchor_rot = ref_data["body_rot"][0, anchor_body_index]
    heading_offset = compute_yaw_offset_np(robot_anchor_rot, motion_anchor_rot)
    log.info(f"  Heading offset: {heading_offset} (should be ~[0,0,0,1])")

    # ------------------------------------------------------------------
    # Identify foot bodies for virtual floor spring
    # ------------------------------------------------------------------
    foot_bodies = _identify_foot_bodies(model)

    # ------------------------------------------------------------------
    # Simulation state variables
    # ------------------------------------------------------------------
    prev_actions = np.zeros(num_dofs, dtype=np.float32)
    fall_frame = None
    root_height_min = float("inf")

    sim_qpos_list = []
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    for step_idx in range(T_sim):
        sim_time = step_idx * control_dt

        # ---- Record current qpos ----
        sim_qpos_list.append(data.qpos.copy())

        # ---- Fall detection ----
        root_h = float(data.qpos[2])
        root_height_min = min(root_height_min, root_h)
        if root_h < FALL_HEIGHT_THRESHOLD or np.any(np.isnan(data.qpos)):
            fall_frame = step_idx
            reason = "NaN" if np.any(np.isnan(data.qpos)) else f"root_h={root_h:.3f}m"
            log.warning(f"  FALL at step {step_idx}/{T_sim}: {reason}")
            break

        # ---- Extract current simulation state (max-coords) ----
        cur_state = extract_sim_state(model, data, num_bodies)

        # ---- Get reference state at current time (INTERPOLATED) ----
        # ProtoMotions' MotionLib ALWAYS uses SLERP/LERP interpolation for reference
        # motion lookup (confirmed from protomotions/components/motion_lib/motion_lib.py).
        # The policy was trained with smooth interpolated references, NOT nearest-frame.
        # Using nearest-frame creates discontinuous jumps in the reference signal that
        # the policy never saw during training, causing incorrect action outputs.
        ref_now = get_reference_at_time(ref_data, sim_time, dt_ref, T_ref)

        # ---- Get future reference states (INTERPOLATED) ----
        # NO heading offset — initial pose already matches reference, so
        # heading offset is identity and applying it risks numerical noise.
        future_states = []
        for fi, fdt in zip(future_step_indices, future_dt_seconds):
            future_time = sim_time + fi * fdt
            future_ref = get_reference_at_time(ref_data, future_time, dt_ref, T_ref)
            future_states.append(future_ref)

        # Stack future states: (num_future_steps, num_bodies, dim)
        future_body_pos = np.stack(
            [fs["body_pos"] for fs in future_states], axis=0)       # (1, 24, 3)
        future_body_rot = np.stack(
            [fs["body_rot"] for fs in future_states], axis=0)       # (1, 24, 4)
        future_body_vel = np.stack(
            [fs["body_vel"] for fs in future_states], axis=0)       # (1, 24, 3)
        future_body_ang_vel = np.stack(
            [fs["body_ang_vel"] for fs in future_states], axis=0)   # (1, 24, 3)

        # ---- Build ONNX inputs ----
        key_to_array = {
            "current.rigid_body_pos": cur_state["body_pos"][None],         # (1, 24, 3)
            "current.rigid_body_rot": cur_state["body_rot"][None],         # (1, 24, 4)
            "current.rigid_body_vel": cur_state["body_vel"][None],         # (1, 24, 3)
            "current.rigid_body_ang_vel": cur_state["body_ang_vel"][None], # (1, 24, 3)
            "ground_heights": np.zeros(1, dtype=np.float32),               # (1,)
            "historical.actions": prev_actions[None, None],                # (1, 1, 69)
            "mimic.future_pos": future_body_pos[None],                     # (1, 1, 24, 3)
            "mimic.future_rot": future_body_rot[None],                     # (1, 1, 24, 4)
            "mimic.future_vel": future_body_vel[None],                     # (1, 1, 24, 3)
            "mimic.future_ang_vel": future_body_ang_vel[None],             # (1, 1, 24, 3)
        }

        onnx_inputs = {}
        for onnx_name, sem_key in onnx_name_to_key.items():
            if sem_key in key_to_array:
                onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)

        # Verify all inputs are present
        missing = [n for n in actual_in_names if n not in onnx_inputs]
        if missing:
            log.error(f"  Missing ONNX inputs: {missing}")
            log.error(f"  Available keys: {list(key_to_array.keys())}")
            log.error(f"  ONNX name->key map: {onnx_name_to_key}")
            raise RuntimeError(f"Missing ONNX inputs: {missing}")

        # ---- ONNX inference ----
        ort_out = session.run(actual_out_names, onnx_inputs)

        # Parse outputs
        out_dict = {name: val for name, val in zip(actual_out_names, ort_out)}
        joint_pos_targets = out_dict["joint_pos_targets"].squeeze().copy()

        # ---- Step-0 diagnostics ----
        if step_idx == 0:
            log.info("  === STEP-0 DIAGNOSTICS ===")
            for oname in actual_out_names:
                arr = out_dict[oname].squeeze()
                log.info(f"  ONNX out '{oname}': shape={arr.shape}, "
                         f"min={arr.min():.4f}, max={arr.max():.4f}, "
                         f"mean={arr.mean():.4f}, std={arr.std():.4f}")
            log.info(f"  joint_pos_targets[:5] = {joint_pos_targets[:5]}")
            log.info(f"  raw actions[:5] = {out_dict['actions'].squeeze()[:5]}")
            if "stiffness_targets" in out_dict:
                st = out_dict["stiffness_targets"].squeeze()
                log.info(f"  stiffness_targets[:5] = {st[:5]}")
                log.info(f"  stiffness range: [{st.min():.1f}, {st.max():.1f}]")
            if "damping_targets" in out_dict:
                dt_ = out_dict["damping_targets"].squeeze()
                log.info(f"  damping_targets[:5] = {dt_[:5]}")
                log.info(f"  damping range: [{dt_.min():.1f}, {dt_.max():.1f}]")
            # Current state info
            log.info(f"  root_pos = {data.qpos[:3]}")
            log.info(f"  root_quat(wxyz) = {data.qpos[3:7]}")
            log.info(f"  body_pos[0](Pelvis) = {cur_state['body_pos'][0]}")
            log.info(f"  body_pos[1](L_Hip) = {cur_state['body_pos'][1]}")
            # Reference state info
            log.info(f"  ref_qpos[0][:7] = {ref_qpos[0][:7]}")
            log.info(f"  ref body_pos[0](Pelvis) = {ref_data['body_pos'][0, 0]}")

            # ---- OBSERVATION DIAGNOSTICS: current vs future max-coords ----
            log.info(f"  === CURRENT vs FUTURE MAX-COORD COMPARISON (step 0) ===")
            cur_pos = cur_state["body_pos"]
            fut_pos = future_body_pos[0]  # (24, 3)
            pos_diff = np.sqrt(((fut_pos - cur_pos)**2).sum(-1))
            log.info(f"  |future_pos - cur_pos| per body: "
                     f"mean={pos_diff.mean():.4f}m, max={pos_diff.max():.4f}m")
            cur_rot = cur_state["body_rot"]
            fut_rot = future_body_rot[0]  # (24, 4)
            # Angular difference via quaternion
            from scipy.spatial.transform import Rotation as R_diag
            R_c = R_diag.from_quat(cur_rot)  # xyzw
            R_f = R_diag.from_quat(fut_rot)  # xyzw
            rot_diff_rv = (R_f * R_c.inv()).as_rotvec()
            rot_diff_mag = np.sqrt((rot_diff_rv**2).sum(-1))
            log.info(f"  |future_rot - cur_rot| per body (rad): "
                     f"mean={rot_diff_mag.mean():.4f}, max={rot_diff_mag.max():.4f}")
            log.info(f"  dt_ref={dt_ref:.4f}s, future_time={0 + future_step_indices[0]*future_dt_seconds[0]:.4f}s")
            log.info(f"  future frame interpolation: frame_f={0.02/dt_ref:.2f}")
            log.info(f"  === END CURRENT vs FUTURE COMPARISON ===")

            # ---- CRITICAL DIAGNOSTIC: Compare jpt vs actual reference DOFs ----
            ref_dofs = ref_qpos[0, 7:]  # What the joints ARE at step 0
            jpt_error = joint_pos_targets - ref_dofs
            log.info(f"  === JPT vs REF DOF COMPARISON (step 0) ===")
            log.info(f"  ref_dofs[:10] = {ref_dofs[:10]}")
            log.info(f"  jpt[:10]      = {joint_pos_targets[:10]}")
            log.info(f"  error[:10]    = {jpt_error[:10]}")
            log.info(f"  |error| stats: mean={np.abs(jpt_error).mean():.4f}, "
                     f"max={np.abs(jpt_error).max():.4f}, "
                     f"std={np.abs(jpt_error).std():.4f}")
            log.info(f"  PD force estimate (kp=800): mean={800*np.abs(jpt_error).mean():.1f} Nm, "
                     f"max={800*np.abs(jpt_error).max():.1f} Nm")
            # Show which joints have largest errors
            worst_joints = np.argsort(np.abs(jpt_error))[::-1][:10]
            log.info(f"  Worst 10 joints (by |jpt-ref_dof| error):")
            for ji in worst_joints:
                log.info(f"    joint[{ji:2d}]: ref_dof={ref_dofs[ji]:.4f}, "
                         f"jpt={joint_pos_targets[ji]:.4f}, "
                         f"error={jpt_error[ji]:.4f}")
            log.info(f"  === END JPT vs REF DOF COMPARISON ===")
            log.info(f"  === END STEP-0 DIAGNOSTICS ===")

            # ---- Additional diagnostics: contact and forces ----
            log.info("  === CONTACT & FORCE DIAGNOSTICS ===")
            log.info(f"  ncon (active contacts) = {data.ncon}")
            for ci in range(min(data.ncon, 5)):
                c = data.contact[ci]
                g1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
                g2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
                log.info(f"    contact[{ci}]: {g1_name} <-> {g2_name}, "
                         f"pos={c.pos}, dist={c.dist:.4f}")
            # Check qacc for large values
            qacc_abs = np.abs(data.qacc)
            log.info(f"  qacc[:6](root) abs max = {qacc_abs[:6].max():.2f}")
            log.info(f"  qacc[6:](joints) abs max = {qacc_abs[6:].max():.2f}")
            log.info(f"  qfrc_actuator abs max = {np.abs(data.qfrc_actuator).max():.2f}")
            log.info(f"  qfrc_constraint abs max = {np.abs(data.qfrc_constraint).max():.2f}")
            log.info(f"  forcelimited actuators: {[i for i in range(model.nu) if model.actuator_forcelimited[i]][:5]}...")
            log.info(f"  actuator_force[:5] = {data.actuator_force[:5]}")
            log.info("  === END CONTACT & FORCE DIAGNOSTICS ===")

        # Optionally apply dynamic PD gains from policy output
        if "stiffness_targets" in out_dict and "damping_targets" in out_dict:
            stiff_out = out_dict["stiffness_targets"].squeeze()
            damp_out = out_dict["damping_targets"].squeeze()
            for i in range(model.nu):
                kp = float(stiff_out[i])
                kd = float(damp_out[i])
                model.actuator_gainprm[i, 0] = kp
                model.actuator_biasprm[i, 1] = -kp
                model.actuator_biasprm[i, 2] = -kd

        # Historical action feedback is a major deployment ambiguity:
        # some ProtoMotions paths feed raw policy actions, while the standalone
        # MuJoCo deployment loop feeds the processed PD target. Keep raw as the
        # default, but allow ablation without editing code.
        if history_action_mode == "processed":
            prev_actions = joint_pos_targets.copy()
        else:
            prev_actions = out_dict["actions"].squeeze().copy()

        # ---- Apply control and step physics ----
        data.ctrl[:] = joint_pos_targets
        for sub_step in range(decimation):
            # Virtual floor spring: emulate IsaacGym contact_offset=0.02m
            # Clear previous external forces and apply virtual spring
            data.xfrc_applied[:] = 0.0
            if virtual_floor_enabled == "true":
                _apply_virtual_floor_spring(model, data, foot_bodies)
            mujoco.mj_step(model, data)
            # NaN guard: break substep loop early if simulation diverged
            if np.any(np.isnan(data.qpos[:7])):
                log.warning(f"  NaN in qpos during substep {sub_step} of step {step_idx}")
                break

        # ---- Progress logging (every 5 steps for debugging) ----
        if step_idx % 5 == 0 or step_idx == T_sim - 1:
            elapsed = time.perf_counter() - t_start
            speed = (step_idx + 1) * control_dt / max(elapsed, 1e-6)
            # Compute tracking error (position MPJPE)
            track_err = np.sqrt(((cur_state["body_pos"] - ref_now["body_pos"])**2).sum(-1)).mean()
            # Contact info
            ncon = data.ncon
            contact_bodies = set()
            for ci in range(ncon):
                c = data.contact[ci]
                g1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
                g2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
                if "floor" in g1_name or "floor" in g2_name:
                    other = g2_name if "floor" in g1_name else g1_name
                    contact_bodies.add(other)
            log.info(f"  step={step_idx:4d}/{T_sim}  "
                     f"root_h={root_h:.3f}  mpjpe={track_err:.4f}  "
                     f"ncon={ncon}  floor_contacts={contact_bodies}  "
                     f"speed={speed:.1f}x")

    # ------------------------------------------------------------------
    # Build results
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    sim_qpos = np.array(sim_qpos_list)
    T_actual = len(sim_qpos)

    log.info(f"  Simulation complete: {T_actual} steps in {elapsed:.1f}s "
             f"({T_actual * control_dt / max(elapsed, 1e-6):.1f}x realtime)")

    # Status
    if fall_frame is not None:
        status = "fell"
    elif root_height_min < 0.4:
        status = "unstable"
    else:
        status = "success"

    stats = {
        "status": status,
        "total_ref_frames": int(T_ref),
        "total_sim_steps": int(T_sim),
        "actual_sim_steps": int(T_actual),
        "fall_frame": int(fall_frame) if fall_frame is not None else None,
        "root_height_min": float(root_height_min),
        "duration_s": float(T_actual * control_dt),
        "sim_time_s": float(elapsed),
        "control_dt": float(control_dt),
        "motion_fps": int(motion_fps),
    }
    log.info(f"  Status: {status} | root_height_min={root_height_min:.3f}")

    return sim_qpos, stats


# ===========================================================================
#  Full Pipeline: NPZ -> RL Sim -> Mesh JSON
# ===========================================================================

def process_single_motion(
    npz_path: str,
    output_dir: str,
    onnx_path: str = _DEFAULT_ONNX,
    mjcf_path: str = _DEFAULT_MJCF,
    yaml_path: str = _DEFAULT_YAML,
    stats_dir: str = None,
) -> dict:
    """Full pipeline: motion_135 NPZ -> RL tracker sim -> SMPL mesh JSON."""
    import yaml

    stem = Path(npz_path).stem
    output_json = Path(output_dir) / f"{stem}.json"

    log.info(f"\n{'=' * 60}")
    log.info(f"Processing: {stem}")
    log.info(f"{'=' * 60}")

    # Load YAML metadata
    with open(yaml_path) as f:
        yaml_meta = yaml.safe_load(f)

    # [1] Decode motion_135
    smpl_pose, transl, fps = decode_motion_135(npz_path)
    T = smpl_pose.shape[0]
    log.info(f"  Decoded: {T} frames @ {fps}fps, duration={T/fps:.1f}s")

    # [2] Y-up -> Z-up
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    # [3] SMPL -> MuJoCo qpos (need model for body_pos_1)
    _temp_model = mujoco.MjModel.from_xml_path(mjcf_path)
    _temp_data = mujoco.MjData(_temp_model)
    body_pos_1 = _temp_model.body_pos[1].copy()
    log.info(f"  body_pos[1] (Pelvis offset): {body_pos_1}")

    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)
    log.info(f"  ref_qpos: {ref_qpos.shape}, "
             f"root_h=[{ref_qpos[:, 2].min():.3f}, {ref_qpos[:, 2].max():.3f}]")

    # [3.5] Ground offset: ensure frame-0 has BILATERAL floor contact in MuJoCo.
    #
    # Problem: MuJoCo requires true geometric intersection for contacts (no
    # proximity detection like IsaacGym's contact_offset=0.02m). Walking
    # motions typically start with one foot mid-stride, so the LOWEST geom
    # may be on one side only. Simply grounding to the lowest single geom
    # gives single-point support → the robot falls immediately.
    #
    # Strategy: Ground the motion so that BOTH foot groups (L and R) are
    # submerged into the floor. We find the lowest geom surface for each
    # foot group and shift so that the HIGHER foot group's lowest geom is
    # at TARGET_GEOM_SURFACE_Z. This guarantees bilateral contact.
    #
    # If one foot is genuinely lifted (swing phase > threshold), we fall back
    # to deeper single-foot grounding to give the stance foot robust support.
    #
    # IMPORTANT: MuJoCo uses HARD contacts — any penetration generates large
    # impulse forces that kick the robot off the ground. Unlike IsaacGym which
    # has contact_offset=0.02m creating "soft" pre-contact, MuJoCo penalizes
    # penetration severely. Place foot geoms AT the floor (z=0) or just barely
    # above. The RL policy will establish proper ground contact through its
    # learned balance behavior.
    TARGET_GEOM_SURFACE_Z = 0.0  # Foot geom surfaces at floor level (margin=0.001 provides pre-contact detection)
    FOOT_SWING_THRESHOLD = 0.08     # If a foot is >8cm higher, it's in swing
    num_bodies_fk = 24  # SMPL bodies (MuJoCo indices 1-24)
    T_ref = ref_qpos.shape[0]

    # --- Identify foot body IDs ---
    # Body ordering: Pelvis=1, L_Hip=2, L_Knee=3, L_Ankle=4, L_Toe=5,
    #                R_Hip=6, R_Knee=7, R_Ankle=8, R_Toe=9, ...
    # (see BODY_NAMES_SMPL in this file)
    left_foot_body_ids = set()
    right_foot_body_ids = set()
    for bid in range(1, _temp_model.nbody):
        bname = mujoco.mj_id2name(_temp_model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in ("L_Ankle", "L_Toe"):
            left_foot_body_ids.add(bid)
        elif bname in ("R_Ankle", "R_Toe"):
            right_foot_body_ids.add(bid)

    # --- Run FK at frame 0 ---
    _temp_data.qpos[:] = ref_qpos[0]
    _temp_data.qvel[:] = 0.0
    mujoco.mj_forward(_temp_model, _temp_data)

    # --- Helper: compute lowest geom surface Z for a set of body IDs ---
    def _compute_lowest_geom_z(body_id_set, model_t, data_t):
        """Find lowest geom surface Z among geoms attached to given bodies."""
        min_z = float("inf")
        min_gname = ""
        for gid in range(model_t.ngeom):
            body_id = model_t.geom_bodyid[gid]
            if body_id not in body_id_set:
                continue
            gtype = int(model_t.geom_type[gid])
            gsize = model_t.geom_size[gid].copy()
            gxpos = data_t.geom_xpos[gid].copy()
            gxmat = data_t.geom_xmat[gid].reshape(3, 3)

            if gtype == 6:  # mjGEOM_BOX
                half_extents = gsize[:3]
                z_extent = (abs(gxmat[2, 0]) * half_extents[0] +
                            abs(gxmat[2, 1]) * half_extents[1] +
                            abs(gxmat[2, 2]) * half_extents[2])
                geom_bottom_z = gxpos[2] - z_extent
            elif gtype == 5:  # mjGEOM_CAPSULE
                radius = gsize[0]
                half_len = gsize[1]
                z_extent = abs(gxmat[2, 2]) * half_len + radius
                geom_bottom_z = gxpos[2] - z_extent
            elif gtype == 3:  # mjGEOM_SPHERE
                geom_bottom_z = gxpos[2] - gsize[0]
            else:
                geom_bottom_z = gxpos[2]

            if geom_bottom_z < min_z:
                min_z = geom_bottom_z
                min_gname = (mujoco.mj_id2name(
                    model_t, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}")
        return min_z, min_gname

    # Compute lowest geom Z for left and right foot groups
    left_min_z, left_min_gname = _compute_lowest_geom_z(
        left_foot_body_ids, _temp_model, _temp_data)
    right_min_z, right_min_gname = _compute_lowest_geom_z(
        right_foot_body_ids, _temp_model, _temp_data)

    # Also compute global minimum (any geom) for logging
    all_body_ids = set(range(1, num_bodies_fk + 1))
    global_min_z, global_min_gname = _compute_lowest_geom_z(
        all_body_ids, _temp_model, _temp_data)

    log.info(f"  Frame-0 grounding (bilateral foot):")
    log.info(f"    L foot lowest geom Z = {left_min_z:.4f}m ({left_min_gname})")
    log.info(f"    R foot lowest geom Z = {right_min_z:.4f}m ({right_min_gname})")
    log.info(f"    Global lowest geom Z = {global_min_z:.4f}m ({global_min_gname})")

    # Determine grounding reference:
    # Use the HIGHER of the two foot minimums — this ensures that when we
    # shift down, BOTH feet will be at or below the target (the higher one
    # hits target, the lower one is even deeper).
    foot_height_diff = abs(left_min_z - right_min_z)
    if foot_height_diff > FOOT_SWING_THRESHOLD:
        # One foot is clearly in swing — ground based on the LOWER (stance) foot
        # but use deeper penetration for stability
        grounding_ref_z = min(left_min_z, right_min_z)
        stance_side = "L" if left_min_z < right_min_z else "R"
        log.info(f"    Swing phase detected (diff={foot_height_diff:.3f}m > "
                 f"{FOOT_SWING_THRESHOLD}m)")
        log.info(f"    Grounding from {stance_side} foot (stance)")
        # Same target for single-foot (no deeper penetration needed with MuJoCo hard contacts)
        effective_target = TARGET_GEOM_SURFACE_Z
    else:
        # Both feet are close to ground — ground using the LOWER one (global min)
        # to ensure ZERO penetration. The higher foot will be slightly above ground
        # (by ~foot_height_diff), which is fine — RL policy tolerates small gaps
        # (trained with contact_offset=0.02m in IsaacGym).
        # Previously used max() which caused the lower foot to penetrate, creating
        # large upward contact forces that bounced the robot off the ground.
        grounding_ref_z = min(left_min_z, right_min_z)
        log.info(f"    Both feet near ground (diff={foot_height_diff:.3f}m)")
        log.info(f"    Grounding from LOWER foot to ensure zero penetration")
        effective_target = TARGET_GEOM_SURFACE_Z

    height_shift = effective_target - grounding_ref_z
    log.info(f"    Height shift = {height_shift:+.4f}m "
             f"(target: {effective_target}m)")

    if abs(height_shift) > 0.0001:
        ref_qpos[:, 2] += height_shift
        log.info(f"    After grounding: root_h=[{ref_qpos[:, 2].min():.3f}, "
                 f"{ref_qpos[:, 2].max():.3f}]")

    del _temp_model, _temp_data

    # [4-5] Run RL tracker simulation
    sim_qpos, stats = run_rl_tracker(
        ref_qpos=ref_qpos,
        motion_fps=fps,
        onnx_path=onnx_path,
        mjcf_path=mjcf_path,
        yaml_meta=yaml_meta,
    )

    T_sim = stats["actual_sim_steps"]
    control_dt = stats["control_dt"]
    status_str = ("COMPLETED" if stats["status"] == "success"
                  else f"FELL at step {stats['fall_frame']}")
    log.info(f"  Result: {T_sim} steps ({T_sim * control_dt:.1f}s) -- {status_str}")

    # [6] Export: MuJoCo qpos -> SMPL axis-angle -> Z-up -> Y-up
    smpl_pose_sim, transl_sim = qpos_to_smpl(sim_qpos, body_pos_1)
    smpl_pose_yup, transl_yup = zup_to_yup(smpl_pose_sim, transl_sim)

    # Resample to original FPS for consistent visualization
    # sim is at 50Hz (control_dt=0.02), need to resample to motion fps (e.g. 30)
    output_fps = fps
    sim_fps = 1.0 / control_dt
    if abs(sim_fps - output_fps) > 0.5:
        # Resample from sim_fps to output_fps
        T_out = int(T_sim * control_dt * output_fps)
        indices = np.linspace(0, T_sim - 1, T_out).astype(int)
        smpl_pose_yup = smpl_pose_yup[indices]
        transl_yup = transl_yup[indices]
        log.info(f"  Resampled: {T_sim} steps @ {sim_fps:.0f}Hz -> "
                 f"{T_out} frames @ {output_fps}Hz")
    else:
        T_out = T_sim

    # [7] Generate mesh JSON
    result = smpl_to_mesh_json(smpl_pose_yup, transl_yup, output_fps)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    size_kb = output_json.stat().st_size / 1024
    log.info(f"  Saved: {output_json} ({size_kb:.1f} KB, {T_out} frames)")

    # Save stats
    if stats_dir:
        Path(stats_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(stats_dir) / f"{stem}.json", "w") as f:
            json.dump(stats, f, indent=2)

    return stats


# ===========================================================================
#  Case Filtering
# ===========================================================================

def is_flat_ground(prompt: str) -> bool:
    """Check if a motion prompt is suitable for flat-ground simulation."""
    prompt_lower = prompt.lower()
    return not any(kw in prompt_lower for kw in EXCLUDE_KEYWORDS)


# ===========================================================================
#  CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SMPL RL tracker: physics simulation with trained ONNX policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz-dir", type=str,
                       help="Directory of motion_135 NPZ files")
    group.add_argument("--npz-file", type=str,
                       help="Single NPZ file to process")

    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for physics-simulated mesh JSONs")
    parser.add_argument("--onnx", type=str, default=_DEFAULT_ONNX,
                        help="Path to SMPL ONNX tracker model")
    parser.add_argument("--mjcf", type=str, default=_DEFAULT_MJCF,
                        help="Path to SMPL humanoid MJCF XML")
    parser.add_argument("--yaml", type=str, default=_DEFAULT_YAML,
                        help="Path to unified_pipeline.yaml metadata")
    parser.add_argument("--meta-dir", type=str, default=None,
                        help="Metadata JSON dir (for case filtering)")
    parser.add_argument("--stats-dir", type=str, default=None,
                        help="Directory to save per-motion sim stats")
    parser.add_argument("--filter-flat-ground", action="store_true",
                        help="Only process flat-ground suitable motions")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip already processed files")
    parser.add_argument("--max-motions", type=int, default=None,
                        help="Max motions to process")

    args = parser.parse_args()

    # Collect NPZ files
    if args.npz_file:
        npz_files = [Path(args.npz_file)]
    else:
        npz_dir = Path(args.npz_dir)
        npz_files = sorted(f for f in npz_dir.iterdir() if f.suffix == ".npz")

    log.info(f"Found {len(npz_files)} NPZ files")

    # Flat-ground filter
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
                filtered.append(npz_path)
        npz_files = filtered
        log.info(f"Flat-ground filter: {len(npz_files)} kept, {excluded} excluded")

    if args.max_motions:
        npz_files = npz_files[:args.max_motions]

    if not npz_files:
        log.info("No files to process.")
        return

    # Process all
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
                str(npz_path),
                args.output_dir,
                onnx_path=args.onnx,
                mjcf_path=args.mjcf,
                yaml_path=args.yaml,
                stats_dir=args.stats_dir,
            )
            all_stats[npz_path.stem] = stats
            success += 1
            if stats["status"] != "success":
                fell += 1
        except Exception as e:
            import traceback
            log.error(f"\n  FAILED: {npz_path.stem}: {e}")
            traceback.print_exc()
            failed += 1

    # Summary
    log.info(f"\n{'=' * 60}")
    log.info(f"SUMMARY")
    log.info(f"{'=' * 60}")
    log.info(f"Total: {len(npz_files)}, Success: {success}, Failed: {failed}, "
             f"Skipped: {skipped}")
    log.info(f"Completed without fall: {success - fell}/{success} "
             f"({100 * (success - fell) / max(success, 1):.0f}%)")

    if all_stats:
        heights = [s["root_height_min"] for s in all_stats.values()]
        log.info(f"Root height min: mean={np.mean(heights):.3f}, "
                 f"min={np.min(heights):.3f}")

    # Save aggregate summary
    if args.stats_dir:
        Path(args.stats_dir).mkdir(parents=True, exist_ok=True)
        summary = {
            "total": len(npz_files),
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "fell": fell,
            "per_motion": all_stats,
        }
        summary_path = Path(args.stats_dir) / "_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info(f"\nStats saved: {summary_path}")


if __name__ == "__main__":
    main()
