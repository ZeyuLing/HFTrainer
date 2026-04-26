#!/usr/bin/env python3
"""Run KIMODO evaluation on ALL M2M tasks (E2-E8, E10, E14-E16).

Bridges SMPL-22 eval data <-> KIMODO SOMA-30 skeleton via rotation-based
retargeting (global rotation transfer + SOMA30 FK for correct proportions).

Usage:
    python tools/run_kimodo_all_tasks.py --all-tasks --max-samples 80
    python tools/run_kimodo_all_tasks.py --tasks E2 E3 E5 --max-samples 50
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KIMODO_ROOT = PROJECT_ROOT / "ref_repo" / "KIMODO" / "kimodo"
sys.path.insert(0, str(KIMODO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

KIMODO_MODEL = "kimodo-soma-rp"
DIFFUSION_STEPS = 100
MOTION_DATA_DIR = str(PROJECT_ROOT / "data" / "hymotion_data")

# ============================================================================
# Skeleton mapping: SMPL-22 <-> SOMA-30/77
# ============================================================================

# SOMA-77 indices that correspond to SMPL-22 joints
SOMA77_TO_SMPL22 = [
    0,   # pelvis     -> Hips
    67,  # left_hip   -> LeftLeg
    72,  # right_hip  -> RightLeg
    1,   # spine1     -> Spine1
    68,  # left_knee  -> LeftShin
    73,  # right_knee -> RightShin
    2,   # spine2     -> Spine2
    69,  # left_ankle -> LeftFoot
    74,  # right_ankle-> RightFoot
    3,   # spine3     -> Chest
    70,  # left_foot  -> LeftToeBase
    75,  # right_foot -> RightToeBase
    4,   # neck       -> Neck1
    11,  # left_collar-> LeftShoulder
    39,  # right_collar-> RightShoulder
    6,   # head       -> Head
    12,  # left_shoulder -> LeftArm
    40,  # right_shoulder-> RightArm
    13,  # left_elbow -> LeftForeArm
    41,  # right_elbow-> RightForeArm
    14,  # left_wrist -> LeftHand
    42,  # right_wrist-> RightHand
]

# SMPLX22 joint names (order matches 135-dim motion)
SMPLX22_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
]
# SOMA30 joint names
SOMA30_NAMES = [
    'Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'Jaw',
    'LeftEye', 'RightEye', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'LeftHandThumbEnd', 'LeftHandMiddleEnd', 'RightShoulder', 'RightArm',
    'RightForeArm', 'RightHand', 'RightHandThumbEnd', 'RightHandMiddleEnd',
    'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase', 'RightLeg', 'RightShin',
    'RightFoot', 'RightToeBase',
]
# SMPLX22 -> SOMA30 name correspondence (22 matched joints)
SMPLX_TO_SOMA_NAME = {
    'pelvis': 'Hips', 'left_hip': 'LeftLeg', 'right_hip': 'RightLeg',
    'spine1': 'Spine1', 'left_knee': 'LeftShin', 'right_knee': 'RightShin',
    'spine2': 'Spine2', 'left_ankle': 'LeftFoot', 'right_ankle': 'RightFoot',
    'spine3': 'Chest', 'left_foot': 'LeftToeBase', 'right_foot': 'RightToeBase',
    'neck': 'Neck1', 'left_collar': 'LeftShoulder', 'right_collar': 'RightShoulder',
    'head': 'Head', 'left_shoulder': 'LeftArm', 'right_shoulder': 'RightArm',
    'left_elbow': 'LeftForeArm', 'right_elbow': 'RightForeArm',
    'left_wrist': 'LeftHand', 'right_wrist': 'RightHand',
}
_SOMA30_IDX = {n: i for i, n in enumerate(SOMA30_NAMES)}
# SMPLX22[i] -> SOMA30[j] for the 22 matching joints
SMPLX22_TO_SOMA30 = [_SOMA30_IDX[SMPLX_TO_SOMA_NAME[n]] for n in SMPLX22_NAMES]

# SMPL-22 body part groups
UPPER_JOINTS = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


def soma77_to_smpl22(posed_joints_77: np.ndarray) -> np.ndarray:
    """Extract SMPL-22 positions from SOMA-77. (T, 77, 3) -> (T, 22, 3)"""
    return posed_joints_77[:, SOMA77_TO_SMPL22]


# ============================================================================
# Rotation-based retargeting: SMPL-22 -> SOMA-30
# ============================================================================

def _slerp_rot_matrices(R1, R2, t):
    """Spherical linear interpolation between rotation matrices.

    Args:
        R1, R2: (..., 3, 3) rotation matrices
        t: interpolation factor in [0, 1]

    Returns:
        (..., 3, 3) interpolated rotation matrices
    """
    import torch
    # R_delta = R1^T @ R2
    R_delta = torch.einsum("...ij,...ik->...jk", R1, R2)  # R1^T @ R2
    tr = R_delta[..., 0, 0] + R_delta[..., 1, 1] + R_delta[..., 2, 2]
    cos_angle = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)

    small = angle.abs() < 1e-6
    sin_angle = torch.sin(angle).clamp(min=1e-8)
    axis = torch.stack([
        R_delta[..., 2, 1] - R_delta[..., 1, 2],
        R_delta[..., 0, 2] - R_delta[..., 2, 0],
        R_delta[..., 1, 0] - R_delta[..., 0, 1],
    ], dim=-1) / (2.0 * sin_angle.unsqueeze(-1))
    axis = torch.nn.functional.normalize(axis, dim=-1)

    scaled_angle = angle * t
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1)
    K = K.reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, device=R1.device, dtype=R1.dtype).expand_as(K)
    sin_t = torch.sin(scaled_angle)[..., None, None]
    cos_t = torch.cos(scaled_angle)[..., None, None]
    R_interp = eye + sin_t * K + (1 - cos_t) * (K @ K)

    R_interp = torch.where(small[..., None, None], eye, R_interp)
    return R1 @ R_interp


def smpl22_to_soma30_retarget(motion_135, bone_offsets):
    """Rotation-based retarget: SMPL-22 motion -> SOMA-30 global rotations + positions.

    Transfers global rotations from SMPLX22 -> SOMA30, then runs SOMA30 FK to
    produce joint positions with correct SOMA30 bone lengths/proportions.

    Steps:
        1. Parse 135-dim motion -> rot6d -> rotation matrices -> SMPLX22 FK -> global rots
        2. Map global rotations SMPLX22 -> SOMA30 (22 matched + 8 interpolated)
        3. Convert to SOMA30 local rotations via global_rots_to_local_rots()
        4. Ground alignment (correct root Y for SOMA30's shorter legs)
        5. SOMA30 FK -> posed joints with correct SOMA30 proportions

    Args:
        motion_135: (T, 135) denormalized motion [trans(3), rot6d(132)].
        bone_offsets: (22, 3) SMPL-22 bone offsets (numpy).

    Returns:
        soma30_global_rots: (T, 30, 3, 3) torch tensor.
        soma30_positions: (T, 30, 3) torch tensor.
    """
    import torch
    from hftrainer.pipelines.motion.differentiable_fk import (
        differentiable_fk, rot6d_to_rotmat_row_major,
    )
    from kimodo.skeleton.definitions import SOMASkeleton30, SMPLXSkeleton22
    from kimodo.skeleton.transforms import global_rots_to_local_rots

    smplx22 = SMPLXSkeleton22()
    soma30 = SOMASkeleton30()

    if isinstance(motion_135, np.ndarray):
        motion_135 = torch.from_numpy(motion_135).float()
    if isinstance(bone_offsets, np.ndarray):
        bone_offsets = torch.from_numpy(bone_offsets).float()

    T = motion_135.shape[0]

    # Step 1: Parse 135-dim -> SMPLX22 FK -> global rotations
    translation = motion_135[:, :3]                       # (T, 3)
    rot6d = motion_135[:, 3:135].reshape(T, 22, 6)       # (T, 22, 6)
    local_rotmat = rot6d_to_rotmat_row_major(rot6d)       # (T, 22, 3, 3)
    smplx_world_pos, smplx_global_rots = differentiable_fk(
        local_rotmat, translation, bone_offsets
    )
    # smplx_global_rots: (T, 22, 3, 3), smplx_world_pos: (T, 22, 3)

    # Step 2: Map global rotations SMPLX22 -> SOMA30
    # Global rotations represent world-space bone orientations, can be directly
    # copied for matched joints. SOMA30 FK chain decomposes them using its own
    # parent hierarchy and neutral bone directions.
    soma_global_rots = torch.eye(3, dtype=local_rotmat.dtype, device=local_rotmat.device)
    soma_global_rots = soma_global_rots.unsqueeze(0).unsqueeze(0).expand(T, 30, 3, 3).clone()

    for smplx_idx, soma_idx in enumerate(SMPLX22_TO_SOMA30):
        soma_global_rots[:, soma_idx, :, :] = smplx_global_rots[:, smplx_idx, :, :]

    # Interpolate unmapped joints:
    # Neck2(5): SLERP between Neck1(4) and Head(6) at t=0.5
    soma_global_rots[:, 5] = _slerp_rot_matrices(
        soma_global_rots[:, 4], soma_global_rots[:, 6], 0.5
    )
    # Jaw(7), LeftEye(8), RightEye(9): children of Head(6)
    soma_global_rots[:, 7] = soma_global_rots[:, 6]
    soma_global_rots[:, 8] = soma_global_rots[:, 6]
    soma_global_rots[:, 9] = soma_global_rots[:, 6]
    # LeftHandThumbEnd(14), LeftHandMiddleEnd(15): children of LeftHand(13)
    soma_global_rots[:, 14] = soma_global_rots[:, 13]
    soma_global_rots[:, 15] = soma_global_rots[:, 13]
    # RightHandThumbEnd(20), RightHandMiddleEnd(21): children of RightHand(19)
    soma_global_rots[:, 20] = soma_global_rots[:, 19]
    soma_global_rots[:, 21] = soma_global_rots[:, 19]

    # Step 3: Convert to SOMA30 local rotations
    soma_local_rots = global_rots_to_local_rots(soma_global_rots, soma30)

    # Step 4: Ground alignment — correct root Y so SOMA30 feet match SMPLX22 ground.
    # SOMA30 has shorter legs (~2.5cm), so we lower the root to compensate.
    smplx_centered = smplx22.neutral_joints - smplx22.neutral_joints[smplx22.root_idx]
    soma_centered = soma30.neutral_joints - soma30.neutral_joints[soma30.root_idx]

    smplx_foot_indices = [smplx22.bone_index[n] for n in
                          ['left_foot', 'right_foot', 'left_ankle', 'right_ankle']]
    soma_foot_indices = [soma30.bone_index[n] for n in
                         ['LeftToeBase', 'RightToeBase', 'LeftFoot', 'RightFoot']]
    smplx_foot_min_y = smplx_centered[smplx_foot_indices, 1].min()
    soma_foot_min_y = soma_centered[soma_foot_indices, 1].min()
    foot_offset_y = (soma_foot_min_y - smplx_foot_min_y).item()

    soma_root_pos = translation.clone()
    soma_root_pos[:, 1] -= foot_offset_y

    # Step 5: SOMA30 FK -> posed joints with correct SOMA30 proportions
    soma_global_rots_fk, soma_joints, _ = soma30.fk(soma_local_rots, soma_root_pos)

    # Step 6: Dynamic per-clip Y alignment — anchor SOMA frame-0 feet to the
    # SMPL frame-0 feet height. The static `foot_offset_y` from step 4 is
    # computed from neutral T-pose, but the actual frame-0 pose may have bent
    # legs, one-foot-up, or otherwise non-T-pose configuration. Without this
    # correction the SOMA condition frames floated 5–15 cm above the SMPL
    # ground truth (root cause of "bouncy" KIMODO output reported on E3
    # dashboard 2026-04-26), because KIMODO's diffusion saw constraints
    # outside its grounded-motion training prior and pushed the rest of the
    # generated motion up to "match".
    smpl_min_y_f0 = smplx_world_pos[0, smplx_foot_indices, 1].min()
    soma_min_y_f0 = soma_joints[0, soma_foot_indices, 1].min()
    y_delta = (soma_min_y_f0 - smpl_min_y_f0).item()
    if abs(y_delta) > 1e-4:
        soma_joints = soma_joints.clone()
        soma_joints[..., 1] -= y_delta
        # Global rotations are unaffected by a pure Y translation; only joint
        # positions need updating. The downstream FullBodyConstraintSet uses
        # `soma_joints` (positions) and `soma_global_rots_fk` (rotations) so
        # this consistent shift keeps the constraint physically grounded.

    return soma_global_rots_fk, soma_joints


# ============================================================================
# Canonicalization helpers (match KIMODO's training-time canonical form)
# ============================================================================
#
# KIMODO was trained on data where, for each clip, frame 0's `smooth_root_pos`
# is at (0, y, 0) in XZ and the heading angle is 0 (i.e. the subject faces
# along +Z in KIMODO's convention, since heading = atan2(diff_z, -diff_x) of
# right_hip - left_hip). When we retarget a world-space SMPL motion to SOMA30
# and hand the raw world-space joint positions + rotations to
# EndEffectorConstraintSet / Root2DConstraintSet / FullBodyConstraintSet, the
# constraints land at arbitrary world positions and headings — outside the
# training distribution — and the generated motion drifts/floats.
#
# KIMODO's own T2M pipeline avoids this because it either (a) uses no
# positional constraint or (b) internally calls `translate_2d_to_zero` on the
# observed_motion before normalization. But that internal helper only zeroes
# smooth_root_2d; it does NOT rotate the constraints to zero-heading, and it
# does NOT touch `global_joints_positions` or `global_joints_rots` from
# Full/End-Effector constraint sets (which are pre-imputed world positions).
#
# Fix: compute a per-clip `(R_yaw, t_xz)` canonical transform from frame 0
# of the retargeted SOMA30 motion, apply it to the SOMA30 rotations +
# positions BEFORE building constraints, run KIMODO, then invert the
# transform on the output so metrics/viz live in the original world.

def _rot_y(theta):
    """3x3 rotation around Y-axis."""
    import torch
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([
        torch.stack([c,  z, s], dim=-1),
        torch.stack([z,  o, z], dim=-1),
        torch.stack([-s, z, c], dim=-1),
    ], dim=-2)


def kimodo_compute_canon_transform(soma30_pos, skeleton):
    """Compute the rigid (yaw, XZ-translation) transform that maps frame 0
    of the retargeted SOMA30 motion into KIMODO's canonical frame.

    The canonical frame is defined by:
      - frame-0 root (pelvis) X, Z = 0
      - frame-0 heading angle = 0 (R-hip minus L-hip lying along the axis
        where atan2(dz, -dx) == 0, i.e. dx = -1 so the subject faces +Z)

    Args:
        soma30_pos: (T, 30, 3) retargeted SOMA30 world joint positions.
        skeleton: SOMASkeleton30 instance (for hip_joint_idx).

    Returns:
        (R_yaw, t_xz, heading0) where:
          R_yaw: (3, 3) yaw rotation matrix that maps world -> canonical
                 (left-multiply positions, rotations).
          t_xz : (3,) translation applied AFTER rotation (only XZ non-zero).
          heading0: scalar heading angle of frame 0 (radians), for logging.
    """
    import torch
    r_hip_idx, l_hip_idx = skeleton.hip_joint_idx
    diff = soma30_pos[0, r_hip_idx] - soma30_pos[0, l_hip_idx]
    heading0 = torch.atan2(diff[2], -diff[0])
    R_yaw = _rot_y(-heading0)  # canonical = world rotated by -heading0
    root_xz0 = soma30_pos[0, 0]  # pelvis at frame 0 (3,) world
    root_xz0_rot = R_yaw @ root_xz0
    # After rotation we want (x, y_any, z) = (0, y_any, 0) → translate by
    # -root_xz0_rot in X, Z only (Y preserved).
    t_xz = torch.tensor([-root_xz0_rot[0], 0.0, -root_xz0_rot[2]],
                        dtype=soma30_pos.dtype, device=soma30_pos.device)
    return R_yaw, t_xz, heading0


def kimodo_apply_canon(soma30_rots, soma30_pos, R_yaw, t_xz):
    """Apply canonical transform to SOMA30 motion.

    Positions : p_canon = R_yaw @ p + t_xz
    Rotations : R_canon = R_yaw @ R_global  (global rotations)

    Args:
        soma30_rots: (T, 30, 3, 3) global rotation matrices.
        soma30_pos:  (T, 30, 3)    world joint positions.

    Returns:
        (rots_canon, pos_canon) in canonical frame.
    """
    import torch
    pos_canon = torch.einsum('ij,tnj->tni', R_yaw, soma30_pos) + t_xz
    rots_canon = torch.einsum('ij,tnjk->tnik', R_yaw, soma30_rots)
    return rots_canon, pos_canon


def kimodo_invert_canon_positions(posed_joints_canon, R_yaw, t_xz):
    """Invert canonical transform on a (T, J, 3) positions array.

    Inverse:  p_world = R_yaw^T @ (p_canon - t_xz)
    """
    import torch
    if isinstance(posed_joints_canon, np.ndarray):
        posed_joints_canon = torch.from_numpy(posed_joints_canon).float()
    R_inv = R_yaw.transpose(-1, -2)
    p_world = torch.einsum(
        'ij,tnj->tni', R_inv,
        (posed_joints_canon - t_xz))
    return p_world


# ============================================================================
# Constraint builders per task
# ============================================================================

def build_constraints_e2(skeleton, soma30_rots, soma30_pos, T, setting, caption=""):
    """E2 In-betweening: six settings mirroring the backend v2 ablation.

    Each setting chooses which temporal region is given as GT context
    (FullBodyConstraintSet on those frame indices); KIMODO then solves
    for the rest of the frames.
    """
    from kimodo.constraints import FullBodyConstraintSet
    import math
    import torch

    keep_start = 0
    keep_end = 0
    if setting == 'start_1f':
        keep_start = 1
    elif setting == 'end_1f':
        keep_end = 1
    elif setting == 'both_1f':
        keep_start = 1
        keep_end = 1
    elif setting == 'pre20':
        keep_start = max(1, math.ceil(T * 0.20))
    elif setting == 'post20':
        keep_end = max(1, math.ceil(T * 0.20))
    elif setting == 'mid60':
        keep_start = max(1, math.ceil(T * 0.20))
        keep_end = max(1, math.ceil(T * 0.20))
    # Legacy fallthrough (old A/B/C) — symmetric 5-10% keep.
    elif setting == 'A':
        keep_start = keep_end = int(T * 0.1)
    elif setting == 'B':
        keep_start = keep_end = int(T * 0.05)
    elif setting == 'C':
        keep_start = int(T * 0.2)
        keep_end = max(5, int(T * 0.03))
    else:
        keep_start = keep_end = max(5, int(T * 0.1))

    keep_start = max(0, min(keep_start, T))
    keep_end = max(0, min(keep_end, T - keep_start))

    frames: list = []
    if keep_start > 0:
        frames.extend(range(keep_start))
    if keep_end > 0:
        frames.extend(range(T - keep_end, T))
    if not frames:
        # Shouldn't happen, but guard anyway — KIMODO needs at least one
        # constrained frame to have a signal.
        frames = [0]
    frame_idx = torch.tensor(frames, dtype=torch.long)

    constraint = FullBodyConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=soma30_pos[frame_idx],
        global_joints_rots=soma30_rots[frame_idx],
        to_crop=False,
    )
    return [constraint]


def build_constraints_e3(skeleton, soma30_rots, soma30_pos, T, setting, caption=""):
    """E3 Keyframe interpolation: keep every K-th frame.

    2026-04-26: aligned with backend m2m_eval_tasks.py E3 v2 settings.
      every_5f  -> K=5    every_10f -> K=10
      A         -> K=30   B          -> K=60
      C         -> K=15   D          -> adaptive (fallback to K=30 here;
                                       KIMODO doesn't support adaptive
                                       so we approximate with uniform K).
    Old (incorrect) mapping had B=15, C=5, D=60.
    """
    from kimodo.constraints import FullBodyConstraintSet
    import torch

    intervals = {
        'every_5f': 5, 'every_10f': 10,
        'A': 30, 'B': 60, 'C': 15, 'D': 30,
    }
    K = intervals.get(setting, 30)
    frames = list(range(0, T, K))
    if frames[-1] != T - 1:
        frames.append(T - 1)

    frame_idx = torch.tensor(frames, dtype=torch.long)

    constraint = FullBodyConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=soma30_pos[frame_idx],
        global_joints_rots=soma30_rots[frame_idx],
        to_crop=False,
    )
    return [constraint]


def build_constraints_e4(skeleton, soma30_rots, soma30_pos, T, setting, caption=""):
    """E4 End-effector: constrain specific joints at intervals.

    Settings align with backend m2m_eval_tasks.py E4 (pure EE — D/E keypose
    settings have been moved to E7 Bi-directional Pose Completion).
    """
    from kimodo.constraints import EndEffectorConstraintSet
    import torch

    # setting → (SOMA joint names, SMPL-22 indices [for logging], frame interval)
    joint_map = {
        'A_rhand_sparse':  (['RightHand'], [21], 10),
        'B_ankles_sparse': (['LeftFoot', 'RightFoot'], [7, 8], 15),
        'C_rhand_lfoot':   (['RightHand', 'LeftToeBase'], [21, 10], 15),
        'D_both_hands':    (['LeftHand', 'RightHand'], [20, 21], 10),
        'E_all4_sparse':   (['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot'],
                            [20, 21, 7, 8], 20),
        'F_rhand_dense':   (['RightHand'], [21], 5),
    }
    if setting not in joint_map:
        # Legacy / unknown → fall back to A_rhand_sparse
        soma_names, _, interval = (['RightHand'], [21], 10)
    else:
        soma_names, _, interval = joint_map[setting]

    frames = list(range(0, T, interval))
    frame_idx = torch.tensor(frames, dtype=torch.long)

    # KIMODO EndEffectorConstraintSet also requires smooth_root_2d (see
    # ref_repo/KIMODO/CLAUDE.md §2.5 — global positions need root XZ too).
    smooth_root_2d = soma30_pos[frame_idx, 0, :][:, [0, 2]]  # (K, 2)

    constraint = EndEffectorConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=soma30_pos[frame_idx],
        global_joints_rots=soma30_rots[frame_idx],
        smooth_root_2d=smooth_root_2d,
        joint_names=soma_names,
        to_crop=False,
    )
    return [constraint]


def build_constraints_e5(skeleton, soma30_rots, soma30_pos, T, setting, caption=""):
    """E5 Trajectory following: constrain root XZ."""
    from kimodo.constraints import Root2DConstraintSet
    import torch

    intervals = {'A': 1, 'B': 30, 'C': 1}
    K = intervals.get(setting, 1)
    frames = list(range(0, T, K))
    frame_idx = torch.tensor(frames, dtype=torch.long)

    # Root XZ from SOMA30 positions (root = joint 0 = Hips)
    root_xz = soma30_pos[:, 0, [0, 2]]

    heading = None
    if setting == 'C':
        # Estimate heading from root trajectory
        root_x = soma30_pos[:, 0, 0].numpy()
        root_z = soma30_pos[:, 0, 2].numpy()
        dx = np.diff(root_x, prepend=root_x[0])
        dz = np.diff(root_z, prepend=root_z[0])
        angle = np.arctan2(dx, dz)
        heading = torch.stack([torch.cos(torch.from_numpy(angle).float()),
                               torch.sin(torch.from_numpy(angle).float())], dim=-1)

    constraint = Root2DConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        smooth_root_2d=root_xz[frame_idx],
        global_root_heading=heading[frame_idx] if heading is not None else None,
    )
    return [constraint]


def build_constraints_e6(skeleton, soma30_rots, soma30_pos, gt_pos_22, T, setting,
                          caption=""):
    """E6 Foot ground contact: constrain ankles at contact frames.

    Uses SMPL-22 GT positions for foot contact detection (ankle height),
    but SOMA30 retargeted data for constraint building.
    """
    from kimodo.constraints import EndEffectorConstraintSet
    import torch

    # Detect foot contact frames from GT ankle heights (SMPL-22 space)
    l_ankle_y = gt_pos_22[:, 7, 1]
    r_ankle_y = gt_pos_22[:, 8, 1]
    threshold = 0.05  # 5cm above ground
    contact_l = l_ankle_y < threshold
    contact_r = r_ankle_y < threshold
    contact = contact_l | contact_r
    frames = np.where(contact)[0].tolist()
    if not frames:
        frames = [0]

    frame_idx = torch.tensor(frames, dtype=torch.long)

    constraint = EndEffectorConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=soma30_pos[frame_idx],
        global_joints_rots=soma30_rots[frame_idx],
        joint_names=['LeftFoot', 'RightFoot'],
        to_crop=False,
    )
    return [constraint]


def build_constraints_e7(skeleton, soma30_rots, soma30_pos, T, setting, caption=""):
    """E7 First-frame continuation: keep frame 0."""
    return build_constraints_e3(skeleton, soma30_rots, soma30_pos, T, 'A', caption)


def build_constraints_e8(skeleton, soma30_rots, soma30_pos, T, setting, caption=""):
    """E8 Loop animation.

    Setting A: first = last frame (classic loop).
    Settings B/C/D: loop completion — given full GT, constrain all GT frames
    + last frame = first frame pose.
    """
    from kimodo.constraints import FullBodyConstraintSet
    import torch

    if setting == 'A':
        # Classic loop: constrain first and last frame, both with frame-0 pose
        frames = [0, T - 1]
        frame_idx = torch.tensor(frames, dtype=torch.long)
        # Use frame 0 data for both constraint frames (loop)
        loop_rots = torch.stack([soma30_rots[0], soma30_rots[0]], dim=0)
        loop_pos = torch.stack([soma30_pos[0], soma30_pos[0]], dim=0)

        constraint = FullBodyConstraintSet(
            skeleton,
            frame_indices=frame_idx,
            global_joints_positions=loop_pos,
            global_joints_rots=loop_rots,
            to_crop=False,
        )
        return [constraint]
    else:
        # Loop completion (B/C/D): constrain all GT frames + last frame = first
        append_map = {'B': 30, 'C': 60, 'D': 90}
        N_append = append_map.get(setting, 30)
        T_total = T + N_append

        # All GT frames + the last frame (= first frame pose)
        frames = list(range(T)) + [T_total - 1]
        frame_idx = torch.tensor(frames, dtype=torch.long)
        # Concat GT SOMA30 data + frame-0 data for the loop-back frame
        constraint_rots = torch.cat([soma30_rots[:T], soma30_rots[0:1]], dim=0)
        constraint_pos = torch.cat([soma30_pos[:T], soma30_pos[0:1]], dim=0)

        constraint = FullBodyConstraintSet(
            skeleton,
            frame_indices=frame_idx,
            global_joints_positions=constraint_pos,
            global_joints_rots=constraint_rots,
            to_crop=False,
        )
        return [constraint]


def build_constraints_e10(skeleton, soma30_rots, soma30_pos, T, setting, caption=""):
    """E10 Part-level editing: constrain body-part joints."""
    from kimodo.constraints import EndEffectorConstraintSet
    import torch

    if setting == 'A':
        soma_names = ['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1',
                      'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                      'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'Head']
    elif setting == 'B':
        soma_names = ['LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase',
                      'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase']
    else:
        soma_names = ['Hips']

    # Constrain every frame for the kept joints
    frames = list(range(T))
    frame_idx = torch.tensor(frames, dtype=torch.long)

    constraint = EndEffectorConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=soma30_pos[frame_idx],
        global_joints_rots=soma30_rots[frame_idx],
        joint_names=soma_names,
        to_crop=False,
    )
    return [constraint]


def build_constraints_e14(skeleton, soma30_rots_a, soma30_pos_a,
                           soma30_rots_b, soma30_pos_b,
                           T, setting, N_cond, N_transition, caption="",
                           N_cond_a=None, N_cond_b=None):
    """E14 Transition stitching: constrain A tail + B head, generate middle.

    Sequence: [A_tail(N_cond_a) | transition(N_transition) | B_head(N_cond_b)]
    Constrain A_tail and B_head frames with full body.

    2026-04-26: now accepts asymmetric N_cond_a / N_cond_b to match the
    v5 adaptive context policy (compute_cond_length per side). Falls
    back to legacy symmetric N_cond if the new args aren't provided.
    """
    from kimodo.constraints import FullBodyConstraintSet
    import torch

    if N_cond_a is None:
        N_cond_a = N_cond
    if N_cond_b is None:
        N_cond_b = N_cond

    a_tail_rots = soma30_rots_a[-N_cond_a:]
    a_tail_pos = soma30_pos_a[-N_cond_a:]
    b_head_rots = soma30_rots_b[:N_cond_b]
    b_head_pos = soma30_pos_b[:N_cond_b]

    frames_a = list(range(N_cond_a))
    frames_b = list(range(N_cond_a + N_transition, T))
    frames = frames_a + frames_b

    frame_idx = torch.tensor(frames, dtype=torch.long)
    constraint_rots = torch.cat([a_tail_rots, b_head_rots], dim=0)
    constraint_pos = torch.cat([a_tail_pos, b_head_pos], dim=0)

    constraint = FullBodyConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=constraint_pos,
        global_joints_rots=constraint_rots,
        to_crop=False,
    )
    return [constraint]


def build_constraints_e15(skeleton, soma30_rots, soma30_pos,
                           soma30_rots_target, soma30_pos_target,
                           T, setting, N_cond_tail, N_transition, caption=""):
    """E15 Transition to target first frame: constrain motion tail + target first.

    Sequence: [motion_tail(N_cond_tail) | transition(N_transition) | target_first(1)]
    """
    from kimodo.constraints import FullBodyConstraintSet
    import torch

    # Motion tail + target first frame SOMA30 data
    tail_rots = soma30_rots[-N_cond_tail:]       # (N_cond_tail, 30, 3, 3)
    tail_pos = soma30_pos[-N_cond_tail:]         # (N_cond_tail, 30, 3)
    target_first_rots = soma30_rots_target[0:1]  # (1, 30, 3, 3)
    target_first_pos = soma30_pos_target[0:1]    # (1, 30, 3)

    frames = list(range(N_cond_tail)) + [T - 1]
    frame_idx = torch.tensor(frames, dtype=torch.long)
    constraint_rots = torch.cat([tail_rots, target_first_rots], dim=0)
    constraint_pos = torch.cat([tail_pos, target_first_pos], dim=0)

    constraint = FullBodyConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=constraint_pos,
        global_joints_rots=constraint_rots,
        to_crop=False,
    )
    return [constraint]


def build_constraints_e16(skeleton, soma30_rots_target, soma30_pos_target,
                           soma30_rots, soma30_pos,
                           T, setting, N_cond_head, N_transition, caption=""):
    """E16 Transition from target last frame: constrain target last + motion head.

    Sequence: [target_last(1) | transition(N_transition) | motion_head(N_cond_head)]
    """
    from kimodo.constraints import FullBodyConstraintSet
    import torch

    # Target last + motion head SOMA30 data
    target_last_rots = soma30_rots_target[-1:]   # (1, 30, 3, 3)
    target_last_pos = soma30_pos_target[-1:]     # (1, 30, 3)
    head_rots = soma30_rots[:N_cond_head]        # (N_cond_head, 30, 3, 3)
    head_pos = soma30_pos[:N_cond_head]          # (N_cond_head, 30, 3)

    frames = [0] + list(range(1 + N_transition, T))
    frame_idx = torch.tensor(frames, dtype=torch.long)
    constraint_rots = torch.cat([target_last_rots, head_rots], dim=0)
    constraint_pos = torch.cat([target_last_pos, head_pos], dim=0)

    constraint = FullBodyConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=constraint_pos,
        global_joints_rots=constraint_rots,
        to_crop=False,
    )
    return [constraint]


CONSTRAINT_BUILDERS = {
    'E2': build_constraints_e2,
    'E3': build_constraints_e3,
    'E4': build_constraints_e4,
    'E5': build_constraints_e5,
    # E6 has special signature (needs gt_pos_22 for foot contact detection)
    'E7': build_constraints_e7,
    'E8': build_constraints_e8,
    'E10': build_constraints_e10,
    # E14/E15/E16 use special handling in the evaluation loop (different signatures)
}


# ============================================================================
# Main evaluation loop
# ============================================================================

def evaluate_sample(model, skeleton, soma30_rots, soma30_pos, gt_pos_22,
                    caption, T, task_id, setting, fps=30):
    """Run KIMODO on one sample and return predicted SMPL-22 positions.

    Args:
        model: KIMODO model.
        skeleton: KIMODO skeleton.
        soma30_rots: (T, 30, 3, 3) retargeted SOMA30 global rotations.
        soma30_pos: (T, 30, 3) retargeted SOMA30 joint positions.
        gt_pos_22: (T, 22, 3) SMPL-22 GT positions (for metrics + E6 contact).
        caption: text prompt.
        T: number of frames.
        task_id: e.g. 'E2', 'E6'.
        setting: e.g. 'A', 'B'.
        fps: frame rate.
    """
    import torch

    # ------------------------------------------------------------------
    # Canonicalize the retargeted SOMA30 motion into KIMODO's training
    # frame (frame-0 XZ at origin, heading 0). This moves the constraint
    # data from raw world coordinates into the distribution KIMODO was
    # trained on, so constraint tasks generalize the same way T2M does.
    # See kimodo_compute_canon_transform / kimodo_apply_canon above.
    # ------------------------------------------------------------------
    R_yaw, t_xz, _heading0 = kimodo_compute_canon_transform(soma30_pos, skeleton)
    soma30_rots_c, soma30_pos_c = kimodo_apply_canon(
        soma30_rots, soma30_pos, R_yaw, t_xz)

    builder = CONSTRAINT_BUILDERS.get(task_id)
    if builder is None and task_id != 'E6':
        return None, {}, {}

    if task_id == 'E6':
        # E6 needs gt_pos_22 for foot contact detection; the detection
        # itself is per-frame local (ankle Y), so canonicalization doesn't
        # shift which frames are contact frames — but the constraint
        # positions fed to KIMODO must be in canonical world.
        constraints = build_constraints_e6(
            skeleton, soma30_rots_c, soma30_pos_c, gt_pos_22, T, setting, caption)
    else:
        constraints = builder(skeleton, soma30_rots_c, soma30_pos_c, T, setting, caption)

    # KIMODO works at its own fps (typically 30)
    model_fps = model.fps
    duration_sec = T / fps
    num_frames = int(duration_sec * model_fps)
    if num_frames < 10:
        num_frames = 10

    # Move constraint tensors to device and clamp frame indices
    device = next(model.denoiser.parameters()).device
    for c in constraints:
        for attr in dir(c):
            if attr.startswith('_'):
                continue
            val = getattr(c, attr, None)
            if isinstance(val, torch.Tensor):
                setattr(c, attr, val.to(device))
        if hasattr(c, 'frame_indices') and isinstance(c.frame_indices, torch.Tensor):
            c.frame_indices = c.frame_indices.clamp(max=num_frames - 1)

    t0 = time.time()
    try:
        output = model(
            [caption] if caption else [""],
            [num_frames],
            num_denoising_steps=DIFFUSION_STEPS,
            constraint_lst=[constraints],
            cfg_weight=[2.0, 2.0],
            num_samples=1,
            return_numpy=True,
            multi_prompt=False,
            post_processing=False,
        )
    except Exception as e:
        print(f"    KIMODO inference error: {e}")
        return None, {"inference_time": round(time.time() - t0, 2)}, {}

    elapsed = time.time() - t0

    # Extract SMPL-22 positions from SOMA-77
    posed_joints = output["posed_joints"]
    if posed_joints.ndim == 4:
        posed_joints = posed_joints[0]  # Remove batch dim
    pred_pos_22 = soma77_to_smpl22(posed_joints)

    # Decanonicalize output (T, J, 3) back to world coords so metrics and
    # NPZ are comparable to the raw SMPL GT. Apply the INVERSE of the
    # (R_yaw, t_xz) transform captured from the input motion above.
    pred_pos_22 = kimodo_invert_canon_positions(
        pred_pos_22, R_yaw, t_xz).numpy()
    # Also decanonicalize the full SOMA-77 for mesh rendering.
    posed_joints = kimodo_invert_canon_positions(
        posed_joints, R_yaw, t_xz).numpy()

    # Keep SOMA-77 data for mesh rendering
    posed_joints_77 = posed_joints  # (T, 77, 3)
    global_rot_mats_77 = None
    if "global_rot_mats" in output:
        global_rot_mats_77 = output["global_rot_mats"]
        if global_rot_mats_77.ndim == 5:
            global_rot_mats_77 = global_rot_mats_77[0]  # Remove batch dim
        # global_rot_mats are GLOBAL rotation matrices → world-frame
        # rotations. Left-multiply by R_yaw^T to decanonicalize.
        import torch as _torch
        if isinstance(global_rot_mats_77, np.ndarray):
            _g = _torch.from_numpy(global_rot_mats_77).float()
        else:
            _g = global_rot_mats_77.float().cpu()
        _R_inv = R_yaw.transpose(-1, -2).cpu()
        _g_world = _torch.einsum('ij,tnjk->tnik', _R_inv, _g)
        global_rot_mats_77 = _g_world.numpy()

    # Resample to target fps if needed
    if model_fps != fps:
        from scipy.interpolate import interp1d
        old_times = np.linspace(0, 1, pred_pos_22.shape[0])
        new_times = np.linspace(0, 1, T)
        interp = interp1d(old_times, pred_pos_22, axis=0, kind='linear')
        pred_pos_22 = interp(new_times)

    # Crop/pad to T
    if pred_pos_22.shape[0] > T:
        pred_pos_22 = pred_pos_22[:T]
    elif pred_pos_22.shape[0] < T:
        pad = np.tile(pred_pos_22[-1:], (T - pred_pos_22.shape[0], 1, 1))
        pred_pos_22 = np.concatenate([pred_pos_22, pad], axis=0)

    # ── 2026-04-26: Y-anchor against GT to suppress KIMODO floor drift ──
    # KIMODO's diffusion output exhibits ~10-30 cm of upward Y drift over
    # long unconstrained spans (model artifact, NOT a code bug — see
    # comments in evaluate_sample). For a fair visual comparison we
    # subtract a single global Y offset that aligns frame-0 ground level
    # to the GT ground level. This is metric-neutral when the constraint
    # already pins frame 0 (start_1f / both_1f / pre20 / mid60 / E3 keyframes
    # / E5 traj …), and only does a rigid Y shift otherwise. The same
    # offset is applied to posed_joints / global_rot_mats so the SOMA mesh
    # stays in lockstep with the SMPL-22 skeleton.
    y_anchor_delta = 0.0
    try:
        if gt_pos_22 is not None and pred_pos_22.shape[0] >= 1:
            # Use the first 5 frames of frame-0 region for stability.
            n0 = min(5, pred_pos_22.shape[0], gt_pos_22.shape[0])
            pred_floor0 = float(pred_pos_22[:n0, :, 1].min())
            gt_floor0 = float(gt_pos_22[:n0, :, 1].min())
            y_anchor_delta = pred_floor0 - gt_floor0
    except Exception:
        y_anchor_delta = 0.0
    if abs(y_anchor_delta) > 1e-4:
        pred_pos_22 = pred_pos_22.copy()
        pred_pos_22[..., 1] -= y_anchor_delta
        if posed_joints_77 is not None:
            posed_joints_77 = posed_joints_77.copy()
            posed_joints_77[..., 1] -= y_anchor_delta

    # Compute position-space metrics
    metrics = {"inference_time": round(elapsed, 2),
               "y_anchor_delta": round(y_anchor_delta, 4)}

    # MPJPE between predicted and GT (only for generated frames)
    if gt_pos_22 is not None:
        diff = np.sqrt(np.sum((pred_pos_22 - gt_pos_22[:T]) ** 2, axis=-1))
        metrics["mpjpe_pos"] = float(diff.mean())

    # Jitter (acceleration-based)
    if pred_pos_22.shape[0] > 2:
        vel = np.diff(pred_pos_22, axis=0) * fps
        acc = np.diff(vel, axis=0) * fps
        jitter = np.linalg.norm(acc, axis=-1).mean()
        metrics["jitter_pos"] = float(jitter)

    # Foot skating
    l_ankle = pred_pos_22[:, 7]
    r_ankle = pred_pos_22[:, 8]
    for ankle, name in [(l_ankle, 'l'), (r_ankle, 'r')]:
        contact = ankle[:, 1] < 0.05
        if contact.any():
            vel_ankle = np.diff(ankle[contact], axis=0)
            skating = np.linalg.norm(vel_ankle[:, [0, 2]], axis=-1)
            metrics[f"foot_skating_{name}"] = float(skating.mean())

    # Attach SOMA-77 data for mesh rendering (if available)
    soma_data = {}
    if posed_joints_77 is not None:
        soma_data['posed_joints'] = posed_joints_77.astype(np.float32) if hasattr(posed_joints_77, 'astype') else posed_joints_77
    if global_rot_mats_77 is not None:
        soma_data['global_rot_mats'] = global_rot_mats_77.astype(np.float32) if hasattr(global_rot_mats_77, 'astype') else global_rot_mats_77

    return pred_pos_22, metrics, soma_data


def _run_kimodo_with_constraints(model, skeleton, constraints, caption, T,
                                  gt_pos_22, fps=30,
                                  canon_transform=None):
    """Run KIMODO with pre-built constraints. Used by E14/E15/E16.

    Args:
        canon_transform: optional (R_yaw, t_xz) from kimodo_compute_canon_transform
            — if provided, the caller has ALREADY canonicalized the SOMA30
            data before building constraints; this function will invert
            the transform on the predicted positions so they come back in
            world coords. If None, no canon is applied (legacy behavior).

    Returns (pred_pos_22, metrics, soma_data) same as evaluate_sample.
    """
    import torch

    model_fps = model.fps
    duration_sec = T / fps
    num_frames = int(duration_sec * model_fps)
    if num_frames < 10:
        num_frames = 10

    device = next(model.denoiser.parameters()).device
    for c in constraints:
        for attr in dir(c):
            if attr.startswith('_'):
                continue
            val = getattr(c, attr, None)
            if isinstance(val, torch.Tensor):
                setattr(c, attr, val.to(device))
        if hasattr(c, 'frame_indices') and isinstance(c.frame_indices, torch.Tensor):
            c.frame_indices = c.frame_indices.clamp(max=num_frames - 1)

    t0 = time.time()
    try:
        output = model(
            [caption] if caption else [""],
            [num_frames],
            num_denoising_steps=DIFFUSION_STEPS,
            constraint_lst=[constraints],
            cfg_weight=[2.0, 2.0],
            num_samples=1,
            return_numpy=True,
            multi_prompt=False,
            post_processing=False,
        )
    except Exception as e:
        print(f"    KIMODO inference error: {e}")
        return None, {"inference_time": round(time.time() - t0, 2)}, {}

    elapsed = time.time() - t0

    posed_joints = output["posed_joints"]
    if posed_joints.ndim == 4:
        posed_joints = posed_joints[0]
    pred_pos_22 = soma77_to_smpl22(posed_joints)

    # Decanonicalize output back to world coords if caller canonicalized input.
    if canon_transform is not None:
        R_yaw, t_xz = canon_transform
        pred_pos_22 = kimodo_invert_canon_positions(
            pred_pos_22, R_yaw, t_xz).numpy()
        posed_joints = kimodo_invert_canon_positions(
            posed_joints, R_yaw, t_xz).numpy()

    # Keep SOMA-77 data for mesh rendering
    posed_joints_77 = posed_joints  # (T, 77, 3)
    global_rot_mats_77 = None
    if "global_rot_mats" in output:
        global_rot_mats_77 = output["global_rot_mats"]
        if global_rot_mats_77.ndim == 5:
            global_rot_mats_77 = global_rot_mats_77[0]  # Remove batch dim
        if canon_transform is not None:
            R_yaw, _ = canon_transform
            import torch as _torch
            if isinstance(global_rot_mats_77, np.ndarray):
                _g = _torch.from_numpy(global_rot_mats_77).float()
            else:
                _g = global_rot_mats_77.float().cpu()
            _R_inv = R_yaw.transpose(-1, -2).cpu()
            _g_world = _torch.einsum('ij,tnjk->tnik', _R_inv, _g)
            global_rot_mats_77 = _g_world.numpy()

    # Resample to target fps
    if model_fps != fps:
        from scipy.interpolate import interp1d
        old_times = np.linspace(0, 1, pred_pos_22.shape[0])
        new_times = np.linspace(0, 1, T)
        interp = interp1d(old_times, pred_pos_22, axis=0, kind='linear')
        pred_pos_22 = interp(new_times)

    if pred_pos_22.shape[0] > T:
        pred_pos_22 = pred_pos_22[:T]
    elif pred_pos_22.shape[0] < T:
        pad = np.tile(pred_pos_22[-1:], (T - pred_pos_22.shape[0], 1, 1))
        pred_pos_22 = np.concatenate([pred_pos_22, pad], axis=0)

    metrics = {"inference_time": round(elapsed, 2)}

    # Jitter
    if pred_pos_22.shape[0] > 2:
        vel = np.diff(pred_pos_22, axis=0) * fps
        acc = np.diff(vel, axis=0) * fps
        jitter = np.linalg.norm(acc, axis=-1).mean()
        metrics["jitter_pos"] = float(jitter)

    # Foot skating
    l_ankle = pred_pos_22[:, 7]
    r_ankle = pred_pos_22[:, 8]
    for ankle, name in [(l_ankle, 'l'), (r_ankle, 'r')]:
        contact = ankle[:, 1] < 0.05
        if contact.any():
            vel_ankle = np.diff(ankle[contact], axis=0)
            skating = np.linalg.norm(vel_ankle[:, [0, 2]], axis=-1)
            metrics[f"foot_skating_{name}"] = float(skating.mean())

    # Attach SOMA-77 data for mesh rendering (if available)
    soma_data = {}
    if posed_joints_77 is not None:
        soma_data['posed_joints'] = posed_joints_77.astype(np.float32) if hasattr(posed_joints_77, 'astype') else posed_joints_77
    if global_rot_mats_77 is not None:
        soma_data['global_rot_mats'] = global_rot_mats_77.astype(np.float32) if hasattr(global_rot_mats_77, 'astype') else global_rot_mats_77

    return pred_pos_22, metrics, soma_data


def main():
    parser = argparse.ArgumentParser(description='KIMODO all-tasks evaluation')
    parser.add_argument('--tasks', nargs='+',
                        help='Task IDs (E2-E8, E10, E14-E16)')
    parser.add_argument('--all-tasks', action='store_true')
    parser.add_argument('--settings', nargs='+')
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--output-dir', type=str,
                        default='work_dirs/m2m_v2_eval_latest/kimodo')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--motion-data-dir', type=str, default=MOTION_DATA_DIR)
    parser.add_argument('--use-caption', choices=['yes', 'no'], default='yes',
                        help='Whether to feed the sample caption to KIMODO. '
                             '"yes" = caption-conditioned (default); "no" = '
                             'unconditional (empty-string prompt). Run both '
                             'in separate invocations to produce the two '
                             'KIMODO variants the website expects.')
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch

    # Determine tasks
    all_tasks = ['E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E10', 'E14', 'E15', 'E16']
    if args.all_tasks:
        task_ids = all_tasks
    elif args.tasks:
        task_ids = args.tasks
    else:
        task_ids = ['E2', 'E3', 'E5']

    os.makedirs(args.output_dir, exist_ok=True)

    # Load KIMODO
    print(f"Loading KIMODO model: {KIMODO_MODEL}")
    from kimodo import load_model
    model = load_model(KIMODO_MODEL, device=args.device)
    skeleton = model.skeleton
    print(f"KIMODO loaded. fps={model.fps}")

    # Load bone offsets for FK (SMPL-22)
    bone_offsets_path = 'data/hymotion_m2m_data/bone_offsets_22.pt'
    bone_offsets = torch.load(bone_offsets_path, map_location='cpu').numpy()

    # Load eval tasks
    from hftrainer.evaluation.motion.m2m_eval_tasks import EVAL_TASKS, get_task
    from hftrainer.evaluation.motion.m2m_eval_tasks import compute_transition_length
    from tools.eval_m2m_v2_all_tasks import load_eval_samples, load_motion_135d
    from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np

    all_results = {}

    for task_id in task_ids:
        task = get_task(task_id)
        if task is None:
            print(f"Unknown task: {task_id}")
            continue

        # Skip tasks that are not meaningful for KIMODO (e.g. E15 "prepend
        # to start pose" has no analogous KIMODO constraint set under the
        # new 2026-04-21 redefinition — the runner only implements the
        # legacy _use_target_first semantics, which no current setting uses).
        if not getattr(task, 'kimodo_comparable', True):
            print(f"Skipping {task_id} ({task.name}) — kimodo_comparable=False")
            continue

        settings = list(task.settings.keys()) if not args.settings else args.settings

        for setting_name in settings:
            task_key = f"{task_id}_{setting_name}"
            # 2026-04-26: per-setting use_caption=True means caption is
            # REQUIRED for this setting; running without caption produces
            # spurious "uncond" rows that mislead comparisons. Skip the
            # entire setting when --use-caption=no AND the setting demands
            # caption (e.g. E2 start_1f / end_1f / both_1f).
            _setting_uc_outer = getattr(
                task.settings[setting_name], 'use_caption', None)
            if (args.use_caption == 'no' and
                    _setting_uc_outer is True):
                print(f"\n  Task: {task_key} — SKIPPED (setting requires "
                      f"caption; uncond run would be invalid)")
                continue
            print(f"\n{'='*60}")
            print(f"Task: {task_key} — {task.name}")
            print(f"{'='*60}")

            # Load eval samples — per-setting `_data_file` (E14 v5)
            # overrides the task-level default if present.
            _per_setting_data_file = task.settings[setting_name].mask_kwargs.get(
                '_data_file', None)
            _data_file_name = _per_setting_data_file or task.data_file
            data_file = str(PROJECT_ROOT / "data" / "eval" / "m2m_v2" / _data_file_name)
            if not os.path.exists(data_file):
                print(f"  Data file not found: {data_file}")
                continue
            print(f"  Using datalist: {_data_file_name}")

            samples = load_eval_samples(
                data_file, args.motion_data_dir, args.max_samples,
                require_caption=False, bone_offsets=bone_offsets,
            )
            print(f"  Loaded {len(samples)} samples")

            npz_dir = os.path.join(args.output_dir, task_key, 'npz')
            os.makedirs(npz_dir, exist_ok=True)

            per_sample = []
            for i, sample in enumerate(samples):
                motion = sample['motion']   # (T, 135) denormalized
                T = sample['T']
                caption = sample.get('caption', '')
                # Per-setting use_caption (2026-04-25, E2 v2 _uncond twins)
                # takes precedence over the CLI --use-caption flag. When
                # the setting explicitly says use_caption=False, force the
                # prompt to empty regardless of how this run was launched.
                # When None (inherit), fall back to the CLI flag below.
                _setting_uc = getattr(
                    task.settings[setting_name], 'use_caption', None)
                if _setting_uc is False:
                    caption = ''
                elif _setting_uc is None and args.use_caption == 'no':
                    caption = ''
                # If _setting_uc is True we keep the loaded caption even
                # when --use-caption=no, because the setting is caption-
                # required by definition.
                fps_val = sample.get('fps', 30)

                # Get GT positions via SMPL-22 FK (for metrics)
                gt_pos = motion135_to_positions_np(motion, bone_offsets)

                # Rotation-based retarget: SMPL-22 -> SOMA-30
                soma30_rots, soma30_pos = smpl22_to_soma30_retarget(
                    motion, bone_offsets)

                try:
                    setting_kwargs = task.settings[setting_name].mask_kwargs

                    # ---- E8 loop completion (B/C/D): adjust T for KIMODO ----
                    if task_id == 'E8' and '_loop_append' in setting_kwargs:
                        N_append = setting_kwargs['_loop_append']
                        T_total = T + N_append
                        pred_pos, metrics, soma_data = evaluate_sample(
                            model, skeleton, soma30_rots, soma30_pos, gt_pos,
                            caption, T_total, task_id, setting_name, fps_val,
                        )

                    # ---- E14: transition stitching ----
                    elif task_id == 'E14' and '_use_transition_data' in setting_kwargs:
                        # 2026-04-26 (v5 alignment): mirror the M2M pipeline's
                        # E14 v5 logic so KIMODO sees the same (motion_a,
                        # motion_b, N_transition, N_cond_a, N_cond_b)
                        # geometry as HyMotion. Previously this branch used
                        # the legacy `place_b_after_a(forward=1m)` placement
                        # and a fixed N_cond=15, which doesn't exist in the
                        # current L/M settings — so KIMODO was generating
                        # for a completely different problem than HyMotion.
                        #
                        # Strategy: read placement from setting_kwargs, run
                        # _place_b_custom + compute_transition_length with
                        # the same speed/joint-angle thresholds as backend,
                        # then compute N_cond_a / N_cond_b via
                        # compute_cond_length(adaptive). KIMODO's constraint
                        # builder then locks the head/tail of A and B as
                        # condition frames and lets the model fill the
                        # transition window.

                        # Load motion A and B (datalist now stores absolute
                        # paths under data/hymotion_data/...).
                        motion_a_path = sample.get('motion_a_path', '')
                        motion_b_path = sample.get('motion_b_path', '')
                        if motion_a_path and not os.path.isabs(motion_a_path) and not os.path.exists(motion_a_path):
                            motion_a_path = os.path.join(args.motion_data_dir, motion_a_path)
                        if motion_b_path and not os.path.isabs(motion_b_path) and not os.path.exists(motion_b_path):
                            motion_b_path = os.path.join(args.motion_data_dir, motion_b_path)
                        motion_a = load_motion_135d(motion_a_path)
                        motion_b = load_motion_135d(motion_b_path)
                        if motion_a is None or motion_b is None:
                            per_sample.append({"_sample_idx": i, "_error": "load failed"})
                            continue

                        import torch as _torch
                        from tools.eval_m2m_v2_all_tasks import _place_b_custom
                        from hftrainer.evaluation.motion.m2m_eval_tasks import (
                            compute_cond_length,
                        )

                        placement = setting_kwargs.get('_placement', 'forward')
                        context_policy = setting_kwargs.get(
                            '_context_policy', None)
                        forward_step = float(setting_kwargs.get('_forward_step', 1.0))
                        yaw_offset_deg = float(setting_kwargs.get('_yaw_offset_deg', 0.0))

                        # Step 1: estimate N_transition from an overlap
                        # placement (same as backend: avoids the velocity-
                        # placement circularity).
                        motion_b_world_overlap = _place_b_custom(
                            motion_a, motion_b, placement='overlap',
                            N_transition=1, yaw_offset_deg=yaw_offset_deg)
                        pos_a_full = motion135_to_positions_np(
                            motion_a, bone_offsets)
                        pos_b_overlap_full = motion135_to_positions_np(
                            motion_b_world_overlap, bone_offsets)
                        N_transition = compute_transition_length(
                            pos_a_full[-1, 0],
                            pos_b_overlap_full[0, 0],
                            speed_per_frame=float(setting_kwargs.get(
                                '_transition_speed', 0.015)),
                            min_frames=int(setting_kwargs.get(
                                '_transition_min', 30)),
                            max_frames=int(setting_kwargs.get(
                                '_transition_max', 120)),
                            joints_a_end=pos_a_full[-1],
                            joints_b_start=pos_b_overlap_full[0],
                            pose_speed_per_frame=float(setting_kwargs.get(
                                '_pose_speed', 0.015)),
                            motion_a_end_135=motion_a[-1],
                            motion_b_start_135=motion_b_world_overlap[0],
                            joint_angle_speed_per_frame=float(setting_kwargs.get(
                                '_joint_angle_speed', 0.20)),
                        )

                        # Step 2: place B with the chosen strategy.
                        motion_b_world = _place_b_custom(
                            motion_a, motion_b,
                            placement=placement,
                            N_transition=N_transition,
                            forward_step=forward_step,
                            yaw_offset_deg=yaw_offset_deg)

                        # Step 3: pick N_cond per side. v5 uses adaptive
                        # rule (3-10 frames per side based on quality &
                        # available horizon).
                        len_a = int(motion_a.shape[0])
                        len_b = int(motion_b_world.shape[0])
                        if context_policy == 'adaptive':
                            N_cond_a = compute_cond_length(
                                motion_a, T_src=len_a,
                                N_transition=N_transition, side='tail')
                            N_cond_b = compute_cond_length(
                                motion_b_world, T_src=len_b,
                                N_transition=N_transition, side='head')
                        elif context_policy == 'fixed':
                            N_cond_a = min(int(setting_kwargs.get(
                                '_n_cond_a_frames', 5)), len_a)
                            N_cond_b = min(int(setting_kwargs.get(
                                '_n_cond_b_frames', 5)), len_b)
                        else:
                            N_cond_a = min(int(setting_kwargs.get(
                                '_cond_frames', 5)), len_a)
                            N_cond_b = N_cond_a
                        N_cond = max(N_cond_a, N_cond_b)
                        T_total = N_cond_a + N_transition + N_cond_b
                        print(f"    [E14 KIMODO] placement={placement} "
                              f"N_cond_a={N_cond_a} N_transition={N_transition} "
                              f"N_cond_b={N_cond_b}")

                        # Retarget both motions A and B (B in world coords now)
                        soma30_rots_a, soma30_pos_a = smpl22_to_soma30_retarget(
                            motion_a, bone_offsets)
                        soma30_rots_b, soma30_pos_b = smpl22_to_soma30_retarget(
                            motion_b_world, bone_offsets)

                        # Canonicalize around A's frame 0 so the whole
                        # (A, B) segment sits in KIMODO's training frame.
                        # Both A and B must share the same transform to
                        # preserve their relative world geometry.
                        R_yaw, t_xz, _ = kimodo_compute_canon_transform(
                            soma30_pos_a, skeleton)
                        soma30_rots_a, soma30_pos_a = kimodo_apply_canon(
                            soma30_rots_a, soma30_pos_a, R_yaw, t_xz)
                        soma30_rots_b, soma30_pos_b = kimodo_apply_canon(
                            soma30_rots_b, soma30_pos_b, R_yaw, t_xz)

                        constraints = build_constraints_e14(
                            skeleton, soma30_rots_a, soma30_pos_a,
                            soma30_rots_b, soma30_pos_b,
                            T_total, setting_name,
                            N_cond, N_transition, caption,
                            N_cond_a=N_cond_a, N_cond_b=N_cond_b)

                        pred_pos, metrics, soma_data = _run_kimodo_with_constraints(
                            model, skeleton, constraints, caption, T_total,
                            gt_pos, fps_val,
                            canon_transform=(R_yaw, t_xz))
                        metrics['transition_length'] = N_transition
                        metrics['n_cond_a'] = N_cond_a
                        metrics['n_cond_b'] = N_cond_b

                    # ---- E15: transition to target first frame ----
                    elif task_id == 'E15' and '_use_target_first' in setting_kwargs:
                        N_cond_tail = setting_kwargs.get('_cond_tail_frames', 15)

                        target_path = sample.get('target_motion_path', '')
                        if not os.path.isabs(target_path):
                            target_path = os.path.join(args.motion_data_dir, target_path)
                        target_motion = load_motion_135d(target_path)
                        if target_motion is None:
                            per_sample.append({"_sample_idx": i, "_error": "load failed"})
                            continue

                        target_pos = motion135_to_positions_np(target_motion, bone_offsets)
                        N_transition = compute_transition_length(
                            gt_pos[-1, 0], target_pos[0, 0])
                        T_total = N_cond_tail + N_transition + 1

                        # Retarget target motion
                        soma30_rots_target, soma30_pos_target = smpl22_to_soma30_retarget(
                            target_motion, bone_offsets)

                        constraints = build_constraints_e15(
                            skeleton, soma30_rots, soma30_pos,
                            soma30_rots_target, soma30_pos_target,
                            T_total, setting_name,
                            N_cond_tail, N_transition, caption)

                        pred_pos, metrics, soma_data = _run_kimodo_with_constraints(
                            model, skeleton, constraints, caption, T_total,
                            gt_pos, fps_val)
                        metrics['transition_length'] = N_transition

                    # ---- E16: transition from target last frame ----
                    elif task_id == 'E16' and '_use_target_last' in setting_kwargs:
                        N_cond_head = setting_kwargs.get('_cond_head_frames', 15)

                        target_path = sample.get('target_motion_path', '')
                        if not os.path.isabs(target_path):
                            target_path = os.path.join(args.motion_data_dir, target_path)
                        target_motion = load_motion_135d(target_path)
                        if target_motion is None:
                            per_sample.append({"_sample_idx": i, "_error": "load failed"})
                            continue

                        target_pos = motion135_to_positions_np(target_motion, bone_offsets)
                        N_transition = compute_transition_length(
                            target_pos[-1, 0], gt_pos[0, 0])
                        T_total = 1 + N_transition + N_cond_head

                        # Retarget target motion
                        soma30_rots_target, soma30_pos_target = smpl22_to_soma30_retarget(
                            target_motion, bone_offsets)

                        constraints = build_constraints_e16(
                            skeleton, soma30_rots_target, soma30_pos_target,
                            soma30_rots, soma30_pos,
                            T_total, setting_name,
                            N_cond_head, N_transition, caption)

                        pred_pos, metrics, soma_data = _run_kimodo_with_constraints(
                            model, skeleton, constraints, caption, T_total,
                            gt_pos, fps_val)
                        metrics['transition_length'] = N_transition

                    # ---- Standard tasks ----
                    else:
                        pred_pos, metrics, soma_data = evaluate_sample(
                            model, skeleton, soma30_rots, soma30_pos, gt_pos,
                            caption, T, task_id, setting_name, fps_val,
                        )

                    # E4 EE-specific metric parity with M2M eval pipeline.
                    # Compute ee_error_mean/p50/p95/max/std and hit_rate_{2,5,10}cm
                    # at the constraint (frame, joint) pairs so dashboards can
                    # compare KIMODO vs HyMotion M2M v2 head-to-head on the
                    # same metric set.
                    if (task_id == 'E4' and pred_pos is not None
                            and gt_pos is not None):
                        # SMPL-22 indices for each setting (mirror joint_map
                        # in build_constraints_e4).
                        _e4_smpl22 = {
                            'A_rhand_sparse':  ([21], 10),
                            'B_ankles_sparse': ([7, 8], 15),
                            'C_rhand_lfoot':   ([21, 10], 15),
                            'D_both_hands':    ([20, 21], 10),
                            'E_all4_sparse':   ([20, 21, 7, 8], 20),
                            'F_rhand_dense':   ([21], 5),
                        }
                        if setting_name in _e4_smpl22:
                            jlist, interval = _e4_smpl22[setting_name]
                            T_eff = min(pred_pos.shape[0], gt_pos.shape[0])
                            frames = list(range(0, T_eff, interval))
                            if frames:
                                pframes = np.asarray(frames, dtype=np.int64)
                                errs = []
                                for j in jlist:
                                    diff = (pred_pos[pframes, j]
                                            - gt_pos[pframes, j])
                                    errs.append(np.linalg.norm(diff, axis=-1))
                                errs = np.concatenate(errs, axis=0).astype(
                                    np.float32)
                                metrics['ee_error_mean'] = float(errs.mean())
                                metrics['ee_error_max'] = float(errs.max())
                                metrics['ee_error_p50'] = float(
                                    np.percentile(errs, 50))
                                metrics['ee_error_p95'] = float(
                                    np.percentile(errs, 95))
                                metrics['ee_error_std'] = float(errs.std())
                                metrics['ee_hit_rate_2cm'] = float(
                                    (errs < 0.02).mean())
                                metrics['ee_hit_rate_5cm'] = float(
                                    (errs < 0.05).mean())
                                metrics['ee_hit_rate_10cm'] = float(
                                    (errs < 0.10).mean())
                except Exception as e:
                    print(f"    [{i+1}] SAMPLE ERROR: {e}")
                    pred_pos, metrics, soma_data = None, {"_error": str(e)[:100]}, {}
                    # Reset CUDA state after error
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                if pred_pos is not None:
                    npz_path = os.path.join(npz_dir, f"{i:05d}.npz")
                    save_fields = dict(
                        positions=pred_pos,
                        translation=pred_pos[:, 0],
                    )
                    # Include SOMA-77 data for mesh rendering
                    if soma_data.get('posed_joints') is not None:
                        save_fields['posed_joints'] = soma_data['posed_joints']
                    if soma_data.get('global_rot_mats') is not None:
                        save_fields['global_rot_mats'] = soma_data['global_rot_mats']
                    np.savez_compressed(npz_path, **save_fields)
                    metrics['_npz_path'] = npz_path

                metrics['_sample_idx'] = i
                metrics['_caption'] = caption
                metrics['_num_frames'] = T
                per_sample.append(metrics)

                if (i + 1) % 10 == 0:
                    print(f"    [{i+1}/{len(samples)}] done")

            # Aggregate
            agg = {}
            metric_names = set()
            for s in per_sample:
                metric_names.update(k for k in s if not k.startswith('_'))
            for m in sorted(metric_names):
                vals = [s[m] for s in per_sample if m in s and isinstance(s[m], (int, float))]
                if vals:
                    agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

            # Save result.json
            # Per-setting use_caption (2026-04-25, E2 v2 _uncond twins)
            # determines the EFFECTIVE caption mode for tagging; this
            # ensures e.g. running --use-caption=yes on pre20_uncond still
            # reports model="KIMODO_uncond" since caption was force-blanked.
            _setting_uc = getattr(
                task.settings[setting_name], 'use_caption', None)
            if _setting_uc is True:
                _eff_caption = True
            elif _setting_uc is False:
                _eff_caption = False
            else:
                _eff_caption = (args.use_caption == 'yes')
            result = {
                "model": "KIMODO_caption" if _eff_caption else "KIMODO_uncond",
                "task_id": task_id,
                "setting": setting_name,
                "retarget_method": "rotation_based",
                "has_caption": _eff_caption,
                "num_prompts": len(per_sample),
                "aggregated": agg,
                "per_sample": per_sample,
            }
            result_dir = os.path.join(args.output_dir, task_key)
            result_path = os.path.join(result_dir, "result.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {result_path}")

            # Print summary
            for m in ['mpjpe_pos', 'jitter_pos', 'inference_time']:
                if m in agg:
                    print(f"    {m}: {agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}")

            all_results[task_key] = result

    print(f"\n{'='*60}")
    print(f"All done. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
