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
KIMODO_MOTION_CORRECTION_ROOT = KIMODO_ROOT / "MotionCorrection" / "python"
sys.path.insert(0, str(KIMODO_ROOT))
if KIMODO_MOTION_CORRECTION_ROOT.exists():
    sys.path.insert(0, str(KIMODO_MOTION_CORRECTION_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

KIMODO_MODEL = "kimodo-soma-rp"
DIFFUSION_STEPS = 100
MOTION_DATA_DIR = str(PROJECT_ROOT / "data" / "hymotion_data")


def _kimodo_cfg_weight():
    text_w = float(os.environ.get("KIMODO_TEXT_CFG", "2.0"))
    cond_w = float(os.environ.get("KIMODO_CONSTRAINT_CFG", "2.0"))
    return [text_w, cond_w]


def _kimodo_num_candidates():
    return max(1, int(os.environ.get("KIMODO_NUM_CANDIDATES", "1")))


def _kimodo_use_boundary_anchors():
    return os.environ.get("KIMODO_BOUNDARY_ANCHORS", "0") == "1"


def _kimodo_post_processing():
    # Upstream KIMODO's CLI enables MotionCorrection by default for SOMA
    # models.  FullBodyConstraintSet stores rotations for this correction
    # pass; the denoiser itself only observes positions/root/heading.  Keeping
    # this default aligned with upstream avoids visible condition/generation
    # rotation seams in E14/E15.  Set KIMODO_POST_PROCESSING=0 for ablations.
    return os.environ.get("KIMODO_POST_PROCESSING", "1") == "1"


def _kimodo_use_first_constraint_heading():
    return os.environ.get("KIMODO_FIRST_CONSTRAINT_HEADING", "0") == "1"

# 2026-04-27 sliding-window inference cap.
# Empirical scan over 720 KIMODO_uncond E3 samples: pj_vmax≥0.5m/frame jump
# rate is 0% at T<300, 59% at T 300-350, 100% at T 350+. Last 25% of frames
# explode — classic train-distribution extrapolation. Cap each segment to
# KIMODO_SAFE_LEN=240 (8s @ 30fps) and let KIMODO's built-in multi-prompt
# stitching glue them with 5-frame transitions (`num_transition_frames=5`,
# `share_transition=True`).  Empirically T<240 had 0% jumps in our scan;
# 240 is conservative.  Total emitted length still equals num_frames (the
# transition frames are blended in-place at segment boundaries, not added
# on top — see KimodoModel._multiprompt at kimodo_model.py:267-311).
KIMODO_SAFE_LEN = 240


def _split_num_frames(n: int, safe_len: int = KIMODO_SAFE_LEN) -> list:
    """Split total `n` frames into K balanced segments each ≤ safe_len.

    Returns a list of segment lengths whose sum equals n.  K=1 when
    n <= safe_len.  Uses balanced split so the worst segment is as short
    as possible: e.g. 360 -> [180,180], 320 -> [160,160], 270 -> [270],
    241 -> [121,120].
    """
    if n <= safe_len:
        return [n]
    K = (n + safe_len - 1) // safe_len  # ceil
    base = n // K
    rem = n - base * K
    return [base + (1 if i < rem else 0) for i in range(K)]


def _make_fullbody_with_rot_constraint_set():
    """Return KIMODO's official FullBody constraint, or a rotation-pinning ablation."""
    from kimodo.constraints import FullBodyConstraintSet, create_pairs
    import torch

    if os.environ.get("KIMODO_PIN_COND_ROT", "0") != "1":
        return FullBodyConstraintSet

    class FullBodyWithRotConstraintSet(FullBodyConstraintSet):
        """Full-body keyframe constraint that also pins global rotations.

        KIMODO's upstream FullBodyConstraintSet stores `global_joints_rots`
        and writes them to saved JSON, but its update_constraints() does not
        append them to the observed-motion mask. Keep this as an explicit
        ablation only: pinning all global rotations made E14/E15 boundary
        poses snap, especially at hands and knees.
        """

        def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
            super().update_constraints(data_dict, index_dict)
            joints = torch.arange(
                self.skeleton.nbjoints,
                device=self.frame_indices.device,
            )
            indices = create_pairs(self.frame_indices, joints)
            data_dict["global_joints_rots"].append(
                self.global_joints_rots.reshape(-1, 3, 3)
            )
            index_dict["global_joints_rots"].append(indices)

        def crop_move(self, start: int, end: int) -> "FullBodyWithRotConstraintSet":
            mask = (self.frame_indices >= start) & (self.frame_indices < end)
            return FullBodyWithRotConstraintSet(
                self.skeleton,
                self.frame_indices[mask] - start,
                self.global_joints_positions[mask],
                self.global_joints_rots[mask],
                self.smooth_root_2d[mask],
            )

    return FullBodyWithRotConstraintSet

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

    # Step 6: Dynamic per-frame foot-floor alignment.
    #
    # A single clip-level Y correction is not enough: once the source pose
    # bends/unbends, SOMA's different leg/foot proportions can make condition
    # frames drift upward again. That is visible as "floating gray context" and
    # also perturbs KIMODO's positional constraints during inference.
    smpl_foot_min_y = smplx_world_pos[:, smplx_foot_indices, 1].min(dim=1).values
    soma_foot_min_y = soma_joints[:, soma_foot_indices, 1].min(dim=1).values
    y_delta = soma_foot_min_y - smpl_foot_min_y
    if torch.max(torch.abs(y_delta)) > 1e-4:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 1] -= y_delta
        soma_global_rots_fk, soma_joints, _ = soma30.fk(soma_local_rots, soma_root_pos)

    # Step 7: Root trajectory lock in XZ only.
    #
    # Keep horizontal trajectory exactly aligned to the source translation, but
    # preserve the Y solved above from foot-floor alignment. Locking Y back to
    # the raw translation reintroduces floating for grounded clips.
    root_delta_xz = translation[:, [0, 2]] - soma_joints[:, soma30.root_idx, :][:, [0, 2]]
    if torch.max(torch.abs(root_delta_xz)) > 1e-6:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 0] += root_delta_xz[:, 0]
        soma_root_pos[:, 2] += root_delta_xz[:, 1]
        soma_global_rots_fk, soma_joints, _ = soma30.fk(soma_local_rots, soma_root_pos)

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


def kimodo_compute_canon_transform(soma30_pos, skeleton, anchor_frame: int = 0):
    """Compute the rigid (yaw, XZ-translation) transform that maps one anchor
    frame of the retargeted SOMA30 motion into KIMODO's canonical frame.

    The canonical frame is defined by:
      - anchor-frame root (pelvis) X, Z = 0
      - anchor-frame heading angle = 0 (R-hip minus L-hip lying along the axis
        where atan2(dz, -dx) == 0, i.e. dx = -1 so the subject faces +Z)

    Args:
        soma30_pos: (T, 30, 3) retargeted SOMA30 world joint positions.
        skeleton: SOMASkeleton30 instance (for hip_joint_idx).
        anchor_frame: frame index used as the canonical anchor.

    Returns:
        (R_yaw, t_xz, heading0) where:
          R_yaw: (3, 3) yaw rotation matrix that maps world -> canonical
                 (left-multiply positions, rotations).
          t_xz : (3,) translation applied AFTER rotation (only XZ non-zero).
          heading0: scalar heading angle of frame 0 (radians), for logging.
    """
    import torch
    anchor_frame = int(anchor_frame)
    r_hip_idx, l_hip_idx = skeleton.hip_joint_idx
    diff = soma30_pos[anchor_frame, r_hip_idx] - soma30_pos[anchor_frame, l_hip_idx]
    heading0 = torch.atan2(diff[2], -diff[0])
    R_yaw = _rot_y(-heading0)  # canonical = world rotated by -heading0
    if os.environ.get("KIMODO_USE_SMOOTH_ROOT_CONSTRAINTS", "0") == "1":
        smooth_root_2d = kimodo_smooth_root_2d_from_positions(soma30_pos)
        root0 = torch.stack([
            smooth_root_2d[anchor_frame, 0],
            soma30_pos[anchor_frame, 0, 1],
            smooth_root_2d[anchor_frame, 1],
        ])
    else:
        # Match upstream KIMODO's plain FullBodyConstraintSet path: unless a
        # separate dense Root2D track is supplied, full-body constraints use
        # the pelvis/root XZ as their root anchor.
        root0 = soma30_pos[anchor_frame, 0]
    root_xz0_rot = R_yaw @ root0
    # After rotation we want (x, y_any, z) = (0, y_any, 0) → translate by
    # -root_xz0_rot in X, Z only (Y preserved).
    t_xz = torch.tensor([-root_xz0_rot[0], 0.0, -root_xz0_rot[2]],
                        dtype=soma30_pos.dtype, device=soma30_pos.device)
    return R_yaw, t_xz, heading0


def kimodo_smooth_root_2d_from_positions(soma_pos):
    """Return KIMODO smooth-root XZ for a SOMA position sequence.

    KIMODO FullBody constraints contain two coupled pieces of positional
    information:
      * ``smooth_root_2d`` for the smooth root planar trajectory; and
      * ``global_joints_positions`` stored relative to that smooth root inside
        the motion representation.

    If ``smooth_root_2d`` is omitted, the official constraint class falls back
    to raw pelvis/root XZ.  The eval path follows that official behavior by
    default; set ``KIMODO_USE_SMOOTH_ROOT_CONSTRAINTS=1`` only for ablations.
    """
    import importlib.util
    import torch

    try:
        from kimodo.motion_rep.smooth_root import get_smooth_root_pos
    except Exception:
        # Some visualization append scripts bootstrap only kimodo.skeleton.*
        # on Python 3.9 and intentionally avoid importing the full kimodo
        # package.  Load smooth_root.py directly with the tiny part of
        # kimodo.tools.ensure_batched that this file needs.
        import types
        if 'kimodo.tools' not in sys.modules:
            tools_mod = types.ModuleType('kimodo.tools')

            def ensure_batched(**dims_by_name):
                def deco(fn):
                    def wrapper(*args, **kwargs):
                        if not dims_by_name:
                            return fn(*args, **kwargs)
                        need_dim = next(iter(dims_by_name.values()))
                        x = args[0]
                        squeezed = False
                        if isinstance(x, torch.Tensor) and x.dim() == need_dim - 1:
                            x = x.unsqueeze(0)
                            squeezed = True
                        out = fn(x, *args[1:], **kwargs)
                        if squeezed and isinstance(out, torch.Tensor):
                            out = out.squeeze(0)
                        return out
                    return wrapper
                return deco

            tools_mod.ensure_batched = ensure_batched
            sys.modules['kimodo.tools'] = tools_mod
        smooth_path = (
            PROJECT_ROOT / 'ref_repo' / 'KIMODO' / 'kimodo' /
            'kimodo' / 'motion_rep' / 'smooth_root.py'
        )
        spec = importlib.util.spec_from_file_location(
            '_kimodo_smooth_root_direct', smooth_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        get_smooth_root_pos = mod.get_smooth_root_pos

    was_numpy = isinstance(soma_pos, np.ndarray)
    if was_numpy:
        soma_pos_t = torch.from_numpy(soma_pos).float()
    else:
        soma_pos_t = soma_pos
    root_pos = soma_pos_t[:, 0, :]
    if root_pos.shape[0] < 3:
        smooth_root = root_pos
    else:
        smooth_root = get_smooth_root_pos(root_pos)
    smooth_2d = smooth_root[:, [0, 2]].to(
        device=soma_pos_t.device, dtype=soma_pos_t.dtype)
    return smooth_2d.cpu().numpy() if was_numpy else smooth_2d.to(soma_pos_t.device)


def kimodo_optional_smooth_root_2d(soma_pos):
    if os.environ.get("KIMODO_USE_SMOOTH_ROOT_CONSTRAINTS", "0") != "1":
        return None
    return kimodo_smooth_root_2d_from_positions(soma_pos)


def _cat_optional_smooth_root(parts):
    if any(p is None for p in parts):
        return None
    import torch
    return torch.cat(parts, dim=0)


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


def soma30_to_soma77(soma30_global_rots, soma30_root_positions, soma30):
    """Expand SOMA-30 global rotations + root positions to SOMA-77 FK output."""
    import torch
    from kimodo.skeleton.transforms import global_rots_to_local_rots

    device = soma30.joint_parents.device
    dtype = soma30_global_rots.dtype
    soma30_global_rots = soma30_global_rots.to(device=device)
    soma30_root_positions = soma30_root_positions.to(device=device, dtype=dtype)
    soma30_local_rots = global_rots_to_local_rots(soma30_global_rots, soma30)
    soma77_local_rots = soma30.to_SOMASkeleton77(soma30_local_rots)
    soma77_skel = soma30.somaskel77.to(device)
    soma77_global_rots, soma77_posed_joints, _ = soma77_skel.fk(
        soma77_local_rots, soma30_root_positions)
    root_delta = soma30_root_positions - soma77_posed_joints[:, 0, :]
    if torch.max(torch.abs(root_delta)) > 1e-8:
        soma77_posed_joints = soma77_posed_joints + root_delta[:, None, :]
    return (
        soma77_posed_joints.detach().cpu().numpy().astype(np.float32),
        soma77_global_rots.detach().cpu().numpy().astype(np.float32),
    )


# ============================================================================
# Constraint builders per task
# ============================================================================

def build_constraints_e2(skeleton, soma30_rots, soma30_pos, T, setting, caption=""):
    """E2 In-betweening: six settings mirroring the backend v2 ablation.

    Each setting chooses which temporal region is given as GT context
    (FullBodyConstraintSet on those frame indices); KIMODO then solves
    for the rest of the frames.
    """
    FullBodyConstraintSet = _make_fullbody_with_rot_constraint_set()
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
        smooth_root_2d=(
            None if kimodo_optional_smooth_root_2d(soma30_pos) is None
            else kimodo_optional_smooth_root_2d(soma30_pos)[frame_idx]
        ),
        to_crop=False,
    )
    return [constraint]


def build_constraints_e3(skeleton, soma30_rots, soma30_pos, T, setting,
                         caption="", motion_135=None, bone_offsets=None):
    """E3 Keyframe interpolation: keep selected anchor frames.

    2026-04-26 (unified naming): mirrors backend m2m_eval_tasks.py E3 v2.
      every_5f -> K=5      every_10f -> K=10     every_15f -> K=15
      every_30f -> K=30    every_60f -> K=60
      adaptive -> top-K acceleration-peak frames (sparse mode), exactly
                  matching the M2M backend's `detect_keyframes_from_motion`
                  semantics. Requires `motion_135` + `bone_offsets`; if
                  unavailable we fall back to K=30 uniform with a warning.
    Legacy A/B/C/D names are accepted for backward compatibility but should
    be migrated by callers ASAP.
    """
    FullBodyConstraintSet = _make_fullbody_with_rot_constraint_set()
    import torch

    legacy_alias = {'A': 'every_30f', 'B': 'every_60f',
                    'C': 'every_15f', 'D': 'adaptive'}
    setting = legacy_alias.get(setting, setting)

    intervals = {
        'every_5f': 5, 'every_10f': 10, 'every_15f': 15,
        'every_30f': 30, 'every_60f': 60,
    }

    if setting == 'adaptive':
        if motion_135 is not None and bone_offsets is not None:
            try:
                from hftrainer.evaluation.motion.m2m_eval_tasks import (
                    detect_keyframes_from_motion,
                )
                m135_np = (motion_135.detach().cpu().numpy()
                           if hasattr(motion_135, 'detach') else motion_135)
                bo_np = (bone_offsets.detach().cpu().numpy()
                         if hasattr(bone_offsets, 'detach') else bone_offsets)
                kf_idx = detect_keyframes_from_motion(
                    m135_np, bo_np,
                    sparse=True,
                    target_density=1.0 / 30.0,
                    peak_distance=10,
                )
                # Clamp to valid frame range and dedupe.
                kf_idx = sorted({int(f) for f in kf_idx if 0 <= int(f) < T})
                if len(kf_idx) == 0:
                    kf_idx = [0, T - 1]
                frames = kf_idx
                print(f"    [KIMODO E3-adaptive] {len(frames)} keyframes "
                      f"from acc peaks (T={T}): {frames[:8]}"
                      + ("..." if len(frames) > 8 else ""))
            except Exception as e:
                print(f"    [KIMODO E3-adaptive] detect failed ({e}); "
                      f"fallback to K=30")
                frames = list(range(0, T, 30))
                if frames[-1] != T - 1:
                    frames.append(T - 1)
        else:
            print("    [KIMODO E3-adaptive] motion_135/bone_offsets not "
                  "supplied; fallback to K=30 uniform")
            frames = list(range(0, T, 30))
            if frames[-1] != T - 1:
                frames.append(T - 1)
    else:
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
        smooth_root_2d=(
            None if kimodo_optional_smooth_root_2d(soma30_pos) is None
            else kimodo_optional_smooth_root_2d(soma30_pos)[frame_idx]
        ),
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
        # FullBodyConstraintSet expects base end-effector names and expands
        # LeftFoot to [LeftFoot, LeftToeBase] internally.
        'C_rhand_lfoot':   (['RightHand', 'LeftFoot'], [21, 10], 15),
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
    """E8 Loop animation (v2 redesign 2026-04-26).

    Setting A — pure loop, caption-aware. Frame 0 and frame T-1 both anchor on
        the GT's first-frame pose. T = sample.num_frames passed by caller; the
        model has to fill the (T-2) intermediate frames and close the loop.

    Setting D — loop completion, uncond. The CALLER has already sliced
        ``soma30_rots`` / ``soma30_pos`` to the GT-tail-as-condition window
        (length T_gt_eff = soma30_rots.shape[0]), and passes
        ``T = T_gt_eff + N_append`` (the resolved adaptive append length).
        We therefore constrain frame indices [0..T_gt_eff-1] with the sliced
        GT data and frame [T-1] with the GT's first-frame pose (the loop
        target), letting the model fill the (N_append-1) gap frames.

    The legacy hardcoded ``append_map={'B':30,'C':60,'D':90}`` from the
    pre-2026-04-26 cohort is gone — the only D variant in m2m_eval_tasks.py is
    now ``_loop_append='auto'`` and the value is resolved upstream by
    ``compute_transition_length``.
    """
    FullBodyConstraintSet = _make_fullbody_with_rot_constraint_set()
    import torch

    if setting == 'A':
        # Pure loop: frame 0 and frame T-1 both = soma30[0]
        frames = [0, T - 1]
        frame_idx = torch.tensor(frames, dtype=torch.long)
        loop_rots = torch.stack([soma30_rots[0], soma30_rots[0]], dim=0)
        loop_pos = torch.stack([soma30_pos[0], soma30_pos[0]], dim=0)

        constraint = FullBodyConstraintSet(
            skeleton,
            frame_indices=frame_idx,
            global_joints_positions=loop_pos,
            global_joints_rots=loop_rots,
            smooth_root_2d=kimodo_optional_smooth_root_2d(loop_pos),
            to_crop=False,
        )
        return [constraint]

    # Setting D (loop completion). soma30_* already trimmed to GT-tail length.
    T_gt_eff = int(soma30_rots.shape[0])
    if T <= T_gt_eff:
        # Caller forgot to add N_append — fall back to "lock everything,
        # generate nothing" rather than crash.
        frames = list(range(T_gt_eff))
        constraint_rots = soma30_rots
        constraint_pos = soma30_pos
    else:
        # Lock the GT tail at frames [0..T_gt_eff-1] and the loop-back pose at
        # frame [T-1]; frames [T_gt_eff..T-2] are unconstrained (generated).
        frames = list(range(T_gt_eff)) + [T - 1]
        loop_target_rots = getattr(build_constraints_e8, "_loop_target_rots", None)
        loop_target_pos = getattr(build_constraints_e8, "_loop_target_pos", None)
        if loop_target_rots is None or loop_target_pos is None:
            loop_target_rots = soma30_rots[0:1]
            loop_target_pos = soma30_pos[0:1]
        constraint_rots = torch.cat([soma30_rots, loop_target_rots], dim=0)
        constraint_pos = torch.cat([soma30_pos, loop_target_pos], dim=0)
    frame_idx = torch.tensor(frames, dtype=torch.long)

    constraint = FullBodyConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=constraint_pos,
        global_joints_rots=constraint_rots,
        smooth_root_2d=kimodo_optional_smooth_root_2d(constraint_pos),
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
    FullBodyConstraintSet = _make_fullbody_with_rot_constraint_set()
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

    # KIMODO's frame constraints are local in time: locking condition frames
    # does not by itself force the adjacent generated frame to have matching
    # velocity/pose. Add transition-edge anchors by default, using linear
    # extrapolation from the source/target clips instead of duplicating the
    # condition frame. This is still inference-time conditioning, not a frame
    # edit or postprocess.
    endpoint_frames = []
    endpoint_rots = []
    endpoint_pos = []
    if _kimodo_use_boundary_anchors() and N_transition >= 2:
        def _extrapolate_next_pos(pos_seq):
            if pos_seq.shape[0] >= 2:
                return pos_seq[-1:] + (pos_seq[-1:] - pos_seq[-2:-1])
            return pos_seq[-1:]

        def _extrapolate_prev_pos(pos_seq):
            if pos_seq.shape[0] >= 2:
                return pos_seq[:1] - (pos_seq[1:2] - pos_seq[:1])
            return pos_seq[:1]

        def _extrapolate_next_rot(rot_seq):
            if rot_seq.shape[0] >= 2:
                delta = torch.matmul(rot_seq[-1:], rot_seq[-2:-1].transpose(-1, -2))
                return torch.matmul(delta, rot_seq[-1:])
            return rot_seq[-1:]

        def _extrapolate_prev_rot(rot_seq):
            if rot_seq.shape[0] >= 2:
                delta = torch.matmul(rot_seq[1:2], rot_seq[:1].transpose(-1, -2))
                return torch.matmul(delta.transpose(-1, -2), rot_seq[:1])
            return rot_seq[:1]

        endpoint_frames = [N_cond_a, N_cond_a + N_transition - 1]
        endpoint_rots = [
            _extrapolate_next_rot(soma30_rots_a),
            _extrapolate_prev_rot(soma30_rots_b),
        ]
        endpoint_pos = [
            _extrapolate_next_pos(soma30_pos_a),
            _extrapolate_prev_pos(soma30_pos_b),
        ]

    frames = frames_a + endpoint_frames + frames_b

    frame_idx = torch.tensor(frames, dtype=torch.long)
    constraint_rots = torch.cat([a_tail_rots, *endpoint_rots, b_head_rots], dim=0)
    constraint_pos = torch.cat([a_tail_pos, *endpoint_pos, b_head_pos], dim=0)

    constraint = FullBodyConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=constraint_pos,
        global_joints_rots=constraint_rots,
        smooth_root_2d=_cat_optional_smooth_root([
            None if kimodo_optional_smooth_root_2d(soma30_pos_a) is None
            else kimodo_optional_smooth_root_2d(soma30_pos_a)[-N_cond_a:],
            *[kimodo_optional_smooth_root_2d(p) for p in endpoint_pos],
            None if kimodo_optional_smooth_root_2d(soma30_pos_b) is None
            else kimodo_optional_smooth_root_2d(soma30_pos_b)[:N_cond_b],
        ]),
        to_crop=False,
    )
    return [constraint]


def build_constraints_e15(skeleton, soma30_rots, soma30_pos,
                           soma30_rots_target, soma30_pos_target,
                           T, setting, N_cond_tail, N_transition, caption=""):
    """E15 Transition to target first frame: constrain motion tail + target first.

    Sequence: [motion_tail(N_cond_tail) | transition(N_transition) | target_first(1)]
    """
    FullBodyConstraintSet = _make_fullbody_with_rot_constraint_set()
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
        smooth_root_2d=_cat_optional_smooth_root([
            None if kimodo_optional_smooth_root_2d(soma30_pos) is None
            else kimodo_optional_smooth_root_2d(soma30_pos)[-N_cond_tail:],
            None if kimodo_optional_smooth_root_2d(soma30_pos_target) is None
            else kimodo_optional_smooth_root_2d(soma30_pos_target)[:1],
        ]),
        to_crop=False,
    )
    return [constraint]


def build_constraints_e15_prepend(skeleton, soma30_rots_P, soma30_pos_P,
                                   soma30_rots_A, soma30_pos_A,
                                   T_total, N_transition, N_cond_A_used,
                                   caption=""):
    """E15 prepend (2026-04-27 v2): constrain frame 0 (= P, target start pose)
    and frames [N_transition..T_total-1] (= the K_used head frames of motion A).

    Sequence layout (T_total = N_transition + N_cond_A_used):
        frame 0           : P = target_motion[0]            (mask=0, condition)
        frames 1..N-1     : transition (generated)          (mask=1)
        frames N..T-1     : A[:N_cond_A_used] (condition)   (mask=0)

    Args:
        skeleton: SOMA30 skeleton instance.
        soma30_rots_P : (1, 30, 3, 3)  target's first frame in canonical space.
        soma30_pos_P  : (1, 30, 3)
        soma30_rots_A : (N_cond_A_used, 30, 3, 3) head of A in canonical space.
        soma30_pos_A  : (N_cond_A_used, 30, 3)
        T_total       : total frames = N_transition + N_cond_A_used.
        N_transition  : number of prepended frames (frame 0 = P, frames
                        1..N_transition-1 are generated).
        N_cond_A_used : number of A frames fed to the model.
    """
    FullBodyConstraintSet = _make_fullbody_with_rot_constraint_set()
    import torch

    endpoint_frames = []
    endpoint_rots = []
    endpoint_pos = []
    if _kimodo_use_boundary_anchors() and N_transition >= 2:
        def _extrapolate_prev_pos(pos_seq):
            if pos_seq.shape[0] >= 2:
                return pos_seq[:1] - (pos_seq[1:2] - pos_seq[:1])
            return pos_seq[:1]

        def _extrapolate_prev_rot(rot_seq):
            if rot_seq.shape[0] >= 2:
                delta = torch.matmul(rot_seq[1:2], rot_seq[:1].transpose(-1, -2))
                return torch.matmul(delta.transpose(-1, -2), rot_seq[:1])
            return rot_seq[:1]

        # Frame 1 is a one-frame hold after the locked start pose; frame
        # N_transition-1 is the natural predecessor of A[0].
        endpoint_frames = [1, N_transition - 1]
        endpoint_rots = [soma30_rots_P[:1], _extrapolate_prev_rot(soma30_rots_A)]
        endpoint_pos = [soma30_pos_P[:1], _extrapolate_prev_pos(soma30_pos_A)]

    frames_P = [0]
    frames_A = list(range(N_transition, T_total))
    frame_idx = torch.tensor(frames_P + endpoint_frames + frames_A,
                             dtype=torch.long)

    constraint_rots = torch.cat(
        [soma30_rots_P[:1], *endpoint_rots, soma30_rots_A], dim=0)
    constraint_pos = torch.cat(
        [soma30_pos_P[:1], *endpoint_pos, soma30_pos_A], dim=0)

    constraint = FullBodyConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=constraint_pos,
        global_joints_rots=constraint_rots,
        smooth_root_2d=_cat_optional_smooth_root([
            None if kimodo_optional_smooth_root_2d(soma30_pos_P) is None
            else kimodo_optional_smooth_root_2d(soma30_pos_P)[:1],
            *[kimodo_optional_smooth_root_2d(p) for p in endpoint_pos],
            None if kimodo_optional_smooth_root_2d(soma30_pos_A) is None
            else kimodo_optional_smooth_root_2d(soma30_pos_A)[:N_cond_A_used],
        ]),
        to_crop=False,
    )
    return [constraint]


def build_constraints_e16(skeleton, soma30_rots_target, soma30_pos_target,
                           soma30_rots, soma30_pos,
                           T, setting, N_cond_head, N_transition, caption=""):
    """E16 Transition from target last frame: constrain target last + motion head.

    Sequence: [target_last(1) | transition(N_transition) | motion_head(N_cond_head)]
    """
    FullBodyConstraintSet = _make_fullbody_with_rot_constraint_set()
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
        smooth_root_2d=_cat_optional_smooth_root([
            None if kimodo_optional_smooth_root_2d(soma30_pos_target) is None
            else kimodo_optional_smooth_root_2d(soma30_pos_target)[-1:],
            None if kimodo_optional_smooth_root_2d(soma30_pos) is None
            else kimodo_optional_smooth_root_2d(soma30_pos)[:N_cond_head],
        ]),
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
                    caption, T, task_id, setting, fps=30,
                    motion_135=None, bone_offsets=None,
                    canon_anchor_frame: int = 0,
                    loop_target_rots=None, loop_target_pos=None):
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
        setting: e.g. 'every_30f', 'adaptive'.
        fps: frame rate.
        motion_135: optional (T, 135) raw SMPL-22 motion for builders that
            need motion-aware constraints (E3 `adaptive` calls
            `detect_keyframes_from_motion`).
        bone_offsets: optional (22, 3) bone offsets, paired with motion_135.
    """
    import torch

    # ------------------------------------------------------------------
    # Canonicalize the retargeted SOMA30 motion into KIMODO's training
    # frame (frame-0 XZ at origin, heading 0). This moves the constraint
    # data from raw world coordinates into the distribution KIMODO was
    # trained on, so constraint tasks generalize the same way T2M does.
    # See kimodo_compute_canon_transform / kimodo_apply_canon above.
    # ------------------------------------------------------------------
    R_yaw, t_xz, _heading0 = kimodo_compute_canon_transform(
        soma30_pos, skeleton, anchor_frame=canon_anchor_frame)
    soma30_rots_c, soma30_pos_c = kimodo_apply_canon(
        soma30_rots, soma30_pos, R_yaw, t_xz)

    # E8-D needs the loop target to stay tied to the ORIGINAL motion's first
    # frame even when the GT condition is clipped to a tail window. Build the
    # target in world space, then canonicalize it with the SAME transform as
    # the main sequence so the final constraint lives in the correct frame.
    prev_loop_rots = getattr(build_constraints_e8, "_loop_target_rots", None)
    prev_loop_pos = getattr(build_constraints_e8, "_loop_target_pos", None)
    if task_id == 'E8' and setting == 'D' and loop_target_rots is not None and loop_target_pos is not None:
        loop_target_rots_c, loop_target_pos_c = kimodo_apply_canon(
            loop_target_rots, loop_target_pos, R_yaw, t_xz)
        build_constraints_e8._loop_target_rots = loop_target_rots_c
        build_constraints_e8._loop_target_pos = loop_target_pos_c
    else:
        build_constraints_e8._loop_target_rots = None
        build_constraints_e8._loop_target_pos = None

    builder = CONSTRAINT_BUILDERS.get(task_id)
    if builder is None and task_id != 'E6':
        build_constraints_e8._loop_target_rots = prev_loop_rots
        build_constraints_e8._loop_target_pos = prev_loop_pos
        return None, {}, {}

    try:
        if task_id == 'E6':
            # E6 needs gt_pos_22 for foot contact detection; the detection
            # itself is per-frame local (ankle Y), so canonicalization doesn't
            # shift which frames are contact frames — but the constraint
            # positions fed to KIMODO must be in canonical world.
            constraints = build_constraints_e6(
                skeleton, soma30_rots_c, soma30_pos_c, gt_pos_22, T, setting, caption)
        elif task_id == 'E3':
            # E3 uses motion-aware logic in `adaptive` setting; pass through
            # the raw motion_135 + bone_offsets so the builder can call
            # `detect_keyframes_from_motion`.
            constraints = build_constraints_e3(
                skeleton, soma30_rots_c, soma30_pos_c, T, setting, caption,
                motion_135=motion_135, bone_offsets=bone_offsets)
        else:
            constraints = builder(skeleton, soma30_rots_c, soma30_pos_c, T, setting, caption)
    finally:
        build_constraints_e8._loop_target_rots = prev_loop_rots
        build_constraints_e8._loop_target_pos = prev_loop_pos

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

    # Sliding-window split: cap each segment ≤ KIMODO_SAFE_LEN to stay inside
    # the model's training distribution (~10s).  See _split_num_frames docstring.
    seg_lens = _split_num_frames(num_frames)
    is_multi = len(seg_lens) > 1
    seg_prompts = ([caption] if caption else [""]) * len(seg_lens)
    # KIMODO API quirk: __call__'s single-prompt branch wants
    #   constraint_lst: list[list[Constraint]]   # per-sample → batch=[constraints]
    # while _multiprompt wants
    #   constraint_lst: list[Constraint]         # one shared list, segments are
    #                                              cropped via constraint.crop_move
    # See ref_repo/KIMODO/kimodo/kimodo/model/kimodo_model.py:178 vs :483.
    constraint_arg = constraints if is_multi else [constraints]
    t0 = time.time()
    try:
        output = model(
            seg_prompts,
            seg_lens,
            num_denoising_steps=DIFFUSION_STEPS,
            constraint_lst=constraint_arg,
            cfg_weight=_kimodo_cfg_weight(),
            num_samples=1,
            return_numpy=True,
            multi_prompt=is_multi,
            # post_processing=False: KIMODO's correct_motion (cm-level
            # foot-skate / root-margin cleanup) requires the C++
            # motion_correction extension which is not installed on every
            # GPU host.  The real fix for the long-rollout joint explosion
            # is the sliding-window split above (T>=300 jumped 59-100%
            # before; split keeps every segment <=240).  post_processing
            # is purely cosmetic and not worth the dependency.
            post_processing=_kimodo_post_processing(),
        )
    except Exception as e:
        print(f"    KIMODO inference error: {e}  (seg_lens={seg_lens})")
        return None, {"inference_time": round(time.time() - t0, 2), "_error": str(e)[:100]}, {}

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
    #
    # 2026-04-26 follow-up: After the canonical-anchor fix (anchoring at
    # A_tail's first frame == model frame 0 instead of A's absolute frame
    # 0) plus the foot-floor alignment in `_place_b_custom`, this fallback
    # is *empirically zero* on all 200 E14 samples. Kept here as a
    # defensive guard and as a regression-detection signal: if you ever
    # see large `y_anchor_delta` values reappear in the per-sample
    # metrics, the canonicalization upstream has likely regressed.
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

    # MPJPE between predicted and GT (only for generated frames).
    # 2026-04-26: pred can be longer than gt for E8-D (T = T_gt_eff +
    # N_append, gt = T_gt_eff). Only score the overlapping prefix to avoid
    # broadcasting against missing GT frames.
    if gt_pos_22 is not None:
        n_overlap = min(pred_pos_22.shape[0], gt_pos_22.shape[0], T)
        if n_overlap > 0:
            diff = np.sqrt(np.sum(
                (pred_pos_22[:n_overlap] - gt_pos_22[:n_overlap]) ** 2,
                axis=-1))
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


def _select_kimodo_candidate(output, selection_boundaries=None):
    """Pick the candidate with the smallest objective seam/internal jumps.

    This is an inference-time best-of-N strategy, not a frame edit: all frames
    come from one raw KIMODO sample.  It guards against stochastic samples that
    satisfy hard condition frames but arrive at the seam with a wildly rotated
    knee/hand.
    """
    posed = output.get("posed_joints")
    rots = output.get("global_rot_mats")
    if posed is None or rots is None or posed.ndim != 4 or posed.shape[0] <= 1:
        return 0
    P = np.asarray(posed, dtype=np.float64)
    G = np.asarray(rots, dtype=np.float64)
    B, T, J = G.shape[:3]

    from tools.append_kimodo_e15_context_soma77 import _bootstrap_kimodo_skeleton
    _bootstrap_kimodo_skeleton()
    from kimodo.skeleton.definitions import SOMASkeleton30
    parents = SOMASkeleton30().somaskel77.joint_parents.numpy()
    if len(parents) != J:
        return 0

    boundaries = []
    for b in selection_boundaries or []:
        b = int(b)
        if 0 <= b < T - 1:
            boundaries.append(b)
    if not boundaries:
        boundaries = [0, T - 2]

    local = G.copy()
    for j in range(J):
        p = parents[j]
        if p >= 0:
            local[:, :, j] = np.einsum(
                "btij,btjk->btik",
                np.swapaxes(G[:, :, p], -1, -2),
                G[:, :, j],
            )

    def _angle_deg(mats):
        trace = np.trace(mats, axis1=-2, axis2=-1)
        cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    dpos = np.linalg.norm(np.diff(P, axis=1), axis=-1)  # (B,T-1,J)
    drot = _angle_deg(
        np.einsum(
            "btij,btjk->btik",
            np.swapaxes(local[:, :-1], -1, -2),
            local[:, 1:],
        )
    )  # (B,T-1,J)

    seam_pos = dpos[:, boundaries].max(axis=(1, 2))
    seam_rot = drot[:, boundaries].max(axis=(1, 2)) / 180.0
    # Penalize internal joint explosions, but keep seam continuity primary.
    internal_rot = np.percentile(drot, 99.0, axis=(1, 2)) / 180.0
    internal_pos = np.percentile(dpos, 99.0, axis=(1, 2))
    score = seam_pos + 0.2 * seam_rot + 0.05 * internal_pos + 0.05 * internal_rot
    return int(np.argmin(score))


def _run_kimodo_with_constraints(model, skeleton, constraints, caption, T,
                                  gt_pos_22, fps=30,
                                  canon_transform=None,
                                  selection_boundaries=None,
                                  first_heading_angle=None):
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

    # Sliding-window split (see _split_num_frames + constraint_arg note above).
    seg_lens = _split_num_frames(num_frames)
    is_multi = len(seg_lens) > 1
    seg_prompts = ([caption] if caption else [""]) * len(seg_lens)
    constraint_arg = constraints if is_multi else [constraints]
    t0 = time.time()
    try:
        output = model(
            seg_prompts,
            seg_lens,
            num_denoising_steps=DIFFUSION_STEPS,
            constraint_lst=constraint_arg,
            cfg_weight=_kimodo_cfg_weight(),
            num_samples=_kimodo_num_candidates(),
            return_numpy=True,
            multi_prompt=is_multi,
            post_processing=_kimodo_post_processing(),
            first_heading_angle=first_heading_angle,
        )
    except Exception as e:
        print(f"    KIMODO inference error: {e}  (seg_lens={seg_lens})")
        return None, {"inference_time": round(time.time() - t0, 2), "_error": str(e)[:100]}, {}

    elapsed = time.time() - t0

    posed_joints = output["posed_joints"]
    if posed_joints.ndim == 4 and posed_joints.shape[0] > 1:
        keep_idx = _select_kimodo_candidate(output, selection_boundaries)
        for key, val in list(output.items()):
            if hasattr(val, "ndim") and val.ndim >= 1 and val.shape[0] == posed_joints.shape[0]:
                output[key] = val[keep_idx:keep_idx + 1]
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
    # 2026-04-27 (KIMODO_uncond every_10f gap-fill): allow resuming a partial
    # run by skipping the first ``--start-idx`` samples. The loop still
    # enumerates 0..max_samples (so prompt indices stay aligned with the
    # full datalist + the existing NPZ filenames), but only computes /
    # writes outputs for ``i >= start_idx``. result.json is merged with any
    # pre-existing one in the same output dir so the final json contains
    # all per_sample entries (old + newly computed).
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Skip prompt indices < start-idx (resume mode). '
                             'Existing NPZ + result.json entries < start-idx are '
                             'preserved as-is.')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='Stop before this prompt index. Used with '
                             '--start-idx for parallel sharded evaluation.')
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
    # 2026-04-26: align with eval_m2m_v2_all_tasks.py — when set, prefer
    # `<base>_rewritten.json` so KIMODO sees the same standardized
    # ("A person ...", 12-20 word) captions HyMotion is fed. Without this
    # flag, KIMODO eats the raw `caption_en` field which for E8 v2 contains
    # 1-2 word stubs ("cleave", "slash") and mixed-language strings — i.e.
    # KIMODO_caption was effectively running unconditioned.
    parser.add_argument('--use-rewritten', action='store_true',
                        help='If set, load the rewritten datalist '
                             '(eval_e*_rewritten.json) for caption-carrying '
                             'tasks (caption-aware models only).')
    parser.add_argument('--data-file-override', type=str, default=None,
                        help='Override task.data_file for ALL tasks. Used '
                             'for ablation runs that want to point KIMODO at '
                             'a custom datalist without editing the registry.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Base random seed. Each sample uses seed+index; '
                             'set negative to keep stochastic sampling.')
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
            if getattr(args, 'data_file_override', None):
                _data_file_name = args.data_file_override
            else:
                _data_file_name = _per_setting_data_file or task.data_file
            # 2026-04-26: prefer `<base>_rewritten.json` when --use-rewritten
            # is set, mirroring eval_m2m_v2_all_tasks.py. Caption-aware
            # KIMODO runs MUST use the rewritten captions to be a fair
            # comparison against caption_local_phase2 / caption_global_phase2.
            if args.use_rewritten and args.use_caption == 'yes':
                base_no_ext = os.path.splitext(_data_file_name)[0]
                _rewritten_name = base_no_ext + '_rewritten.json'
                _rewritten_path = PROJECT_ROOT / "data" / "eval" / "m2m_v2" / _rewritten_name
                if _rewritten_path.is_file():
                    _data_file_name = _rewritten_name
                else:
                    print(f"  [note] no rewritten datalist for {_data_file_name}, "
                          f"falling back to raw")
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

            # Resume mode: when start_idx > 0 and an existing result.json
            # exists, load its per_sample list and keep entries with
            # _sample_idx < start_idx untouched. This way the final json
            # written below covers the union (old < start_idx) ∪ (newly
            # generated >= start_idx).
            per_sample = []
            _existing_result_path = os.path.join(args.output_dir, task_key, 'result.json')
            if args.start_idx > 0 and os.path.exists(_existing_result_path):
                try:
                    with open(_existing_result_path) as _rf:
                        _existing_data = json.load(_rf)
                    for _s in _existing_data.get('per_sample', []):
                        _idx = _s.get('_sample_idx')
                        if isinstance(_idx, int) and _idx < args.start_idx:
                            per_sample.append(_s)
                    print(f"  Resume mode: loaded {len(per_sample)} existing "
                          f"per_sample entries with _sample_idx < {args.start_idx}")
                except Exception as _e:
                    print(f"  Resume warning: failed to load existing result.json: {_e}")
            for i, sample in enumerate(samples):
                _e8_layout = None
                _e8_gt_tail_pos22 = None
                _e8_target_pos22 = None
                _e8_cond_soma77 = None
                _e8_target_soma77 = None
                if i < args.start_idx:
                    continue
                if args.end_idx is not None and i >= args.end_idx:
                    continue
                if args.seed is not None and int(args.seed) >= 0:
                    import random
                    import torch as _torch_seed
                    seed_i = int(args.seed) + int(i)
                    random.seed(seed_i)
                    np.random.seed(seed_i)
                    _torch_seed.manual_seed(seed_i)
                    if _torch_seed.cuda.is_available():
                        _torch_seed.cuda.manual_seed_all(seed_i)
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

                    # ---- E8 loop completion (D): adjust T for KIMODO ----
                    if task_id == 'E8' and '_loop_append' in setting_kwargs:
                        # 2026-04-26 v2 alignment: mirror the M2M backend's
                        # E8-D handling (eval_m2m_v2_all_tasks.py around
                        # line ~1580). We need to (1) resolve N_append the
                        # same way (auto -> compute_transition_length on
                        # motion[-1]<->motion[0] using root + joint-pos +
                        # joint-angle 3-term rule), and (2) clip the GT
                        # condition to T_gt_eff = T_PAD_MAX - N_append so
                        # KIMODO solves the same problem M2M does.
                        N_append_raw = setting_kwargs['_loop_append']
                        if (isinstance(N_append_raw, str) and
                                N_append_raw == 'auto') or \
                           (isinstance(N_append_raw, int) and N_append_raw <= 0):
                            joints_first = gt_pos[0]   # (22, 3)
                            joints_last = gt_pos[-1]
                            N_append = compute_transition_length(
                                joints_last[0], joints_first[0],
                                speed_per_frame=float(setting_kwargs.get(
                                    '_transition_speed', 0.015)),
                                min_frames=int(setting_kwargs.get(
                                    '_transition_min', 30)),
                                max_frames=int(setting_kwargs.get(
                                    '_transition_max', 150)),
                                joints_a_end=joints_last,
                                joints_b_start=joints_first,
                                pose_speed_per_frame=float(setting_kwargs.get(
                                    '_pose_speed', 0.015)),
                                motion_a_end_135=motion[-1],
                                motion_b_start_135=motion[0],
                                joint_angle_speed_per_frame=float(setting_kwargs.get(
                                    '_joint_angle_speed', 0.20)),
                            )
                        else:
                            N_append = int(N_append_raw)

                        # Clip GT-tail condition the same way M2M does.
                        T_PAD_MAX = 360
                        T_gt_full = int(soma30_rots.shape[0])
                        T_gt_eff = max(1, min(T_gt_full,
                                              T_PAD_MAX - N_append))
                        soma30_rots_d = soma30_rots[-T_gt_eff:]
                        soma30_pos_d = soma30_pos[-T_gt_eff:]
                        loop_target_rots = soma30_rots[:1]
                        loop_target_pos = soma30_pos[:1]
                        T_total = T_gt_eff + N_append
                        print(f"    [E8-D KIMODO] N_append={N_append} "
                              f"T_gt_full={T_gt_full} T_gt_eff={T_gt_eff} "
                              f"T_total={T_total}")
                        pred_pos, metrics, soma_data = evaluate_sample(
                            model, skeleton, soma30_rots_d, soma30_pos_d,
                            gt_pos[-T_gt_eff:],
                            caption, T_total, task_id, setting_name, fps_val,
                            motion_135=motion, bone_offsets=bone_offsets,
                            canon_anchor_frame=0,
                            loop_target_rots=loop_target_rots,
                            loop_target_pos=loop_target_pos,
                        )
                        # E8-D mirrors E14/E15's visualization contract:
                        # condition frames shown in the dashboard must be the
                        # exact SOMA frames sent into KIMODO, not the model's
                        # soft-constraint reconstruction.  Replace the GT tail
                        # and final loop target on save below.
                        _e8_layout = {
                            'task': 'E8',
                            'T_gt_eff': int(T_gt_eff),
                            'N_append': int(N_append),
                            'n_dropped_prefix': int(T_gt_full - T_gt_eff),
                        }
                        _e8_gt_tail_pos22 = gt_pos[-T_gt_eff:].astype(np.float32)
                        _e8_target_pos22 = gt_pos[0:1].astype(np.float32)
                        _e8_cond_soma77 = soma30_to_soma77(
                            soma30_rots_d,
                            soma30_pos_d[:, 0, :].contiguous(),
                            skeleton,
                        )
                        _e8_target_soma77 = soma30_to_soma77(
                            loop_target_rots,
                            loop_target_pos[:, 0, :].contiguous(),
                            skeleton,
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
                            N_transition=1, yaw_offset_deg=yaw_offset_deg,
                            bone_offsets=bone_offsets)
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
                        # 2026-04-26: pass bone_offsets so _place_b_custom
                        # can foot-floor align B against A (was floating
                        # in setting M because B's pelvis Y was preserved
                        # raw from B's own clip, ignoring A's floor).
                        motion_b_world = _place_b_custom(
                            motion_a, motion_b,
                            placement=placement,
                            N_transition=N_transition,
                            forward_step=forward_step,
                            yaw_offset_deg=yaw_offset_deg,
                            bone_offsets=bone_offsets)

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

                        # Canonicalize around the FIRST frame of A_tail
                        # (= the model's output frame 0). Previously this
                        # used A's frame 0 (pre-tail), which placed A_tail
                        # at an arbitrary XZ + heading inside the model's
                        # input window — OOD relative to KIMODO's training
                        # distribution where every clip starts at (0, *,
                        # 0) facing +Z.
                        # 2026-04-26 fix: anchor at soma30_pos_a[-N_cond_a]
                        # so model frame 0 lives at the canonical origin.
                        # Both A and B share the same transform to preserve
                        # their relative world geometry post-canonicalization.
                        R_yaw, t_xz, _ = kimodo_compute_canon_transform(
                            soma30_pos_a[-N_cond_a:], skeleton)
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
                            canon_transform=(R_yaw, t_xz),
                            selection_boundaries=[
                                N_cond_a - 1,
                                N_cond_a + N_transition - 1,
                            ])
                        metrics['transition_length'] = N_transition
                        metrics['n_cond_a'] = N_cond_a
                        metrics['n_cond_b'] = N_cond_b

                    # ---- E15: prepend to start pose (2026-04-27 v2) ----
                    # Mirrors the M2M pipeline's `_use_start_pose` branch
                    # in tools/eval_m2m_v2_all_tasks.py so KIMODO and
                    # HyMotion solve the same problem geometry.
                    elif task_id == 'E15' and '_use_start_pose' in setting_kwargs:
                        target_path = sample.get('target_motion_path', '')
                        if target_path and not os.path.isabs(target_path) and \
                                not os.path.exists(target_path):
                            target_path = os.path.join(args.motion_data_dir,
                                                       target_path)
                        target_motion = load_motion_135d(target_path)
                        if target_motion is None:
                            per_sample.append({"_sample_idx": i,
                                               "_error": "target load failed"})
                            continue

                        import torch as _torch
                        from hftrainer.pipelines.motion.transition_utils import (
                            canonicalize_segment,
                        )
                        from tools.eval_m2m_v2_all_tasks import _place_b_custom
                        from hftrainer.evaluation.motion.m2m_eval_tasks import (
                            compute_cond_length,
                        )

                        motion_a_full = motion  # (len_A, 135), world coords

                        # ── Step 1: P = target[0] in canonical (origin) space ──
                        P_single = target_motion[0:1].copy()
                        P_canon_t, _Rp, _Op = canonicalize_segment(
                            _torch.from_numpy(P_single).float(),
                            anchor_frame=0,
                            rotation_space='local',
                        )
                        P_canon = P_canon_t.numpy()  # (1, 135)

                        # ── Step 2: place A so A[0] sits at P's xz=(0,0) ──
                        yaw_offset_deg = float(setting_kwargs.get(
                            '_yaw_offset_deg', 0.0))
                        motion_a_placed_full = _place_b_custom(
                            P_canon, motion_a_full,
                            placement='overlap',
                            N_transition=1,
                            yaw_offset_deg=yaw_offset_deg,
                            y_align='preserve_b',
                        )

                        # ── Step 3: adaptive N_transition ──
                        P_joints = motion135_to_positions_np(
                            P_canon, bone_offsets)[0]              # (22, 3)
                        A0_joints = motion135_to_positions_np(
                            motion_a_placed_full[0:1], bone_offsets)[0]
                        if '_prepend_N' in setting_kwargs:
                            N_transition = int(setting_kwargs['_prepend_N'])
                        else:
                            N_transition = compute_transition_length(
                                P_joints[0], A0_joints[0],
                                speed_per_frame=float(setting_kwargs.get(
                                    '_transition_speed', 0.015)),
                                min_frames=int(setting_kwargs.get(
                                    '_transition_min', 15)),
                                max_frames=int(setting_kwargs.get(
                                    '_transition_max', 90)),
                                joints_a_end=P_joints,
                                joints_b_start=A0_joints,
                                pose_speed_per_frame=float(setting_kwargs.get(
                                    '_pose_speed', 0.015)),
                                motion_a_end_135=P_canon[0],
                                motion_b_start_135=motion_a_placed_full[0],
                                joint_angle_speed_per_frame=float(setting_kwargs.get(
                                    '_joint_angle_speed', 0.20)),
                            )

                        # ── Step 4: N_cond_A truncation ──
                        n_cond_a_policy = setting_kwargs.get(
                            '_n_cond_a_policy', None)
                        n_cond_a_frames = setting_kwargs.get(
                            '_n_cond_a_frames', None)
                        if n_cond_a_policy == 'adaptive':
                            model_K = compute_cond_length(
                                motion_a_placed_full,
                                T_src=int(motion_a_placed_full.shape[0]),
                                N_transition=N_transition,
                                side='head',
                            )
                        elif n_cond_a_frames is not None:
                            model_K = int(min(int(n_cond_a_frames),
                                              motion_a_placed_full.shape[0]))
                        else:
                            model_K = int(motion_a_placed_full.shape[0])
                        motion_a_placed = motion_a_placed_full[:model_K]

                        # 360-frame ceiling guard (same as M2M)
                        T_total = N_transition + motion_a_placed.shape[0]
                        MAX_FRAMES = 360
                        if T_total > MAX_FRAMES:
                            keep_A = MAX_FRAMES - N_transition
                            if keep_A <= 1:
                                per_sample.append({
                                    "_sample_idx": i,
                                    "_error": f"E15 N_transition={N_transition} "
                                              f"leaves no room for A under "
                                              f"{MAX_FRAMES}-frame window",
                                })
                                continue
                            motion_a_placed = motion_a_placed[:keep_A]
                            T_total = N_transition + motion_a_placed.shape[0]

                        # ── Step 5: assemble + final canonicalize (near-id) ──
                        transition_pad = (
                            np.zeros((N_transition - 1, 135), dtype=np.float32)
                            if N_transition > 1 else
                            np.zeros((0, 135), dtype=np.float32))
                        world_segment = np.concatenate(
                            [P_canon, transition_pad, motion_a_placed], axis=0)
                        world_segment_t = _torch.from_numpy(
                            world_segment).float()
                        canon_segment_t, _Rc, _Oc = canonicalize_segment(
                            world_segment_t, anchor_frame=0,
                            rotation_space='local',
                        )
                        canon_segment = canon_segment_t.numpy()

                        # ── Step 6: retarget to SOMA30 ──
                        # P slice (single frame) and A_placed slice
                        # (model_K frames after the pad).
                        canon_P = canon_segment[0:1]
                        canon_A = canon_segment[N_transition:]
                        rots_P, pos_P = smpl22_to_soma30_retarget(
                            canon_P, bone_offsets)
                        rots_A, pos_A = smpl22_to_soma30_retarget(
                            canon_A, bone_offsets)

                        # Optional ablation: a second SOMA-space
                        # canonicalization can be enabled for debugging, but
                        # the default path mirrors upstream KIMODO's
                        # FullBody demo more closely: feed the already
                        # canonical SMPL->SOMA constraints directly and let
                        # FullBodyConstraintSet use pelvis/root XZ unless a
                        # separate dense Root2D track is explicitly supplied.
                        R_kimodo = t_kimodo = heading_kimodo = None
                        if os.environ.get("KIMODO_E15_SOMA_CANON", "0") == "1":
                            soma_pos_for_canon = torch.cat([pos_P, pos_A], dim=0)
                            R_kimodo, t_kimodo, heading_kimodo = (
                                kimodo_compute_canon_transform(
                                    soma_pos_for_canon, skeleton,
                                    anchor_frame=0)
                            )
                            rots_P, pos_P = kimodo_apply_canon(
                                rots_P, pos_P, R_kimodo, t_kimodo)
                            rots_A, pos_A = kimodo_apply_canon(
                                rots_A, pos_A, R_kimodo, t_kimodo)

                        # Update gt_pos for metric eval to the canonical
                        # full segment (so ee/jitter/etc are computed in
                        # the same frame KIMODO solves in).
                        gt_pos_canon = motion135_to_positions_np(
                            canon_segment, bone_offsets)

                        N_cond_tail = int(canon_A.shape[0])

                        constraints = build_constraints_e15_prepend(
                            skeleton, rots_P, pos_P, rots_A, pos_A,
                            T_total, N_transition, N_cond_tail, caption)

                        first_heading_angle = None
                        if _kimodo_use_first_constraint_heading():
                            from kimodo.motion_rep.feature_utils import (
                                compute_heading_angle,
                            )
                            first_heading_angle = compute_heading_angle(
                                pos_P[:1], skeleton)[0]

                        pred_pos, metrics, soma_data = _run_kimodo_with_constraints(
                            model, skeleton, constraints, caption, T_total,
                            gt_pos_canon, fps_val,
                            canon_transform=(
                                (R_kimodo, t_kimodo)
                                if R_kimodo is not None else None
                            ),
                            selection_boundaries=[0, N_transition - 1],
                            first_heading_angle=first_heading_angle)
                        metrics['transition_length'] = N_transition
                        metrics['n_cond_a_used'] = N_cond_tail
                        if heading_kimodo is not None:
                            metrics['kimodo_heading_canon'] = float(
                                heading_kimodo.detach().cpu())

                    # ---- E15 (legacy): transition to target first frame ----
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
                            gt_pos, fps_val,
                            selection_boundaries=[N_cond_tail - 1, T_total - 2])
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
                            gt_pos, fps_val,
                            selection_boundaries=[0, N_transition])
                        metrics['transition_length'] = N_transition

                    # ---- Standard tasks ----
                    else:
                        pred_pos, metrics, soma_data = evaluate_sample(
                            model, skeleton, soma30_rots, soma30_pos, gt_pos,
                            caption, T, task_id, setting_name, fps_val,
                            motion_135=motion, bone_offsets=bone_offsets,
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
                    pred_pos_save = pred_pos
                    if task_id == 'E8' and setting_name == 'D' and _e8_layout is not None:
                        n_cond_e8 = int(_e8_layout.get('T_gt_eff', 0))
                        if (_e8_gt_tail_pos22 is not None and
                                n_cond_e8 > 0 and n_cond_e8 <= pred_pos.shape[0]):
                            pred_pos_save = pred_pos.copy()
                            pred_pos_save[:n_cond_e8] = _e8_gt_tail_pos22[:n_cond_e8]
                        if (_e8_target_pos22 is not None and
                                pred_pos_save.shape[0] >= 1):
                            if pred_pos_save is pred_pos:
                                pred_pos_save = pred_pos.copy()
                            pred_pos_save[-1:] = _e8_target_pos22[:1]
                    save_fields = dict(
                        positions=pred_pos_save,
                        translation=pred_pos_save[:, 0],
                    )
                    # Include SOMA-77 data for mesh rendering
                    if soma_data.get('posed_joints') is not None:
                        save_fields['posed_joints'] = soma_data['posed_joints']
                    if soma_data.get('global_rot_mats') is not None:
                        save_fields['global_rot_mats'] = soma_data['global_rot_mats']
                    if (task_id == 'E8' and setting_name == 'D' and
                            _e8_layout is not None and
                            'posed_joints' in save_fields and
                            'global_rot_mats' in save_fields):
                        n_cond_e8 = int(_e8_layout.get('T_gt_eff', 0))
                        cond_pair = _e8_cond_soma77
                        target_pair = _e8_target_soma77
                        if cond_pair is not None and n_cond_e8 > 0:
                            cond_pj, cond_gr = cond_pair
                            if n_cond_e8 <= save_fields['posed_joints'].shape[0]:
                                save_fields['posed_joints'] = save_fields['posed_joints'].copy()
                                save_fields['global_rot_mats'] = save_fields['global_rot_mats'].copy()
                                save_fields['posed_joints'][:n_cond_e8] = cond_pj[:n_cond_e8]
                                save_fields['global_rot_mats'][:n_cond_e8] = cond_gr[:n_cond_e8]
                        if target_pair is not None and save_fields['posed_joints'].shape[0] >= 1:
                            target_pj, target_gr = target_pair
                            if 'posed_joints' not in save_fields or 'global_rot_mats' not in save_fields:
                                pass
                            else:
                                if not save_fields['posed_joints'].flags.writeable:
                                    save_fields['posed_joints'] = save_fields['posed_joints'].copy()
                                    save_fields['global_rot_mats'] = save_fields['global_rot_mats'].copy()
                                save_fields['posed_joints'][-1:] = target_pj[:1]
                                save_fields['global_rot_mats'][-1:] = target_gr[:1]

                    # ── 2026-04-27 viz-bug fix: write layout_json ──────
                    # The dashboard's stitchSourceMotionsGeneric() needs
                    # layout.N_cond_a / N_cond_b / N_transition to slice
                    # the gray prefix/suffix at the correct frames. The
                    # M2M v2 pipeline already writes this (see
                    # tools/eval_m2m_v2_all_tasks.py L3261-3293), but
                    # KIMODO had been silently dropping it — so the
                    # dashboard fell back to the legacy 5/15/30 estimate
                    # for KIMODO, which now (post v5 dynamic budgets,
                    # actual N_cond_a/b ≈ 3-10, N_transition ≈ 30-120)
                    # cuts the gray context at the WRONG frame and
                    # produces a visible "jump" between blue
                    # (network output) and gray (motion B suffix).
                    # Mirror the M2M v2 layout schema so the dashboard
                    # treats both models identically.
                    _layout = None
                    if task_id == 'E14':
                        _actual_t = int(
                            save_fields['posed_joints'].shape[0]
                            if 'posed_joints' in save_fields else pred_pos.shape[0]
                        )
                        _n_cond_b = int(N_cond_b)
                        _expected_t = int(N_cond_a) + int(N_transition) + _n_cond_b
                        if _actual_t != _expected_t:
                            _n_cond_b = max(0, min(_n_cond_b, _actual_t - int(N_cond_a) - int(N_transition)))
                        _layout = {
                            'task': 'E14',
                            'N_cond_a': int(N_cond_a),
                            'N_transition': int(N_transition),
                            'N_cond_b': int(_n_cond_b),
                        }
                    elif task_id == 'E15':
                        # Prepend layout: [P(1) | gen(N_trans-1) | A_used]
                        # N_cond_A = number of A frames fed; len_A_full and
                        # len_A are equal here because KIMODO doesn't viz
                        # the full A separately. For dashboard parity with
                        # M2M v2's _e15_len_A_full / _e15_len_A pair, we
                        # emit both.
                        _layout = {
                            'task': 'E15',
                            'N_transition': int(N_transition),
                            'N_cond_A': int(N_cond_tail),
                            'len_A': int(N_cond_tail),
                            'len_A_full': int(N_cond_tail),
                        }
                    elif task_id == 'E16':
                        _layout = {
                            'task': 'E16',
                            'N_transition': int(N_transition),
                            'N_cond_head': int(N_cond_head),
                        }
                    elif task_id == 'E8' and setting_name == 'D':
                        _layout = _e8_layout
                    if _layout is not None:
                        import json as _json
                        save_fields['layout_json'] = np.frombuffer(
                            _json.dumps(_layout).encode('utf-8'),
                            dtype=np.uint8)

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
