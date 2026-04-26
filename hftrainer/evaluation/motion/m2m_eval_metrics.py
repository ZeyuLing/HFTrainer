"""Evaluation metrics for HyMotion M2M v2 (E1-E12 tasks).

All metrics operate on 135-dim motion tensors (3 abs_transl + 22*6 rot6d).
For FK-based metrics (MPJPE, end-effector error), bone_offsets are required.

Units: all position-based metrics output in **meters** by default (matching
the training data coordinate system). Callers can multiply by 1000 for mm.

Supported metrics:
  - MPJPE (mean per-joint position error via FK)
  - Jitter (3rd-order finite difference on positions)
  - Bone length CV (coefficient of variation across frames)
  - Trajectory ADE/FDE (root XZ trajectory)
  - Boundary smoothness (acceleration jump at mask transition)
  - Loop continuity (first-last frame MPJPE + velocity diff)
  - End-effector position error
  - Foot ground metrics (penetration, float, skating)
  - FK consistency (rotation FK vs position channel consistency)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# =====================================================================
# FK-based position computation
# =====================================================================

def motion135_to_positions_np(
    motion: np.ndarray,
    bone_offsets: np.ndarray,
) -> np.ndarray:
    """Convert 135-dim motion to world-space joint positions via FK.

    Args:
        motion: (T, 135) denormalized motion.
        bone_offsets: (22, 3) bone offsets.

    Returns:
        positions: (T, 22, 3) world-space joint positions.
    """
    motion_t = torch.from_numpy(motion).float()
    offsets_t = torch.from_numpy(bone_offsets).float()

    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
    with torch.no_grad():
        world_pos, _, _, _ = motion135_to_fk(motion_t, offsets_t, rotation_space='local')
    return world_pos.numpy()


def motion135_to_positions_global_np(
    motion: np.ndarray,
    bone_offsets: np.ndarray,
) -> np.ndarray:
    """Convert 135-dim motion (global rotation space) to positions via FK.

    Args:
        motion: (T, 135) denormalized motion in global rotation space.
        bone_offsets: (22, 3) bone offsets.

    Returns:
        positions: (T, 22, 3) world-space joint positions.
    """
    motion_t = torch.from_numpy(motion).float()
    offsets_t = torch.from_numpy(bone_offsets).float()

    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
    with torch.no_grad():
        world_pos, _, _, _ = motion135_to_fk(motion_t, offsets_t, rotation_space='global')
    return world_pos.numpy()


# =====================================================================
# MPJPE
# =====================================================================

def compute_mpjpe(
    pred_pos: np.ndarray,
    gt_pos: np.ndarray,
    mask: Optional[np.ndarray] = None,
    joint_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Mean Per-Joint Position Error.

    Args:
        pred_pos: (T, 22, 3) predicted positions.
        gt_pos: (T, 22, 3) ground truth positions.
        mask: (T, 135) optional mask. If given, only computes on masked
            (mask=1) frames. If None, computes on all frames.
        joint_indices: optional subset of joints to evaluate.

    Returns:
        Dict with mpjpe_mean, mpjpe_per_joint (list).
    """
    T = pred_pos.shape[0]
    assert pred_pos.shape == gt_pos.shape == (T, 22, 3)

    # Determine which frames to evaluate
    if mask is not None:
        # mask=1 means generated region
        frame_mask = mask.max(axis=-1) > 0.5  # (T,)
        # Handle length mismatch between mask and pred_pos (E14/E15/E16
        # stitched sequences include prefix/suffix outside the mask window).
        if frame_mask.shape[0] != T:
            T_mask = frame_mask.shape[0]
            if T_mask < T:
                pad_total = T - T_mask
                left = pad_total // 2
                right = pad_total - left
                frame_mask = np.concatenate(
                    [np.zeros(left, dtype=bool), frame_mask,
                     np.zeros(right, dtype=bool)], axis=0)
            else:
                crop_total = T_mask - T
                start = crop_total // 2
                frame_mask = frame_mask[start:start + T]
    else:
        frame_mask = np.ones(T, dtype=bool)

    if not frame_mask.any():
        return {'mpjpe_mean': 0.0, 'mpjpe_per_joint': [0.0] * 22}

    pred_sel = pred_pos[frame_mask]  # (N, 22, 3)
    gt_sel = gt_pos[frame_mask]

    if joint_indices is not None:
        pred_sel = pred_sel[:, joint_indices]
        gt_sel = gt_sel[:, joint_indices]

    # Per-joint L2 error
    per_joint_err = np.linalg.norm(pred_sel - gt_sel, axis=-1)  # (N, J)
    mpjpe_mean = float(per_joint_err.mean())
    mpjpe_per_joint = per_joint_err.mean(axis=0).tolist()

    return {
        'mpjpe_mean': mpjpe_mean,
        'mpjpe_per_joint': mpjpe_per_joint,
    }


# =====================================================================
# Jitter (position-based)
# =====================================================================

def compute_jitter_positions(positions: np.ndarray, fps: float = 30.0) -> float:
    """Compute jitter as mean jerk (3rd-order finite diff) of joint positions.

    Args:
        positions: (T, 22, 3) world-space positions.
        fps: frames per second.

    Returns:
        Jitter value in m/s^3.
    """
    if positions.shape[0] < 4:
        return 0.0
    dt = 1.0 / fps
    # 3rd order finite difference: x[t+3] - 3x[t+2] + 3x[t+1] - x[t]
    diff3 = positions[3:] - 3 * positions[2:-1] + 3 * positions[1:-2] - positions[:-3]
    jerk = diff3 / (dt ** 3)
    return float(np.mean(np.linalg.norm(jerk.reshape(jerk.shape[0], -1), axis=-1)))


def compute_jitter_135(motion: np.ndarray) -> float:
    """Compute jitter directly on 135-dim representation (no FK needed).

    Args:
        motion: (T, 135) denormalized motion.

    Returns:
        Jitter value (unitless, on normalized scale).
    """
    if motion.shape[0] < 4:
        return 0.0
    diff3 = motion[3:] - 3 * motion[2:-1] + 3 * motion[1:-2] - motion[:-3]
    return float(np.mean(np.abs(diff3)))


# =====================================================================
# Bone Length Consistency
# =====================================================================

# SMPL-22 bone pairs: (parent, child)
SMPL_22_BONE_PAIRS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (4, 7), (5, 8),
    (7, 10), (8, 11), (3, 6), (6, 9), (9, 12), (9, 13), (9, 14),
    (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]


def compute_bone_length_cv(positions: np.ndarray) -> Dict[str, float]:
    """Coefficient of variation of bone lengths across frames.

    Args:
        positions: (T, 22, 3) world-space positions.

    Returns:
        Dict with bone_length_cv_mean, bone_length_cv_max.
    """
    T = positions.shape[0]
    if T < 2:
        return {'bone_length_cv_mean': 0.0, 'bone_length_cv_max': 0.0}

    bone_lengths = []
    for p, c in SMPL_22_BONE_PAIRS:
        bl = np.linalg.norm(positions[:, c] - positions[:, p], axis=-1)  # (T,)
        bone_lengths.append(bl)
    bone_lengths = np.array(bone_lengths)  # (21, T)

    mean_bl = bone_lengths.mean(axis=1)  # (21,)
    std_bl = bone_lengths.std(axis=1)    # (21,)
    cv = std_bl / (mean_bl + 1e-8)       # (21,)

    return {
        'bone_length_cv_mean': float(cv.mean()),
        'bone_length_cv_max': float(cv.max()),
    }


# =====================================================================
# Trajectory metrics
# =====================================================================

def compute_trajectory_metrics(
    pred_motion: np.ndarray,
    gt_motion: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Trajectory ADE/FDE on root XZ plane.

    Args:
        pred_motion: (T, 135) predicted motion.
        gt_motion: (T, 135) ground truth motion.
        mask: (T, 135) optional mask.

    Returns:
        Dict with trajectory_ade, trajectory_fde (in meters).
    """
    pred_root_xz = pred_motion[:, [0, 2]]  # XZ from translation
    gt_root_xz = gt_motion[:, [0, 2]]

    if mask is not None:
        frame_mask = mask.max(axis=-1) > 0.5
        if not frame_mask.any():
            return {'trajectory_ade': 0.0, 'trajectory_fde': 0.0}
        # Handle length mismatch between mask (masked region only) and
        # pred_motion (may include stitched prefix/suffix for E14/E15/E16).
        # Align the mask to pred_motion length by centering if shorter,
        # truncating if longer. If shapes don't match after this alignment,
        # pad mask with False on both sides so indexing never out-of-bounds.
        T_pred = pred_root_xz.shape[0]
        T_mask = frame_mask.shape[0]
        if T_mask != T_pred:
            if T_mask < T_pred:
                # Pad mask to match pred length: False for prefix/suffix,
                # keeping the given mask values as the center.
                pad_total = T_pred - T_mask
                left = pad_total // 2
                right = pad_total - left
                frame_mask = np.concatenate(
                    [np.zeros(left, dtype=bool), frame_mask,
                     np.zeros(right, dtype=bool)], axis=0)
            else:
                # Mask longer than pred — take center slice matching pred.
                crop_total = T_mask - T_pred
                start = crop_total // 2
                frame_mask = frame_mask[start:start + T_pred]
        pred_root_xz = pred_root_xz[frame_mask]
        gt_root_xz = gt_root_xz[frame_mask]

    ade = float(np.mean(np.linalg.norm(pred_root_xz - gt_root_xz, axis=-1)))

    # FDE: error at last frame
    fde = float(np.linalg.norm(pred_root_xz[-1] - gt_root_xz[-1]))

    return {
        'trajectory_ade': ade,
        'trajectory_fde': fde,
    }


def compute_heading_error(
    pred_motion: np.ndarray,
    gt_motion: np.ndarray,
    bone_offsets: np.ndarray,
) -> float:
    """Heading error in degrees based on pelvis forward direction.

    Computes from FK world rotation of pelvis (joint 0).

    Args:
        pred_motion: (T, 135) predicted.
        gt_motion: (T, 135) ground truth.
        bone_offsets: (22, 3).

    Returns:
        Mean heading error in degrees.
    """
    pred_t = torch.from_numpy(pred_motion).float()
    gt_t = torch.from_numpy(gt_motion).float()
    offsets_t = torch.from_numpy(bone_offsets).float()

    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
    with torch.no_grad():
        _, pred_rot, _, _ = motion135_to_fk(pred_t, offsets_t)
        _, gt_rot, _, _ = motion135_to_fk(gt_t, offsets_t)

    # Extract pelvis rotation -> forward direction (Z axis in world)
    pred_fwd = pred_rot[:, 0, :, 2].numpy()  # (T, 3)
    gt_fwd = gt_rot[:, 0, :, 2].numpy()

    # Project to XZ plane
    pred_fwd_xz = pred_fwd[:, [0, 2]]
    gt_fwd_xz = gt_fwd[:, [0, 2]]

    # Normalize
    pred_fwd_xz = pred_fwd_xz / (np.linalg.norm(pred_fwd_xz, axis=-1, keepdims=True) + 1e-8)
    gt_fwd_xz = gt_fwd_xz / (np.linalg.norm(gt_fwd_xz, axis=-1, keepdims=True) + 1e-8)

    # Angle between
    cos_angle = np.clip(np.sum(pred_fwd_xz * gt_fwd_xz, axis=-1), -1, 1)
    angles_rad = np.arccos(cos_angle)
    return float(np.degrees(angles_rad).mean())


# =====================================================================
# Boundary smoothness
# =====================================================================

def compute_boundary_smoothness(
    motion: np.ndarray,
    mask: np.ndarray,
    bone_offsets: Optional[np.ndarray] = None,
    boundary_width: int = 3,
    fps: float = 30.0,
) -> Dict[str, float]:
    """Acceleration discontinuity at mask boundary frames.

    Measures how smooth the transition is between known and generated regions.

    Args:
        motion: (T, 135) output motion (blended/composited).
        mask: (T, 135) mask (0=known, 1=generated).
        bone_offsets: if provided, compute on positions; otherwise on raw 135-dim.
        boundary_width: frames around boundary to evaluate.
        fps: frames per second.

    Returns:
        Dict with boundary_accel_jump, boundary_mpjpe (if gt available).
    """
    T = motion.shape[0]
    if T < 5:
        return {'boundary_accel_jump': 0.0}

    # Find boundary frames
    mask_per_frame = mask.max(axis=-1) > 0.5  # (T,)
    # Align mask length to motion length (E14/E15/E16 stitched sequences).
    if mask_per_frame.shape[0] != T:
        T_mask = mask_per_frame.shape[0]
        if T_mask < T:
            pad_total = T - T_mask
            left = pad_total // 2
            right = pad_total - left
            mask_per_frame = np.concatenate(
                [np.zeros(left, dtype=bool), mask_per_frame,
                 np.zeros(right, dtype=bool)], axis=0)
        else:
            crop_total = T_mask - T
            start = crop_total // 2
            mask_per_frame = mask_per_frame[start:start + T]
    boundary_frames = set()
    for t in range(1, T):
        if mask_per_frame[t] != mask_per_frame[t - 1]:
            for dt in range(-boundary_width, boundary_width + 1):
                ft = t + dt
                if 0 <= ft < T:
                    boundary_frames.add(ft)

    if not boundary_frames:
        return {'boundary_accel_jump': 0.0}

    # Compute acceleration
    if bone_offsets is not None:
        data = motion135_to_positions_np(motion, bone_offsets)  # (T, 22, 3)
        data = data.reshape(T, -1)  # (T, 66)
    else:
        data = motion  # (T, 135)

    dt = 1.0 / fps
    if T < 3:
        return {'boundary_accel_jump': 0.0}

    # Acceleration: 2nd order finite diff
    accel = (data[2:] - 2 * data[1:-1] + data[:-2]) / (dt ** 2)

    # Accel jump: diff of acceleration at boundaries
    boundary_list = sorted(boundary_frames)
    accel_jumps = []
    for bf in boundary_list:
        # Acceleration at bf and bf-1
        if 1 <= bf < T - 1 and 1 <= bf - 1 < T - 1:
            a1 = accel[bf - 1]  # accel index = frame - 1
            a0 = accel[bf - 2] if bf >= 2 else a1
            jump = np.linalg.norm(a1 - a0)
            accel_jumps.append(jump)

    return {
        'boundary_accel_jump': float(np.mean(accel_jumps)) if accel_jumps else 0.0,
    }


# =====================================================================
# Loop continuity
# =====================================================================

def compute_loop_continuity(
    motion: np.ndarray,
    bone_offsets: Optional[np.ndarray] = None,
    fps: float = 30.0,
) -> Dict[str, float]:
    """Loop continuity metrics: first-last frame MPJPE + velocity diff.

    Args:
        motion: (T, 135) output motion.
        bone_offsets: for FK-based position comparison.
        fps: frames per second.

    Returns:
        Dict with loop_position_error, loop_velocity_error.
    """
    T = motion.shape[0]
    if T < 3:
        return {'loop_position_error': 0.0, 'loop_velocity_error': 0.0}

    if bone_offsets is not None:
        pos = motion135_to_positions_np(motion, bone_offsets)  # (T, 22, 3)
        # Position error: first vs last frame
        pos_err = float(np.mean(np.linalg.norm(pos[0] - pos[-1], axis=-1)))
        # Velocity error: velocity at frame 0 vs frame T-1
        vel_first = (pos[1] - pos[0]) * fps
        vel_last = (pos[-1] - pos[-2]) * fps
        vel_err = float(np.mean(np.linalg.norm(vel_first - vel_last, axis=-1)))
    else:
        pos_err = float(np.mean(np.abs(motion[0] - motion[-1])))
        vel_first = (motion[1] - motion[0]) * fps
        vel_last = (motion[-1] - motion[-2]) * fps
        vel_err = float(np.mean(np.abs(vel_first - vel_last)))

    return {
        'loop_position_error': pos_err,
        'loop_velocity_error': vel_err,
    }


# =====================================================================
# End-effector position error
# =====================================================================

# Joint name -> index mapping
JOINT_NAME_TO_IDX = {
    'pelvis': 0, 'l_hip': 1, 'r_hip': 2, 'spine1': 3,
    'l_knee': 4, 'r_knee': 5, 'spine2': 6, 'l_ankle': 7,
    'r_ankle': 8, 'spine3': 9, 'l_foot': 10, 'r_foot': 11,
    'neck': 12, 'l_collar': 13, 'r_collar': 14, 'head': 15,
    'l_shoulder': 16, 'r_shoulder': 17, 'l_elbow': 18, 'r_elbow': 19,
    'l_wrist': 20, 'r_wrist': 21,
}


def compute_end_effector_error(
    pred_pos: np.ndarray,
    constraint_positions: np.ndarray,
    constraint_frames: np.ndarray,
    constraint_joints: np.ndarray,
) -> Dict[str, float]:
    """End-effector position error at constraint points.

    For E4 we care about how close the model ACTUALLY got to the
    target position at each constraint frame. The metric is simply the
    Euclidean distance ‖pred_joint - target‖ (meters), aggregated over
    all (frame, joint) constraint pairs.

    Args:
        pred_pos: (T, 22, 3) predicted world positions from FK.
        constraint_positions: (N, 3) target positions.
        constraint_frames: (N,) frame indices.
        constraint_joints: (N,) joint indices.

    Returns:
        Dict with:
          ee_error_mean:   average distance across all constraint points
          ee_error_max:    worst-case distance (one frame can dominate)
          ee_error_p50:    median (robust centre vs mean)
          ee_error_p95:    95th percentile (catches tail failures that
                           mean hides, distinguishes "all bad" from
                           "mostly fine + a few bad")
          ee_error_std:    spread; high = inconsistent across frames
          ee_hit_rate_2cm  / _5cm / _10cm:
                           fraction of constraint points within that
                           distance of target. 2 cm is "essentially
                           satisfied", 10 cm is "within joint radius".
    """
    N = len(constraint_positions)
    empty = {
        'ee_error_mean': 0.0, 'ee_error_max': 0.0,
        'ee_error_p50': 0.0, 'ee_error_p95': 0.0,
        'ee_error_std': 0.0,
        'ee_hit_rate_2cm': 0.0, 'ee_hit_rate_5cm': 0.0,
        'ee_hit_rate_10cm': 0.0,
    }
    if N == 0:
        return empty

    errors = []
    for i in range(N):
        f = int(constraint_frames[i])
        j = int(constraint_joints[i])
        if f < pred_pos.shape[0]:
            err = np.linalg.norm(pred_pos[f, j] - constraint_positions[i])
            errors.append(err)

    if not errors:
        return empty

    errors_np = np.asarray(errors, dtype=np.float32)
    return {
        'ee_error_mean': float(errors_np.mean()),
        'ee_error_max': float(errors_np.max()),
        'ee_error_p50': float(np.percentile(errors_np, 50)),
        'ee_error_p95': float(np.percentile(errors_np, 95)),
        'ee_error_std': float(errors_np.std()),
        'ee_hit_rate_2cm': float((errors_np < 0.02).mean()),
        'ee_hit_rate_5cm': float((errors_np < 0.05).mean()),
        'ee_hit_rate_10cm': float((errors_np < 0.10).mean()),
    }


# =====================================================================
# Foot ground metrics (simplified)
# =====================================================================

# Ankle/foot joint indices
L_ANKLE_IDX = 7
R_ANKLE_IDX = 8
L_FOOT_IDX = 10
R_FOOT_IDX = 11
FOOT_JOINTS = [L_ANKLE_IDX, R_ANKLE_IDX, L_FOOT_IDX, R_FOOT_IDX]


def compute_foot_ground_metrics(
    positions: np.ndarray,
    ground_y: float = 0.0,
    contact_threshold: float = 0.05,  # 5cm
    skating_threshold: float = 0.01,  # 1cm/frame
    fps: float = 30.0,
) -> Dict[str, float]:
    """Foot ground interaction metrics.

    Args:
        positions: (T, 22, 3) world-space joint positions.
        ground_y: ground plane Y coordinate.
        contact_threshold: height below which foot is considered in contact.
        skating_threshold: velocity threshold for skating detection.
        fps: frames per second.

    Returns:
        Dict with penetration, float_height, skating_ratio, avg_skate.
    """
    T = positions.shape[0]
    if T < 2:
        return {
            'foot_penetration': 0.0, 'foot_float': 0.0,
            'foot_skating_ratio': 0.0, 'foot_avg_skate': 0.0,
        }

    foot_pos = positions[:, FOOT_JOINTS, :]  # (T, 4, 3)
    foot_y = foot_pos[:, :, 1]  # (T, 4) Y coordinate

    # Penetration: how much below ground
    penetration = np.maximum(ground_y - foot_y, 0)  # (T, 4)
    avg_penetration = float(penetration.mean())

    # Float: height above ground when in "contact" (low velocity)
    foot_vel = np.linalg.norm(np.diff(foot_pos, axis=0), axis=-1) * fps  # (T-1, 4)
    # Contact frames: low velocity
    contact = foot_vel < skating_threshold * fps  # (T-1, 4)
    # Float height at contact frames
    float_heights = []
    for t in range(T - 1):
        for j in range(4):
            if contact[t, j] and foot_y[t, j] > ground_y + contact_threshold:
                float_heights.append(foot_y[t, j] - ground_y)
    avg_float = float(np.mean(float_heights)) if float_heights else 0.0

    # Skating: XZ velocity during ground contact
    foot_xz_vel = np.diff(foot_pos[:, :, [0, 2]], axis=0) * fps  # (T-1, 4, 2)
    foot_xz_speed = np.linalg.norm(foot_xz_vel, axis=-1)  # (T-1, 4)

    # Contact mask: foot close to ground
    contact_mask = foot_y[:-1] < ground_y + contact_threshold  # (T-1, 4)

    skating_frames = contact_mask & (foot_xz_speed > skating_threshold * fps)
    skating_ratio = float(skating_frames.sum()) / max(contact_mask.sum(), 1)
    skating_speeds = foot_xz_speed[skating_frames]
    avg_skate = float(skating_speeds.mean()) if len(skating_speeds) > 0 else 0.0

    return {
        'foot_penetration': avg_penetration,
        'foot_float': avg_float,
        'foot_skating_ratio': skating_ratio,
        'foot_avg_skate': avg_skate,
    }


# =====================================================================
# FK consistency (rotation FK vs position channel)
# =====================================================================

def compute_fk_consistency(
    motion_with_pos: np.ndarray,
    bone_offsets: np.ndarray,
    pos_start_dim: int = 135,
) -> float:
    """Consistency between rotation-FK positions and position channel.

    Only relevant for 198-dim representation where positions are explicit.
    For 135-dim, this returns 0.

    Args:
        motion_with_pos: (T, D) motion potentially with position channels.
        bone_offsets: (22, 3).
        pos_start_dim: where position channels start.

    Returns:
        Mean L2 error between FK positions and position channel.
    """
    D = motion_with_pos.shape[-1]
    if D <= pos_start_dim:
        return 0.0  # No position channel in 135-dim

    # FK from rotation → (T, 22, 3) including pelvis
    fk_pos = motion135_to_positions_np(motion_with_pos[:, :135], bone_offsets)
    # Position channel: 63-dim = 21 joints × 3 (pelvis excluded, XZ relative to pelvis, Y absolute)
    pos_dim = D - pos_start_dim
    if pos_dim < 63:
        return 0.0  # Not enough position dimensions
    pos_channel = motion_with_pos[:, pos_start_dim:pos_start_dim + 63].reshape(-1, 21, 3)
    # Compare only the 21 non-pelvis joints (fk_pos[:, 1:])
    return float(np.mean(np.linalg.norm(fk_pos[:, 1:] - pos_channel, axis=-1)))


# =====================================================================
# Aggregate metrics runner
# =====================================================================

def compute_all_metrics(
    pred_motion: np.ndarray,
    gt_motion: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    bone_offsets: np.ndarray,
    rotation_space: str = 'local',
    fps: float = 30.0,
    compute_fk: bool = True,
) -> Dict[str, float]:
    """Compute all available metrics for a single sample.

    Args:
        pred_motion: (T, 135) predicted denormalized motion.
        gt_motion: (T, 135) ground truth denormalized motion (optional).
        mask: (T, 135) task mask (0=known, 1=generated).
        bone_offsets: (22, 3) bone offsets for FK.
        rotation_space: 'local' or 'global'.
        fps: frames per second.
        compute_fk: whether to compute FK (slower but more metrics).

    Returns:
        Dict of metric_name -> float.
    """
    metrics: Dict[str, float] = {}

    fk_fn = motion135_to_positions_global_np if rotation_space == 'global' else motion135_to_positions_np
    # GT is ALWAYS in local rotation space (from dataset), regardless of model
    gt_fk_fn = motion135_to_positions_np

    # --- Always computed ---
    metrics['jitter_135'] = compute_jitter_135(pred_motion)

    pred_pos = None
    gt_pos = None

    if compute_fk:
        pred_pos = fk_fn(pred_motion, bone_offsets)  # (T, 22, 3)
        metrics['jitter_pos'] = compute_jitter_positions(pred_pos, fps)

        # bone_length_cv removed 2026-04-23 (always ~0 for M2M outputs — not
        # informative, takes dashboard real estate). compute_bone_length_cv
        # remains available for ad-hoc debugging.

        foot = compute_foot_ground_metrics(pred_pos, fps=fps)
        metrics.update(foot)

    if gt_motion is not None:
        # Trajectory metrics (no FK needed)
        traj = compute_trajectory_metrics(pred_motion, gt_motion, mask)
        metrics.update(traj)

        if compute_fk:
            gt_pos = gt_fk_fn(gt_motion, bone_offsets)

            # MPJPE - all frames
            mpjpe_all = compute_mpjpe(pred_pos, gt_pos)
            metrics['mpjpe_all'] = mpjpe_all['mpjpe_mean']

            # MPJPE - masked region only
            if mask is not None:
                mpjpe_masked = compute_mpjpe(pred_pos, gt_pos, mask=mask)
                metrics['mpjpe_masked'] = mpjpe_masked['mpjpe_mean']

                # MPJPE - unmasked region (should be ~0 for imputation)
                inv_mask = 1.0 - mask
                mpjpe_unmasked = compute_mpjpe(pred_pos, gt_pos, mask=inv_mask)
                metrics['mpjpe_unmasked'] = mpjpe_unmasked['mpjpe_mean']

    if mask is not None:
        bnd = compute_boundary_smoothness(
            pred_motion, mask, bone_offsets if compute_fk else None, fps=fps)
        metrics.update(bnd)

    return metrics


def aggregate_metrics(
    per_sample_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-sample metrics into mean/std/median.

    Args:
        per_sample_metrics: list of metric dicts.

    Returns:
        Dict of metric_name -> {mean, std, median, min, max}.
    """
    if not per_sample_metrics:
        return {}

    all_keys = set()
    for m in per_sample_metrics:
        all_keys.update(m.keys())

    agg = {}
    for key in sorted(all_keys):
        vals = [m[key] for m in per_sample_metrics if key in m]
        if not vals:
            continue
        # Filter out non-numeric values (e.g. string labels from QC checkers)
        numeric_vals = [v for v in vals if isinstance(v, (int, float, np.integer, np.floating))]
        if not numeric_vals:
            continue
        arr = np.array(numeric_vals, dtype=np.float64)
        agg[key] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'count': len(numeric_vals),
        }

    return agg
