#!/usr/bin/env python3
"""Unified M2M evaluation across T1-T8 tasks.

Single entry point for comprehensive evaluation of HyMotion M2M models.

Usage:
    # Smoke test (5 samples, 10 steps)
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_m2m_all_tasks.py \
        --task T1 --models uncond_fm_man --max-samples 5 --num-steps 10

    # Full eval
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_m2m_all_tasks.py \
        --task T1 T2 T3 T4 T5 T6 T7 T8 \
        --models uncond_fm_man uncond_fm_man_globalrot \
        --max-samples 100 --num-steps 50 \
        --output-dir output/eval_results/m2m/ --save-viz

Task → Representative Setting:
    T1: T1-C (首尾各5帧, UMO setting)
    T2: T2-D (每30帧1关键帧, KIMODO setting)
    T3: T3-B (首帧+89帧生成)
    T4: T4-B (90帧循环)
    T5: T5-B (30帧prefix→90帧预测)
    T6: T6-A (下→上补全)
    T7: T7-A (completion mode)
    T8: T8-B (GT轨迹, 120帧)
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Block heavy transitive imports (same pattern as eval_m2m_completion.py)
# ============================================================================
import types as _types
_dummy_modules = [
    'hftrainer.models',
    'hftrainer.models.motion',
    'hftrainer.datasets',
    'hftrainer.datasets.motion',
    'hftrainer.datasets.motion.motionhub',
]
for _mod_name in _dummy_modules:
    if _mod_name not in sys.modules:
        _dummy = _types.ModuleType(_mod_name)
        _dummy.__path__ = [str(PROJECT_ROOT / _mod_name.replace('.', '/'))]
        _dummy.__package__ = _mod_name
        sys.modules[_mod_name] = _dummy

# ============================================================================
# Constants
# ============================================================================
MAX_FRAME = 360
D = 135  # 3 abs transl + 22*6 rot6d

# SMPL-22 kinematic tree
_SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Joint groups
UPPER_BODY_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LOWER_BODY_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]

# ============================================================================
# Model registry — primary uncond variants
# ============================================================================
M2M_CONFIGS = {
    "uncond_fm_man": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b.py",
        "work_dir": "hymotion_m2m_completion_uncond_fm_man_046b",
        "desc": "Uncond FM MAN 0.46B (local rot, epoch 1000)",
        "needs_text": False,
    },
    "uncond_fm_man_globalrot": {
        "config": "configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_globalrot_046b.py",
        "work_dir": "hymotion_m2m_completion_uncond_fm_man_globalrot_046b",
        "desc": "Uncond FM MAN 0.46B (global rot, epoch 527)",
        "needs_text": False,
    },
}

# ============================================================================
# Task definitions
# ============================================================================
TASK_DEFINITIONS = {
    "T1": {
        "name": "Transition",
        "setting": "T1-C",
        "desc": "T1 Transition: head/tail 5 frames kept (UMO setting)",
        "datalist": "eval_transition.json",
        "min_frames": 60,
    },
    "T2": {
        "name": "Keyframe Interpolation",
        "setting": "T2-D",
        "desc": "T2 Keyframe: every 30 frames kept (KIMODO setting)",
        "datalist": "eval_keyframe.json",
        "min_frames": 120,
    },
    "T3": {
        "name": "First-Frame Conditioned",
        "setting": "T3-B",
        "desc": "T3 First-frame + 89 frames generation",
        "datalist": "eval_first_frame_cond.json",
        "min_frames": 30,
    },
    "T4": {
        "name": "Loop Animation",
        "setting": "T4-B",
        "desc": "T4 Loop: first=last frame, 90 frames cycle",
        "datalist": "eval_loop_animation.json",
        "min_frames": 60,
    },
    "T5": {
        "name": "Prediction",
        "setting": "T5-B",
        "desc": "T5 Prediction: 30f prefix → 90f prediction",
        "datalist": "eval_transition.json",
        "min_frames": 120,
    },
    "T6": {
        "name": "Joint Completion",
        "setting": "T6-A",
        "desc": "T6 Joint: lower body kept → upper body generated",
        "datalist": "eval_transition.json",
        "min_frames": 60,
    },
    "T7": {
        "name": "Repair",
        "setting": "T7-A",
        "desc": "T7 Repair: checker-detected defects, completion mode",
        "datalist": "eval_repair_focused.json",
        "min_frames": 30,
    },
    "T8": {
        "name": "Trajectory",
        "setting": "T8-B",
        "desc": "T8 Trajectory: GT root transl kept, 120 frames",
        "datalist": "eval_trajectory.json",
        "min_frames": 60,
    },
    "T9": {
        "name": "Upsample 5fps",
        "setting": "T9-A",
        "desc": "T9: sparse keyframes every 6 frames → 30fps interpolation",
        "datalist": "eval_keyframe.json",
        "min_frames": 60,
    },
    "T10": {
        "name": "Upsample 1fps",
        "setting": "T10-A",
        "desc": "T10: sparse keyframes every 30 frames → 30fps interpolation",
        "datalist": "eval_keyframe.json",
        "min_frames": 120,
    },
    "T11": {
        "name": "Upsample 0.5fps",
        "setting": "T11-A",
        "desc": "T11: sparse keyframes every 60 frames → 30fps interpolation",
        "datalist": "eval_keyframe.json",
        "min_frames": 120,
    },
    "T12": {
        "name": "Upsample Auto",
        "setting": "T12-A",
        "desc": "T12: auto-detected keyposes (angular velocity peaks) → 30fps interpolation",
        "datalist": "eval_keyframe.json",
        "min_frames": 120,
    },
}

# ============================================================================
# Motion utilities (from eval_m2m_completion.py)
# ============================================================================

def _smplh_to_rot6d_22(poses_aa: np.ndarray) -> np.ndarray:
    """Convert SMPL-H axis-angle (T,156) to row-major rot6d (T, 132)."""
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_rotation_6d,
    )
    if poses_aa.ndim == 2:
        n_joints = poses_aa.shape[1] // 3
        if n_joints == 52:
            poses_aa = np.concatenate(
                [poses_aa[:, :66], np.zeros((poses_aa.shape[0], 9), dtype=poses_aa.dtype), poses_aa[:, 66:]],
                axis=1,
            )
        poses_aa = poses_aa.reshape(poses_aa.shape[0], -1, 3)
    aa = poses_aa[:, :22, :]
    T = aa.shape[0]
    aa_flat = aa.reshape(T * 22, 3)
    r6d = axis_angle_to_rotation_6d(aa_flat).reshape(T, 22, 6)
    # column-major -> row-major
    r6d = r6d[:, :, [0, 3, 1, 4, 2, 5]]
    return r6d.reshape(T, 132).astype(np.float32)


def load_npz_as_motion(npz_path: str):
    """Load NPZ -> (T, 135) motion tensor with abs translation."""
    data = dict(np.load(npz_path, allow_pickle=True))
    poses = np.array(data["poses"], dtype=np.float32)
    trans = np.array(data.get("trans", data.get("transl")), dtype=np.float32)
    if trans.ndim == 1:
        trans = trans.reshape(-1, 3)
    fps = int(data.get("mocap_framerate", 30))
    pose_rot6d = _smplh_to_rot6d_22(poses)
    transl_abs = trans.astype(np.float32)
    motion = np.concatenate([transl_abs, pose_rot6d], axis=-1)
    return torch.from_numpy(motion).float(), motion.shape[0], fps, data


def motion_135_to_npz(motion_135, orig_data, output_path, fps=30):
    """Convert (T, 135) back to axis-angle NPZ and save."""
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        rotation_6d_to_axis_angle,
    )
    motion = motion_135.float().numpy()
    T = motion.shape[0]
    abs_transl = motion[:, 0:3]
    rot6d = motion[:, 3:135].reshape(T * 22, 6)
    rot6d_colmajor = rot6d[:, [0, 2, 4, 1, 3, 5]]
    axis_angle = rotation_6d_to_axis_angle(rot6d_colmajor)
    axis_angle = np.array(axis_angle, dtype=np.float32).reshape(T, 22, 3)

    orig_poses = np.array(orig_data.get("poses", np.zeros((T, 156))), dtype=np.float32)
    T_save = min(T, orig_poses.shape[0])
    full_poses = np.zeros((T, orig_poses.shape[1]), dtype=np.float32)
    full_poses[:T_save, :66] = axis_angle[:T_save].reshape(-1, 66)
    if orig_poses.shape[1] > 66:
        full_poses[:T_save, 66:] = orig_poses[:T_save, 66:]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(
        output_path,
        poses=full_poses[:T],
        trans=abs_transl[:T],
        betas=orig_data.get("betas", np.zeros((1, 16), dtype=np.float32)),
        mocap_framerate=fps,
        gender=str(orig_data.get("gender", "neutral")),
        num_frames=T,
    )


# ============================================================================
# Global <-> Local rotation conversion
# ============================================================================

def _local_to_global_rot6d(local_rot6d: torch.Tensor) -> torch.Tensor:
    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        rot6d_to_rotation_matrix, rotation_matrix_to_rot6d,
    )
    local_mat = rot6d_to_rotation_matrix(local_rot6d)
    global_mat = torch.zeros_like(local_mat)
    for j, p in enumerate(_SMPL22_PARENTS):
        if p < 0:
            global_mat[..., j, :, :] = local_mat[..., j, :, :]
        else:
            global_mat[..., j, :, :] = global_mat[..., p, :, :] @ local_mat[..., j, :, :]
    return rotation_matrix_to_rot6d(global_mat)


def _global_to_local_rot6d(global_rot6d: torch.Tensor) -> torch.Tensor:
    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        rot6d_to_rotation_matrix, rotation_matrix_to_rot6d,
    )
    global_mat = rot6d_to_rotation_matrix(global_rot6d)
    local_mat = torch.zeros_like(global_mat)
    for j, p in enumerate(_SMPL22_PARENTS):
        if p < 0:
            local_mat[..., j, :, :] = global_mat[..., j, :, :]
        else:
            local_mat[..., j, :, :] = global_mat[..., p, :, :].transpose(-2, -1) @ global_mat[..., j, :, :]
    return rotation_matrix_to_rot6d(local_mat)


# ============================================================================
# Mask builders for T1-T8
# ============================================================================

def build_mask_T1(T, **kwargs):
    """T1-C: head/tail 5 frames kept, middle generated."""
    head_f = kwargs.get("head_f", 5)
    tail_f = kwargs.get("tail_f", 5)
    mask = torch.ones(T, D)
    mask[:head_f, :] = 0
    mask[-tail_f:, :] = 0
    return mask


def build_mask_T2(T, **kwargs):
    """T2-D: every 30 frames kept as keyframe."""
    interval = kwargs.get("interval", 30)
    mask = torch.ones(T, D)
    keyframes = list(range(0, T, interval))
    if (T - 1) not in keyframes:
        keyframes.append(T - 1)
    for kf in keyframes:
        mask[kf, :] = 0
    return mask


def build_mask_T3(T, **kwargs):
    """T3-B: first frame kept, rest generated."""
    mask = torch.ones(T, D)
    mask[0, :] = 0
    return mask


def build_mask_T4(T, **kwargs):
    """T4-B: first and last frame kept (same pose for loop)."""
    mask = torch.ones(T, D)
    mask[0, :] = 0
    mask[-1, :] = 0
    return mask


def build_mask_T5(T, **kwargs):
    """T5-B: prefix 30 frames kept, rest generated."""
    prefix_len = kwargs.get("prefix_len", 30)
    mask = torch.ones(T, D)
    mask[:prefix_len, :] = 0
    return mask


def build_mask_T6(T, **kwargs):
    """T6-A: lower body + transl kept, upper body generated."""
    mask = torch.zeros(T, D)
    for j in UPPER_BODY_JOINTS:
        start = 3 + j * 6
        end = start + 6
        mask[:, start:end] = 1.0
    return mask


def build_mask_T7(T, **kwargs):
    """T7-A: adaptive mask from MoGenDIT — only mask detected defect regions.

    Reads precomputed adaptive_masks/<motion_path>.npz containing:
      joint_mask: (T_orig, 22) bool — per-joint per-frame defect indicator
      trans_mask: (T_orig,) bool — per-frame translation defect indicator

    Light temporal dilation (±2 frames) for boundary coverage.
    Falls back to full mask if no adaptive mask available.
    """
    adaptive_mask_path = kwargs.get("adaptive_mask_path", None)
    temporal_dilate = kwargs.get("temporal_dilate", 2)

    if adaptive_mask_path and os.path.isfile(adaptive_mask_path):
        data = np.load(adaptive_mask_path, allow_pickle=True)
        joint_mask = data["joint_mask"].astype(np.float32)  # (T_orig, 22)
        trans_mask = data.get("trans_mask", np.zeros(joint_mask.shape[0], dtype=np.float32))
        trans_mask = trans_mask.astype(np.float32)

        T_mask = min(T, joint_mask.shape[0])

        # Build (T, 23) combined: col 0 = trans, cols 1..22 = joints
        combined = np.zeros((T, 23), dtype=np.float32)
        combined[:T_mask, 0] = trans_mask[:T_mask]
        combined[:T_mask, 1:23] = joint_mask[:T_mask, :22]

        # Light temporal dilation for boundary coverage
        if temporal_dilate > 0:
            for col in range(23):
                arr = combined[:, col]
                dilated = arr.copy()
                for _ in range(temporal_dilate):
                    padded = np.pad(dilated, 1, mode='edge')
                    dilated = np.maximum(np.maximum(padded[:-2], padded[2:]), padded[1:-1])
                combined[:, col] = dilated

        # Convert (T, 23) → (T, 135)
        mask = torch.zeros(T, D, dtype=torch.float32)
        trans_col = torch.from_numpy(combined[:, 0])
        mask[:, 0] = trans_col
        mask[:, 1] = trans_col
        mask[:, 2] = trans_col
        for j in range(22):
            start = 3 + j * 6
            joint_col = torch.from_numpy(combined[:, j + 1])
            for d in range(6):
                mask[:, start + d] = joint_col

        return mask
    else:
        # Fallback: full mask (should not happen if masks are precomputed)
        mask = torch.ones(T, D)
        return mask


def build_mask_T8(T, **kwargs):
    """T8-B: root translation kept (first 3 dims), all joints generated."""
    mask = torch.ones(T, D)
    mask[:, :3] = 0  # keep abs translation
    return mask


def _build_upsample_mask(T, interval):
    """Build keyframe mask for upsampling tasks (T9-T11)."""
    mask = torch.ones(T, D)
    keyframes = sorted(set(list(range(0, T, interval)) + [T - 1]))
    for i in keyframes:
        mask[i, :] = 0.0
    return mask, keyframes


def _build_auto_keypose_mask(T, motion_135):
    """Build auto-detected keypose mask for T12."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    # rot6d angular velocity
    rot6d = motion_135[:, 3:135].reshape(T, 22, 6)
    vel = (rot6d[1:] - rot6d[:-1]).norm(dim=-1).sum(dim=-1)  # (T-1,)
    # Smooth
    smoothed = gaussian_filter1d(vel.numpy(), sigma=2)
    # Local maxima (distance=15 → min 0.5s apart at 30fps)
    peaks, _ = find_peaks(smoothed, distance=15)
    keyframes = sorted(set([0, T - 1] + peaks.tolist()))
    # Guarantee at least 3 keyposes
    if len(keyframes) < 3:
        mid = T // 2
        keyframes = sorted(set(keyframes + [mid]))
    mask = torch.ones(T, D)
    for i in keyframes:
        mask[i, :] = 0.0
    return mask, keyframes


def build_mask_T9(T, **kwargs):
    """T9-A: upsample from 5fps (every 6 frames)."""
    mask, kf = _build_upsample_mask(T, interval=6)
    return mask


def build_mask_T10(T, **kwargs):
    """T10-A: upsample from 1fps (every 30 frames)."""
    mask, kf = _build_upsample_mask(T, interval=30)
    return mask


def build_mask_T11(T, **kwargs):
    """T11-A: upsample from 0.5fps (every 60 frames)."""
    mask, kf = _build_upsample_mask(T, interval=60)
    return mask


def build_mask_T12(T, **kwargs):
    """T12-A: upsample from auto-detected keyposes."""
    motion_135 = kwargs.get("motion_135", None)
    if motion_135 is None:
        # Fallback to interval-based
        mask, kf = _build_upsample_mask(T, interval=30)
        return mask
    mask, kf = _build_auto_keypose_mask(T, motion_135)
    return mask


def _get_keyframes_for_task(task_id, T, motion_135=None):
    """Return keyframe indices for upsample tasks (T9-T12)."""
    if task_id == "T9":
        return sorted(set(list(range(0, T, 6)) + [T - 1]))
    elif task_id == "T10":
        return sorted(set(list(range(0, T, 30)) + [T - 1]))
    elif task_id == "T11":
        return sorted(set(list(range(0, T, 60)) + [T - 1]))
    elif task_id == "T12" and motion_135 is not None:
        _, kf = _build_auto_keypose_mask(T, motion_135)
        return kf
    return []


MASK_BUILDERS = {
    "T1": build_mask_T1,
    "T2": build_mask_T2,
    "T3": build_mask_T3,
    "T4": build_mask_T4,
    "T5": build_mask_T5,
    "T6": build_mask_T6,
    "T7": build_mask_T7,
    "T8": build_mask_T8,
    "T9": build_mask_T9,
    "T10": build_mask_T10,
    "T11": build_mask_T11,
    "T12": build_mask_T12,
}


# ============================================================================
# Metrics
# ============================================================================

def compute_mpjpe(pred_135, gt_135, mask_135=None):
    """Compute translation error and rotation error, both overall and masked-region."""
    if pred_135.shape[0] != gt_135.shape[0]:
        T = min(pred_135.shape[0], gt_135.shape[0])
        pred_135 = pred_135[:T]
        gt_135 = gt_135[:T]
        if mask_135 is not None:
            mask_135 = mask_135[:T]

    trans_err = (pred_135[:, :3] - gt_135[:, :3]).norm(dim=-1)
    rot_diff = (pred_135[:, 3:] - gt_135[:, 3:]).reshape(-1, 22, 6)
    rot_err_per_joint = rot_diff.norm(dim=-1)

    result = {
        "trans_err_mm": round(trans_err.mean().item() * 1000, 2),
        "rot_err": round(rot_err_per_joint.mean().item(), 6),
    }

    if mask_135 is not None:
        frame_mask = mask_135.mean(dim=-1) > 0  # frames with any mask
        if frame_mask.sum() > 0:
            result["masked_trans_err_mm"] = round(trans_err[frame_mask].mean().item() * 1000, 2)
            result["masked_rot_err"] = round(rot_err_per_joint[frame_mask].mean().item(), 6)
        # Also compute per-dim masked error for joint-level tasks
        rot_mask = mask_135[:, 3:].reshape(-1, 22, 6).mean(dim=-1) > 0.5  # (T, 22) bool
        if rot_mask.any():
            result["masked_joint_rot_err"] = round(rot_err_per_joint[rot_mask].mean().item(), 6)

    return result


def compute_boundary_smoothness(pred_135, mask_135, window=3):
    """Compute boundary smoothness at mask transitions."""
    frame_mask = mask_135.mean(dim=-1)
    T = frame_mask.shape[0]
    boundaries = []
    for t in range(1, T):
        if (frame_mask[t] > 0.5) != (frame_mask[t - 1] > 0.5):
            boundaries.append(t)
    if not boundaries:
        return {"boundary_jerk": 0.0, "num_boundaries": 0}
    jerks = []
    for b in boundaries:
        lo = max(0, b - window)
        hi = min(T, b + window)
        if hi - lo < 3:
            continue
        segment = pred_135[lo:hi]
        vel = segment[1:] - segment[:-1]
        acc = vel[1:] - vel[:-1]
        jerk = acc.norm(dim=-1).mean().item()
        jerks.append(jerk)
    return {
        "boundary_jerk": round(np.mean(jerks) if jerks else 0.0, 6),
        "num_boundaries": len(boundaries),
    }


def compute_jitter(pred_135, fps=30):
    """Compute jitter (acceleration magnitude)."""
    if pred_135.shape[0] < 3:
        return {"jitter": 0.0}
    vel = (pred_135[1:] - pred_135[:-1]) * fps
    acc = (vel[1:] - vel[:-1]) * fps
    jitter = acc.norm(dim=-1).mean().item()
    return {"jitter": round(jitter, 4)}


def compute_foot_skating(pred_135, fps=30):
    """Root speed as skating proxy."""
    trans = pred_135[:, :3]
    if trans.shape[0] < 2:
        return {"root_speed_mean": 0.0}
    vel = (trans[1:] - trans[:-1]) * fps
    speed = vel.norm(dim=-1)
    return {"root_speed_mean": round(speed.mean().item(), 4)}


def compute_trajectory_error(pred_135, gt_135):
    """T8-specific: root translation accuracy."""
    pred_trans = pred_135[:, :3]
    gt_trans = gt_135[:, :3]
    T = min(pred_trans.shape[0], gt_trans.shape[0])
    pred_trans = pred_trans[:T]
    gt_trans = gt_trans[:T]
    ade = (pred_trans - gt_trans).norm(dim=-1).mean().item()
    fde = (pred_trans[-1] - gt_trans[-1]).norm().item()
    return {
        "traj_ade_mm": round(ade * 1000, 2),
        "traj_fde_mm": round(fde * 1000, 2),
    }


def compute_loop_continuity(pred_135):
    """T4-specific: first-last frame difference."""
    diff = (pred_135[0] - pred_135[-1]).norm().item()
    return {"loop_cont_err": round(diff, 6)}


def compute_keyframe_accuracy(pred_135, gt_135, keyframe_indices):
    """Compute keyframe reconstruction accuracy for upsample tasks (T9-T12)."""
    if not keyframe_indices:
        return {}
    T = min(pred_135.shape[0], gt_135.shape[0])
    kf = [i for i in keyframe_indices if i < T]
    if not kf:
        return {}
    pred_kf = pred_135[kf]
    gt_kf = gt_135[kf]
    trans_err = (pred_kf[:, :3] - gt_kf[:, :3]).norm(dim=-1).mean().item()
    rot_diff = (pred_kf[:, 3:] - gt_kf[:, 3:]).reshape(-1, 22, 6)
    rot_err = rot_diff.norm(dim=-1).mean().item()
    return {
        "kf_trans_err_mm": round(trans_err * 1000, 2),
        "kf_rot_err": round(rot_err, 6),
    }


def compute_all_metrics(task_id, pred_135, gt_135, mask_135, fps=30, keyframe_indices=None):
    """Compute all relevant metrics for a task."""
    m = {}
    m.update(compute_mpjpe(pred_135, gt_135, mask_135))
    m.update(compute_jitter(pred_135, fps))

    # Task-specific metrics
    if task_id in ("T1", "T2", "T4", "T5", "T6", "T9", "T10", "T11", "T12"):
        m.update(compute_boundary_smoothness(pred_135, mask_135))
    if task_id == "T4":
        m.update(compute_loop_continuity(pred_135))
    if task_id == "T8":
        m.update(compute_trajectory_error(pred_135, gt_135))
    if task_id in ("T9", "T10", "T11", "T12") and keyframe_indices:
        m.update(compute_keyframe_accuracy(pred_135, gt_135, keyframe_indices))

    return m


# ============================================================================
# Model building (from eval_m2m_completion.py)
# ============================================================================

def find_latest_checkpoint(work_dir_name):
    work_dir = PROJECT_ROOT / "work_dirs" / work_dir_name
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")
    ckpt_dirs = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: d.stat().st_mtime,
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {work_dir}")
    return str(ckpt_dirs[-1])


def _extract_epoch(ckpt_path):
    """Extract epoch number from checkpoint path like '.../checkpoint-epoch_527'."""
    import re
    m = re.search(r'checkpoint-epoch_(\d+)', str(ckpt_path))
    return int(m.group(1)) if m else None


def find_training_config(checkpoint_path):
    work_dir = Path(checkpoint_path).parent
    run_dirs = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and d.name[:4].isdigit()],
        key=lambda d: d.name,
    )
    for rd in reversed(run_dirs):
        cfg_path = rd / "config.py"
        if cfg_path.is_file():
            return str(cfg_path)
    return None


def build_m2m_model(config_name, device, num_steps):
    """Build M2M bundle + pipeline for a given config."""
    from mmengine.config import Config
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    info = M2M_CONFIGS[config_name]
    ckpt_path = find_latest_checkpoint(info["work_dir"])
    print(f"  [M2M] {config_name}: ckpt={ckpt_path}")

    training_config = find_training_config(ckpt_path)
    source_config = str(PROJECT_ROOT / info["config"])
    config_path = training_config or source_config

    cfg = Config.fromfile(config_path)
    bundle = HyMotionM2MBundle.from_config(cfg.model)
    bundle = bundle.to(device)
    bundle.eval()

    model_pt_path = os.path.join(ckpt_path, "model.pt")
    raw = torch.load(model_pt_path, map_location=device, weights_only=False)
    transformer_sd = raw["motion_transformer"]
    prefixed_sd = {f"motion_transformer.{k}": v for k, v in transformer_sd.items()}

    bundle_params = raw.get("__bundle_params__", {})
    if bundle_params:
        for pname, pval in bundle_params.items():
            if hasattr(bundle, pname):
                attr = getattr(bundle, pname)
                if isinstance(attr, torch.nn.Parameter):
                    attr.data.copy_(pval.to(device))
                elif isinstance(attr, torch.Tensor):
                    attr.copy_(pval.to(device))

    missing, unexpected = bundle.load_state_dict(prefixed_sd, strict=False)

    # Fallback for null embeddings
    if "null_vtxt_feat" in missing and not bundle_params:
        t2m_path = "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt"
        if os.path.exists(t2m_path):
            t2m = torch.load(t2m_path, map_location=device, weights_only=False)
            t2m_sd = t2m.get("model_state_dict", t2m)
            if "null_vtxt_feat" in t2m_sd:
                bundle.null_vtxt_feat.data.copy_(t2m_sd["null_vtxt_feat"].to(device))
                bundle.null_ctxt_input.data.copy_(t2m_sd["null_ctxt_input"].to(device))
            del t2m

    replacement = "skip_last"
    pipeline = HyMotionM2MPipeline(bundle, num_steps=num_steps, replacement_guidance=replacement)
    return pipeline, bundle, ckpt_path


# ============================================================================
# Inference core
# ============================================================================

def run_completion(pipeline, bundle, motion_135, mask_135, device,
                   max_frames=MAX_FRAME, task_id=None):
    """Run M2M completion. Returns (combined_motion, raw_output)."""
    T_orig = motion_135.shape[0]
    T = min(T_orig, max_frames)
    motion_in = motion_135[:T].clone()

    is_global = getattr(bundle, 'rotation_space', 'local') == 'global'
    if is_global:
        trans = motion_in[:, :3]
        rot6d_local = motion_in[:, 3:].reshape(T, 22, 6)
        rot6d_global = _local_to_global_rot6d(rot6d_local)
        motion_in = torch.cat([trans, rot6d_global.reshape(T, 132)], dim=-1)

    # For T4 loop: set last frame = first frame in src_motion
    if task_id == "T4":
        motion_in[-1] = motion_in[0].clone()

    motion_norm = bundle.normalize_motion(motion_in.unsqueeze(0).to(device))
    msk = mask_135[:T].unsqueeze(0).to(device)

    # Keep clean copy for replacement guidance, then zero masked regions
    clean_motion = motion_norm.clone()
    motion_norm = motion_norm * (1 - msk)

    if T < max_frames:
        pad_len = max_frames - T
        motion_norm = torch.nn.functional.pad(motion_norm, (0, 0, 0, pad_len), value=0)
        clean_motion = torch.nn.functional.pad(clean_motion, (0, 0, 0, pad_len), value=0)
        msk = torch.nn.functional.pad(msk, (0, 0, 0, pad_len), value=0)

    batch = {
        "src_motion": motion_norm,
        "src_mask": msk,
        "clean_motion": clean_motion,
        "src_length": [T],
        "tgt_length": [T],
    }

    # No per-task replacement override needed — skip_last from pipeline
    # construction works well for all tasks including T7 repair.

    with torch.no_grad():
        result = pipeline(batch)

    repaired_latent = result["latent"][0, :T].cpu()
    repaired_raw = bundle.denormalize_motion(repaired_latent.unsqueeze(0).to(device))[0].cpu()

    # For T4: use modified reference (last=first) for combine step
    ref_motion = motion_135[:T].clone()
    if task_id == "T4":
        ref_motion[-1] = ref_motion[0].clone()

    if is_global:
        r_rot6d_global = repaired_raw[:, 3:].reshape(T, 22, 6)
        r_rot6d_local = _global_to_local_rot6d(r_rot6d_global)
        repaired_raw = torch.cat([repaired_raw[:, :3], r_rot6d_local.reshape(T, 132)], dim=-1)

    # No hard blend — skip_last imputation already preserves known regions
    # during ODE integration. Hard blend introduces discontinuities at mask
    # boundaries (the #1 source of joint_jump artifacts in T7 repair).
    combined = repaired_raw

    return combined, repaired_raw


# ============================================================================
# Data loading
# ============================================================================

DATA_ROOT = PROJECT_ROOT / "data" / "hymotion_data"
# Some datalists use "Game/..." under 3D/20251111/motions/
MOTION_ROOTS = [
    DATA_ROOT / "3D" / "20251111" / "motions",
    DATA_ROOT,
]
EVAL_DIR = PROJECT_ROOT / "data" / "eval" / "hymotion_m2m"


def resolve_motion_path(motion_path):
    """Try multiple roots to find the motion file."""
    for root in MOTION_ROOTS:
        full = str(root / motion_path)
        if os.path.isfile(full):
            return full
    return None


def load_eval_data(datalist_name, max_samples=100, min_frames=30, seed=42):
    """Load test samples from eval datalist."""
    datalist_path = EVAL_DIR / datalist_name
    with open(str(datalist_path)) as f:
        data = json.load(f)

    items = data["data_list"]
    items = [it for it in items if it.get("num_frames", 999) >= min_frames]

    rng = np.random.RandomState(seed)
    rng.shuffle(items)
    items = items[:max_samples]

    valid = []
    for it in items:
        mp = it.get("motion_path", it.get("path", ""))
        full_path = resolve_motion_path(mp)
        if full_path:
            it["full_path"] = full_path
            it["motion_path"] = mp
            valid.append(it)

    print(f"[DATA] {datalist_name}: {len(valid)}/{max_samples} valid (min_frames={min_frames})")
    return valid


# ============================================================================
# Evaluation core
# ============================================================================

def run_task(task_id, task_def, samples, config_name, pipeline, bundle,
             device, output_dir, num_steps, save_viz=False, ckpt_path=""):
    """Run one task for one config on all samples."""
    metrics_list = []
    errors = 0
    mask_builder = MASK_BUILDERS[task_id]

    for idx, sample in enumerate(samples):
        case_id = f"case_{idx:04d}"
        case_dir = os.path.join(output_dir, task_id, config_name, case_id)
        meta_path = os.path.join(case_dir, "meta.json")

        # Skip if already done
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    existing = json.load(f)
                if existing.get("metrics"):
                    metrics_list.append(existing["metrics"])
                continue
            except Exception:
                pass

        try:
            motion_135, num_frames, fps, orig_data = load_npz_as_motion(sample["full_path"])

            # Task-specific frame constraints
            if task_id == "T4":
                T = min(num_frames, 90)  # T4-B: 90 frames
            elif task_id == "T5":
                T = min(num_frames, 120)  # T5-B: 120 frames
            elif task_id == "T3":
                T = min(num_frames, 90)  # T3-B: 90 frames
            elif task_id == "T8":
                T = min(num_frames, 120)  # T8-B: 120 frames
            else:
                T = min(num_frames, MAX_FRAME)

            motion_135 = motion_135[:T]

            # Build mask
            mask_kwargs = {}
            if task_id == "T7":
                # Use precomputed MoGenDIT adaptive mask (permanent location)
                adaptive_mask_dir = os.path.join(
                    str(PROJECT_ROOT), "data", "eval", "hymotion_m2m", "adaptive_masks_mogendit")
                mp = sample.get("motion_path", "")
                adaptive_mask_path = os.path.join(adaptive_mask_dir, mp)
                mask_kwargs["adaptive_mask_path"] = adaptive_mask_path
            if task_id == "T12":
                mask_kwargs["motion_135"] = motion_135
            mask = mask_builder(T, **mask_kwargs)

            # Get keyframe indices for upsample tasks
            keyframe_indices = None
            if task_id in ("T9", "T10", "T11", "T12"):
                keyframe_indices = _get_keyframes_for_task(task_id, T, motion_135)

            # Run completion
            combined, raw_output = run_completion(
                pipeline, bundle, motion_135, mask, device, task_id=task_id)

            # Metrics
            m = compute_all_metrics(task_id, combined, motion_135, mask, fps,
                                    keyframe_indices=keyframe_indices)

            # Save results
            os.makedirs(case_dir, exist_ok=True)

            if save_viz:
                motion_135_to_npz(combined, orig_data,
                                  os.path.join(case_dir, "output.npz"), fps)
                if not os.path.isfile(os.path.join(case_dir, "gt.npz")):
                    motion_135_to_npz(motion_135, orig_data,
                                      os.path.join(case_dir, "gt.npz"), fps)

            meta = {
                "task": task_id,
                "setting": task_def["setting"],
                "config": config_name,
                "epoch": _extract_epoch(ckpt_path),
                "motion_path": sample.get("motion_path", ""),
                "num_frames": T,
                "fps": fps,
                "num_steps": num_steps,
                "mask_ratio": float(mask.mean()),
                "metrics": m,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if keyframe_indices is not None:
                meta["keyframe_indices"] = keyframe_indices
                meta["keyframe_ratio"] = round(len(keyframe_indices) / T, 4)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

            metrics_list.append(m)

            if (idx + 1) % 20 == 0:
                print(f"    [{config_name}] {task_id}: {idx + 1}/{len(samples)} done")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    ERROR case {idx}: {e}")
                traceback.print_exc()

    return metrics_list, errors


def aggregate_metrics(metrics_list):
    """Compute mean/std of metrics across samples."""
    if not metrics_list:
        return {}
    keys = set()
    for m in metrics_list:
        keys.update(m.keys())
    agg = {}
    for k in sorted(keys):
        vals = [m[k] for m in metrics_list if k in m and isinstance(m[k], (int, float))]
        if vals:
            agg[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "n": len(vals),
            }
    return agg


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="M2M All-Tasks Evaluation")
    parser.add_argument("--task", nargs="+", default=["T1"],
                        choices=list(TASK_DEFINITIONS.keys()),
                        help="Tasks to evaluate (default: T1)")
    parser.add_argument("--models", nargs="+", default=["uncond_fm_man"],
                        help="Model configs to evaluate")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="../output/eval_results/m2m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-viz", action="store_true",
                        help="Save output NPZ files for visualization")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = os.path.join(str(PROJECT_ROOT), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tasks = args.task
    configs = args.models

    # Validate configs
    valid_configs = []
    for c in configs:
        if c not in M2M_CONFIGS:
            print(f"WARN: unknown config '{c}', available: {list(M2M_CONFIGS.keys())}")
            continue
        try:
            find_latest_checkpoint(M2M_CONFIGS[c]["work_dir"])
            valid_configs.append(c)
        except FileNotFoundError as e:
            print(f"WARN: {c}: {e}, skipping")
    configs = valid_configs

    if not configs:
        print("ERROR: No valid model configs found!")
        return

    print(f"\n{'=' * 70}")
    print(f"M2M Comprehensive Evaluation (T1-T12)")
    print(f"  Output:   {output_dir}")
    print(f"  Tasks:    {tasks}")
    print(f"  Models:   {configs}")
    print(f"  Samples:  {args.max_samples}")
    print(f"  Steps:    {args.num_steps}")
    print(f"  Device:   {args.device}")
    print(f"  Save viz: {args.save_viz}")
    print(f"{'=' * 70}")

    device = torch.device(args.device)
    all_results = {}
    total_start = time.time()

    for config_name in configs:
        print(f"\n{'=' * 60}")
        print(f"Loading model: {config_name} ({M2M_CONFIGS[config_name]['desc']})")
        print(f"{'=' * 60}")

        try:
            pipeline, bundle, ckpt_path = build_m2m_model(config_name, device, args.num_steps)
        except Exception as e:
            print(f"ERROR loading {config_name}: {e}")
            traceback.print_exc()
            continue

        config_results = {}
        for task_id in tasks:
            task_def = TASK_DEFINITIONS[task_id]

            # Load data for this task
            samples = load_eval_data(
                task_def["datalist"],
                args.max_samples,
                min_frames=task_def["min_frames"],
                seed=args.seed,
            )
            if not samples:
                print(f"  No valid samples for {task_id}")
                continue

            print(f"\n  --- {task_id} ({task_def['setting']}): {len(samples)} samples ---")
            print(f"      {task_def['desc']}")
            t0 = time.time()

            metrics_list, errors = run_task(
                task_id, task_def, samples, config_name,
                pipeline, bundle, device, output_dir, args.num_steps,
                save_viz=args.save_viz, ckpt_path=ckpt_path,
            )

            elapsed = time.time() - t0
            agg = aggregate_metrics(metrics_list)
            config_results[task_id] = {
                "setting": task_def["setting"],
                "aggregated": agg,
                "num_samples": len(metrics_list),
                "num_errors": errors,
                "elapsed_sec": round(elapsed, 1),
            }

            print(f"    Completed in {elapsed:.1f}s ({len(metrics_list)} ok, {errors} errors)")
            for k, v in agg.items():
                print(f"    {k}: {v['mean']:.4f} ± {v['std']:.4f}")

        all_results[config_name] = {
            "desc": M2M_CONFIGS[config_name]["desc"],
            "checkpoint": ckpt_path if 'ckpt_path' in dir() else "",
            "epoch": _extract_epoch(ckpt_path) if 'ckpt_path' in dir() else None,
            "tasks": config_results,
        }

        del pipeline, bundle
        torch.cuda.empty_cache()

    # Save summary
    total_elapsed = time.time() - total_start
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": round(total_elapsed, 1),
        "num_steps": args.num_steps,
        "max_samples": args.max_samples,
        "tasks": tasks,
        "models": configs,
        "results": all_results,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 70}")
    print(f"Evaluation complete! Total: {total_elapsed:.1f}s")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 70}")

    # Print summary table
    print(f"\n{'=' * 90}")
    print("SUMMARY TABLE: model × task")
    print(f"{'=' * 90}")
    header = f"{'Model':<30}"
    for t in tasks:
        header += f" {t:>10}"
    print(header)
    print("-" * 90)

    for config_name in configs:
        cr = all_results.get(config_name, {})
        row = f"{config_name:<30}"
        for t in tasks:
            tr = cr.get("tasks", {}).get(t, {})
            agg = tr.get("aggregated", {})
            te = agg.get("trans_err_mm", {})
            val = te.get("mean", "N/A")
            if isinstance(val, float):
                row += f" {val:>10.1f}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    print(f"\n(Values shown: trans_err_mm mean)")


if __name__ == "__main__":
    main()
