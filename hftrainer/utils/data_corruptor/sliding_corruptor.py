"""
Sliding corruptor: 脚步滑动（Foot Sliding）— 根节点速度与下肢步幅不匹配。

参考 scripts/m2m/synth_data/lq_sliding.py：
- 根节点速度缩放 (Root Velocity Mismatch)：水平位移速度缩放，与腿步幅脱节。
- 根轨迹平滑 (Trans Smoothing)：破坏性平滑，抹去急停信号。
- 下肢步幅扭曲 (Leg Stride Warping)：放大/缩小大腿、膝盖相对平均姿态的摆动幅度。
- 下肢抖动 (Lower Body Jitter)：大腿+膝盖+脚踝的低频不稳。
- 陷地/浮空 (Floor Offset)：整体高度小偏移。

每次调用随机决定 intensity、失配模式（root_error / leg_error / compound_error）。
不调用外部 normalize_motion_data，与 dataset 的归一化流程一致。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Any, Dict, List, Optional, Tuple

from .base_corruptor import BaseCorruptor

# -----------------------------------------------------------------------------
# 下肢关节（SMPL：1/2 髋/大腿, 4/5 膝, 7/8 踝）
# -----------------------------------------------------------------------------
LOWER_BODY_IDS = [1, 2, 4, 5, 7, 8]
THIGH_IDS = [1, 2]
KNEE_IDS = [4, 5]

# -----------------------------------------------------------------------------
# 强度 -> 参数范围（与 lq_sliding 一致）
# -----------------------------------------------------------------------------
INTENSITY_SLIDING = {
    "low": {
        "root_scale_range": (0.8, 1.2),
        "stride_scale_range": (0.8, 1.2),
        "jitter_amp": 0.02,
        "trans_sigma": 3.0,
        "floor_offset_range": (-0.03, 0.03),
        "jitter_prob": 0.4,
    },
    "medium": {
        "root_scale_range": (0.6, 1.4),
        "stride_scale_range": (0.7, 1.3),
        "jitter_amp": 0.05,
        "trans_sigma": 5.0,
        "floor_offset_range": (-0.05, 0.05),
        "jitter_prob": 0.5,
    },
    "high": {
        "root_scale_range": (0.5, 1.5),
        "stride_scale_range": (0.5, 1.5),
        "jitter_amp": 0.08,
        "trans_sigma": 8.0,
        "floor_offset_range": (-0.08, 0.08),
        "jitter_prob": 0.6,
    },
}


def _apply_root_velocity_mismatch(trans: np.ndarray, scale: float) -> np.ndarray:
    """缩放根节点水平位移速度，保留垂直分量。"""
    F = trans.shape[0]
    velocity = np.zeros_like(trans, dtype=np.float64)
    velocity[1:] = trans[1:] - trans[:-1]
    velocity[0] = velocity[1]
    horiz_vel = velocity[:, [0, 2]] * scale
    vert_vel = velocity[:, 1]
    trans_mod = np.zeros_like(trans)
    trans_mod[0] = trans[0].copy()
    curr = trans[0].copy()
    for f in range(1, F):
        step = np.array([horiz_vel[f, 0], vert_vel[f], horiz_vel[f, 1]], dtype=np.float64)
        curr = curr + step
        trans_mod[f] = curr
    return trans_mod


def _apply_trans_smoothing(trans: np.ndarray, sigma: float) -> np.ndarray:
    """对水平分量做破坏性平滑，抹去急停。"""
    out = trans.copy()
    out[:, 0] = gaussian_filter1d(trans[:, 0].astype(np.float64), sigma=sigma, mode="nearest")
    out[:, 2] = gaussian_filter1d(trans[:, 2].astype(np.float64), sigma=sigma, mode="nearest")
    return out


def _apply_leg_stride_warping(
    poses: np.ndarray, scale_factor: float, thigh_ids: List[int], knee_ids: List[int], J: int
) -> np.ndarray:
    """步幅扭曲：对大腿、膝盖相对均值的偏差做缩放。"""
    poses_mod = poses.copy()
    F = poses_mod.shape[0]
    target_joints = [j for j in (thigh_ids + knee_ids) if j < J]
    for j in target_joints:
        mean_pose = np.mean(poses_mod[:, j, :], axis=0)
        deviation = poses_mod[:, j, :] - mean_pose
        poses_mod[:, j, :] = mean_pose + deviation * scale_factor
    return poses_mod


def _apply_lower_body_jitter(poses: np.ndarray, noise_scale: float, lower_body_ids: List[int], J: int) -> np.ndarray:
    """下肢低频抖动。"""
    poses_mod = poses.copy()
    F = poses_mod.shape[0]
    for j in lower_body_ids:
        if j >= J:
            continue
        noise = np.random.randn(F, 3).astype(np.float64)
        noise = gaussian_filter1d(noise, sigma=5.0, axis=0, mode="nearest")
        std = np.std(noise) + 1e-9
        noise = noise / std * noise_scale
        poses_mod[:, j, :] += noise
    return poses_mod


class SlidingCorruptor(BaseCorruptor):
    """
    脚步滑动 corruptor：通过根节点速度与下肢步幅的失配制造滑步感。
    修改 poses（下肢）与 trans（根轨迹）。
    """

    def __init__(
        self,
        body_model: Optional[Any] = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(body_model=body_model, device=device)

    def _apply_corruption(
        self,
        data_mod: Dict,
        poses: np.ndarray,
        trans: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        intensity = kwargs.get("intensity") or str(np.random.choice(list(INTENSITY_SLIDING.keys())))
        params = INTENSITY_SLIDING.get(intensity, INTENSITY_SLIDING["medium"]).copy()
        root_scale_range = params["root_scale_range"]
        stride_scale_range = params["stride_scale_range"]
        jitter_amp = params["jitter_amp"]
        trans_sigma = params["trans_sigma"]
        floor_range = params["floor_offset_range"]
        jitter_prob = params["jitter_prob"]

        F, J, _ = poses.shape
        affected = ["root_trajectory", "lower_body_kinematics"]

        mode = str(np.random.choice(["root_error", "leg_error", "compound_error"]))
        act_root_scale = 1.0
        act_stride_scale = 1.0

        if mode == "root_error":
            if np.random.random() < 0.5:
                act_root_scale = float(np.random.uniform(root_scale_range[0], 0.9))
            else:
                act_root_scale = float(np.random.uniform(1.1, root_scale_range[1]))
        elif mode == "leg_error":
            if np.random.random() < 0.5:
                act_stride_scale = float(np.random.uniform(stride_scale_range[0], 0.9))
            else:
                act_stride_scale = float(np.random.uniform(1.1, stride_scale_range[1]))
        else:
            if np.random.random() < 0.5:
                act_root_scale = 1.2
                act_stride_scale = 0.8
            else:
                act_root_scale = 0.8
                act_stride_scale = 1.2

        trans_out = trans.copy()
        if abs(act_root_scale - 1.0) > 0.01:
            trans_out = _apply_root_velocity_mismatch(trans_out, act_root_scale)
            trans_out = _apply_trans_smoothing(trans_out, trans_sigma)

        poses_out = poses.copy()
        if abs(act_stride_scale - 1.0) > 0.01:
            poses_out = _apply_leg_stride_warping(poses_out, act_stride_scale, THIGH_IDS, KNEE_IDS, J)

        if np.random.random() < jitter_prob:
            poses_out = _apply_lower_body_jitter(poses_out, jitter_amp, LOWER_BODY_IDS, J)
            affected.append("leg_jitter")

        floor_offset = float(np.random.uniform(floor_range[0], floor_range[1]))
        trans_out[:, 1] += floor_offset

        # Build _mask_info for joint_corrupted_mask generation
        # Sliding affects lower body joints and root translation on all frames.
        affected_joints = list(LOWER_BODY_IDS)
        _mask_info = {
            "all_frames": True,
            "corrupted_joints": affected_joints,
            "trans_corrupted": True,
        }

        meta = {
            "synthesis_type": "sliding",
            "description": f"Sliding ({intensity}): Root x{act_root_scale:.2f}, Legs x{act_stride_scale:.2f}",
            "synthesis_method": {
                "pattern_type": "kinematic_mismatch",
                "intensity_level": intensity,
                "mode": mode,
                "parameters": {
                    "root_velocity_scale": act_root_scale,
                    "leg_stride_scale": act_stride_scale,
                    "floor_offset": floor_offset,
                },
            },
            "degradation_details": {
                "affected_components": affected,
                "logic": "root_limb_desynchronization",
            },
            "_mask_info": _mask_info,
        }
        return poses_out, trans_out, meta
