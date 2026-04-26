"""
Foot sliding checker: detects foot sliding artifacts.

Foot sliding occurs when foot/toe joints are in contact with the ground (low height)
but still have significant horizontal velocity -- the feet "slide" along the floor
instead of being planted.

Rule:
  - FK -> world-space joint positions for feet and toes.
  - For each frame, if a foot/toe joint is near ground (Y < threshold), check its
    horizontal (XZ) velocity. If the horizontal velocity exceeds a threshold for
    several consecutive frames, flag as foot sliding.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .base_checker import BaseQualityChecker, CheckResult, NUM_BODY_JOINTS_DEFAULT, normalize_betas_array
from .root_motion_utils import (
    apply_inverse_root_rotation,
    root_angular_velocity_deg_per_frame,
    root_rotation_matrices_from_poses,
)

from ._model_compat import SmplxLiteJ24

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_BODY_JOINTS = 22
# Foot and toe joint indices
FOOT_JOINT_INDICES = [7, 8, 10, 11]  # LFoot, RFoot, LToeBase, RToeBase
FOOT_JOINT_NAMES = ["LFoot", "RFoot", "LToeBase", "RToeBase"]
FOOT_SIDE_GROUPS = [
    {
        "side": "left",
        "indices": [0, 2],
        "joint_ids": [7, 10],
        "joint_names": ["LFoot", "LToeBase"],
    },
    {
        "side": "right",
        "indices": [1, 3],
        "joint_ids": [8, 11],
        "joint_names": ["RFoot", "RToeBase"],
    },
]
SIDE_REPAIR_JOINT_IDS = {
    "left": [0, 1, 4, 7, 10],
    "right": [0, 2, 5, 8, 11],
}
SIDE_JOINT_GROUPS = {
    "left": {"hip": 1, "knee": 4, "foot": 7, "toe": 10},
    "right": {"hip": 2, "knee": 5, "foot": 8, "toe": 11},
}

# A joint is "on the ground" if its Y-coordinate is below this threshold (meters).
# Uses a small margin above 0 to account for ground contact.
DEFAULT_CONTACT_HEIGHT_MARGIN_M = 0.02
# Contact should also look vertically stable; otherwise this is a foot lift, not a plant.
DEFAULT_CONTACT_VERTICAL_VELOCITY_M_PER_FRAME = 0.004
# Horizontal velocity (m/frame) above this while on ground = sliding.
# 0.008 m/frame = 0.24 m/s at 30fps. This is small enough to catch subtle
# planted-foot drift, while still ignoring tiny contact noise.
DEFAULT_SLIDING_VELOCITY_M_PER_FRAME = 0.008
# Minimum number of consecutive sliding frames to flag.
DEFAULT_MIN_SLIDING_FRAMES = 4
# Minimum total sliding frames to consider the motion invalid.
DEFAULT_MIN_TOTAL_SLIDING_FRAMES = 12
# A single sustained slide should also be enough even if it happens only once.
DEFAULT_STRONG_SEGMENT_FRAMES = 8
# Contact masks often flicker for 1-2 frames because toe and foot alternate contact.
DEFAULT_MAX_CONTACT_GAP_FRAMES = 2
DEFAULT_MIN_DIRECTIONAL_CONSISTENCY = 0.75
DEFAULT_MIN_NET_DISPLACEMENT_M = 0.035
DEFAULT_MAX_SLIDING_TURNS = 1
DEFAULT_TRANSLATION_ONLY_LOCAL_VEL_M_PER_FRAME = 0.006
DEFAULT_LEG_CHAIN_LOCAL_VEL_M_PER_FRAME = 0.010
DEFAULT_MIN_CONTACT_SUPPORT_MEAN = 1.35
DEFAULT_MIN_DOUBLE_CONTACT_RATIO = 0.35


def _get_joints_from_pose(
    poses: np.ndarray,
    trans: np.ndarray,
    betas: Optional[np.ndarray] = None,
    body_model: Optional[Any] = None,
    device: str = "cpu",
) -> np.ndarray:
    """FK: world-space joint positions. Returns (F, 24, 3)."""
    if body_model is None or SmplxLiteJ24 is None:
        raise RuntimeError("FootSlidingChecker requires body_model (SmplxLiteJ24) for FK.")
    F = poses.shape[0]
    poses_t = torch.as_tensor(poses, dtype=torch.float32, device=device)
    trans_t = torch.as_tensor(trans, dtype=torch.float32, device=device)
    if betas is None:
        betas_t = torch.zeros((1, 16), dtype=torch.float32, device=device)
    else:
        betas_t = torch.as_tensor(betas, dtype=torch.float32, device=device)
        if betas_t.ndim == 1:
            betas_t = betas_t.unsqueeze(0)
    global_orient = poses_t[:, 0, :]
    body_pose = poses_t[:, 1:22, :].reshape(F, 63)
    with torch.no_grad():
        joints = body_model(
            body_pose=body_pose,
            betas=betas_t,
            global_orient=global_orient,
            transl=trans_t,
            rotation_mode="aa",
        )
    return joints.cpu().numpy()


class FootSlidingChecker(BaseQualityChecker):
    """Detects foot sliding: feet near ground with significant horizontal velocity."""

    name = "foot_sliding"

    def __init__(
        self,
        body_model: Optional[Any] = None,
        device: str = "cuda",
        contact_height_margin_m: float = DEFAULT_CONTACT_HEIGHT_MARGIN_M,
        contact_vertical_velocity_m_per_frame: float = DEFAULT_CONTACT_VERTICAL_VELOCITY_M_PER_FRAME,
        sliding_velocity_m_per_frame: float = DEFAULT_SLIDING_VELOCITY_M_PER_FRAME,
        min_sliding_frames: int = DEFAULT_MIN_SLIDING_FRAMES,
        min_total_sliding_frames: int = DEFAULT_MIN_TOTAL_SLIDING_FRAMES,
        strong_segment_frames: int = DEFAULT_STRONG_SEGMENT_FRAMES,
        max_contact_gap_frames: int = DEFAULT_MAX_CONTACT_GAP_FRAMES,
        min_directional_consistency: float = DEFAULT_MIN_DIRECTIONAL_CONSISTENCY,
        min_net_displacement_m: float = DEFAULT_MIN_NET_DISPLACEMENT_M,
        max_sliding_turns: int = DEFAULT_MAX_SLIDING_TURNS,
        translation_only_local_vel_m_per_frame: float = DEFAULT_TRANSLATION_ONLY_LOCAL_VEL_M_PER_FRAME,
        leg_chain_local_vel_m_per_frame: float = DEFAULT_LEG_CHAIN_LOCAL_VEL_M_PER_FRAME,
        min_contact_support_mean: float = DEFAULT_MIN_CONTACT_SUPPORT_MEAN,
        min_double_contact_ratio: float = DEFAULT_MIN_DOUBLE_CONTACT_RATIO,
    ) -> None:
        super().__init__(body_model=body_model, device=device)
        if self.body_model is None and SmplxLiteJ24 is not None:
            self.body_model = SmplxLiteJ24(gender="neutral").to(self.device)
            self.body_model.eval()
        self.contact_height_margin_m = contact_height_margin_m
        self.contact_vertical_velocity_m_per_frame = contact_vertical_velocity_m_per_frame
        self.sliding_velocity_m_per_frame = sliding_velocity_m_per_frame
        self.min_sliding_frames = min_sliding_frames
        self.min_total_sliding_frames = min_total_sliding_frames
        self.strong_segment_frames = strong_segment_frames
        self.max_contact_gap_frames = max_contact_gap_frames
        self.min_directional_consistency = min_directional_consistency
        self.min_net_displacement_m = min_net_displacement_m
        self.max_sliding_turns = max_sliding_turns
        self.translation_only_local_vel_m_per_frame = translation_only_local_vel_m_per_frame
        self.leg_chain_local_vel_m_per_frame = leg_chain_local_vel_m_per_frame
        self.min_contact_support_mean = min_contact_support_mean
        self.min_double_contact_ratio = min_double_contact_ratio

    def get_required_keys(self) -> list:
        return ["poses", "trans"]

    @staticmethod
    def _close_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
        closed = np.asarray(mask, dtype=bool).copy()
        if max_gap_frames <= 0 or closed.size == 0:
            return closed
        start = 0
        n = closed.size
        while start < n:
            if closed[start]:
                start += 1
                continue
            end = start
            while end < n and not closed[end]:
                end += 1
            if start > 0 and end < n and (end - start) <= max_gap_frames:
                closed[start:end] = True
            start = end
        return closed

    @staticmethod
    def _find_true_segments(mask: np.ndarray, min_length: int) -> List[Dict[str, int]]:
        segments: List[Dict[str, int]] = []
        start = None
        for idx, value in enumerate(mask):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                if idx - start >= min_length:
                    segments.append({"start": start, "end": idx, "num_frames": idx - start})
                start = None
        if start is not None and len(mask) - start >= min_length:
            segments.append({"start": start, "end": len(mask), "num_frames": len(mask) - start})
        return segments

    @staticmethod
    def _trajectory_metrics(points_2d: np.ndarray) -> Dict[str, float]:
        if points_2d.shape[0] <= 1:
            return {
                "path_length_m": 0.0,
                "net_displacement_m": 0.0,
                "directional_consistency": 0.0,
                "turn_count": 0,
            }
        step = np.diff(points_2d, axis=0)
        step_norm = np.linalg.norm(step, axis=1)
        path_length = float(np.sum(step_norm))
        net_vec = points_2d[-1] - points_2d[0]
        net_disp = float(np.linalg.norm(net_vec))
        directional_consistency = float(net_disp / max(path_length, 1e-8))
        if net_disp > 1e-8:
            dominant = net_vec / net_disp
        else:
            mean_step = np.mean(step, axis=0)
            mean_norm = float(np.linalg.norm(mean_step))
            dominant = mean_step / max(mean_norm, 1e-8)
        signed_step = step @ dominant
        eps = max(1e-4, 0.1 * float(np.percentile(np.abs(signed_step), 75)) if signed_step.size else 1e-4)
        signs = np.where(np.abs(signed_step) >= eps, np.where(signed_step > 0.0, 1, -1), 0)
        non_zero = signs[signs != 0]
        turn_count = int(np.sum(non_zero[1:] != non_zero[:-1])) if non_zero.size >= 2 else 0
        return {
            "path_length_m": path_length,
            "net_displacement_m": net_disp,
            "directional_consistency": directional_consistency,
            "turn_count": turn_count,
        }

    def check(self, motion: Union[Dict, str, Path]) -> CheckResult:
        if isinstance(motion, (str, Path)):
            data = self.load_motion(motion)
        else:
            data = dict(motion)
            if "transl" in data and "trans" not in data:
                data["trans"] = data["transl"]

        err = self.validate_motion_dict(data)
        if err is not None:
            return CheckResult(
                is_valid=False,
                invalid_reason=err,
                invalid_mask=None,
                details={"has_sliding": False, "reason": err},
            )

        poses = np.array(data["poses"])
        trans = np.array(data["trans"])
        if len(poses) < 3:
            reason = "Too short (need at least 3 frames)"
            return CheckResult(
                is_valid=True,
                invalid_reason=reason,
                invalid_mask=None,
                details={"has_sliding": False, "reason": reason},
            )

        try:
            poses_3d = np.asarray(data.get("_cached_poses_22")) if data.get("_cached_poses_22") is not None else None
            if poses_3d is None:
                poses_3d = self.normalize_poses(poses, NUM_BODY_JOINTS)
        except ValueError as e:
            return CheckResult(
                is_valid=False,
                invalid_reason=str(e),
                invalid_mask=None,
                details={"has_sliding": False, "reason": str(e)},
            )

        try:
            joints = np.asarray(data.get("_cached_joints_22")) if data.get("_cached_joints_22") is not None else None
            if joints is None:
                joints = _get_joints_from_pose(
                    poses_3d,
                    trans,
                    normalize_betas_array(data.get("betas")),
                    body_model=self.body_model,
                    device=self.device,
                )[:, :NUM_BODY_JOINTS, :]
        except Exception as e:
            return CheckResult(
                is_valid=False,
                invalid_reason=f"FK failed: {e}",
                invalid_mask=None,
                details={"has_sliding": False, "reason": f"FK failed: {str(e)}"},
            )
        F = joints.shape[0]

        foot_joints = joints[:, FOOT_JOINT_INDICES, :]  # (F, 4, 3)
        root_rot_mats = (
            np.asarray(data.get("_cached_root_rot_mats_22"))
            if data.get("_cached_root_rot_mats_22") is not None
            else root_rotation_matrices_from_poses(poses_3d, device=self.device)
        )
        root_ang_vel = root_angular_velocity_deg_per_frame(root_rot_mats)
        root_horiz_vel = np.linalg.norm(np.diff(joints[:, 0, [0, 2]], axis=0), axis=1)
        foot_joints_local = apply_inverse_root_rotation(foot_joints - joints[:, :1, :], root_rot_mats)
        joints_local = apply_inverse_root_rotation(joints - joints[:, :1, :], root_rot_mats)
        foot_vel = np.diff(foot_joints, axis=0)  # (F-1, 4, 3)
        foot_horiz_vel = np.linalg.norm(foot_vel[:, :, [0, 2]], axis=2)  # (F-1, 4)
        foot_vert_vel = np.abs(foot_vel[:, :, 1])  # (F-1, 4)
        joints_local_vel = np.linalg.norm(np.diff(joints_local[:, :, [0, 2]], axis=0), axis=2)
        foot_heights = foot_joints[:-1, :, 1]  # (F-1, 4)
        joint_ground_y = np.percentile(foot_heights, 3, axis=0)
        joint_contact_threshold = joint_ground_y + self.contact_height_margin_m
        joint_contact = (
            (foot_heights <= joint_contact_threshold[None, :])
            & (foot_vert_vel <= self.contact_vertical_velocity_m_per_frame)
        )

        sliding_segments: List[Dict] = []
        rejected_segments: List[Dict] = []
        total_sliding_frames = 0
        longest_sliding_segment = 0
        invalid_mask = np.zeros((F, NUM_BODY_JOINTS), dtype=bool)
        borderline_mask = np.zeros((F, NUM_BODY_JOINTS), dtype=bool)
        for group in FOOT_SIDE_GROUPS:
            group_indices = group["indices"]
            side_joint_ids = SIDE_JOINT_GROUPS[group["side"]]
            side_positions = np.mean(foot_joints[:, group_indices, :], axis=1)
            side_positions_local = np.mean(foot_joints_local[:, group_indices, :], axis=1)
            side_vel = np.diff(side_positions, axis=0)
            side_vel_local = np.diff(side_positions_local, axis=0)
            side_horiz_vel = np.linalg.norm(side_vel[:, [0, 2]], axis=1)
            side_horiz_vel_local = np.linalg.norm(side_vel_local[:, [0, 2]], axis=1)
            side_vert_vel = np.abs(side_vel[:, 1])
            side_heights = np.mean(foot_heights[:, group_indices], axis=1)
            side_ground_y = float(np.min(joint_ground_y[group_indices]))
            side_contact_support = np.sum(joint_contact[:, group_indices], axis=1)
            side_contact = (
                ((side_contact_support >= 2) | ((side_contact_support >= 1) & (side_heights <= side_ground_y + self.contact_height_margin_m * 1.5)))
                & (side_vert_vel <= self.contact_vertical_velocity_m_per_frame)
            )
            side_sliding = (
                side_contact
                & (side_vert_vel <= self.contact_vertical_velocity_m_per_frame)
                & (side_horiz_vel >= self.sliding_velocity_m_per_frame)
            )
            side_sliding = self._close_short_gaps(side_sliding, self.max_contact_gap_frames)
            segments = self._find_true_segments(side_sliding, self.min_sliding_frames)
            for seg in segments:
                start = seg["start"]
                end = seg["end"]
                world_metrics = self._trajectory_metrics(side_positions[start : end + 1][:, [0, 2]])
                local_metrics = self._trajectory_metrics(side_positions_local[start : end + 1][:, [0, 2]])
                if (
                    world_metrics["directional_consistency"] < self.min_directional_consistency
                    or world_metrics["net_displacement_m"] < self.min_net_displacement_m
                    or world_metrics["turn_count"] > self.max_sliding_turns
                ):
                    rejected_segments.append(
                        {
                            "side": group["side"],
                            "start_frame": start,
                            "end_frame": end,
                            "num_frames": seg["num_frames"],
                            "directional_consistency": world_metrics["directional_consistency"],
                            "net_displacement_m": world_metrics["net_displacement_m"],
                            "turn_count": world_metrics["turn_count"],
                            "reason": "wobble_like_or_too_short_drift",
                        }
                    )
                    continue
                per_joint_mean_vel = [
                    float(np.mean(foot_horiz_vel[start:end, local_idx])) for local_idx in group_indices
                ]
                rep_offset = int(np.argmax(per_joint_mean_vel))
                mean_world_horiz_vel = float(np.mean(side_horiz_vel[start:end]))
                mean_local_horiz_vel = float(np.mean(side_horiz_vel_local[start:end]))
                mean_root_horiz_vel = float(np.mean(root_horiz_vel[start:end])) if root_horiz_vel.size else 0.0
                mean_root_ang_vel = float(np.mean(root_ang_vel[start:end])) if root_ang_vel.size else 0.0
                mean_hip_local_vel = float(np.mean(joints_local_vel[start:end, side_joint_ids["hip"]]))
                mean_knee_local_vel = float(np.mean(joints_local_vel[start:end, side_joint_ids["knee"]]))
                mean_leg_chain_local_vel = float(np.mean([mean_hip_local_vel, mean_knee_local_vel, mean_local_horiz_vel]))
                mean_contact_support = float(np.mean(side_contact_support[start:end])) if side_contact_support.size else 0.0
                double_contact_ratio = float(np.mean(side_contact_support[start:end] >= 2)) if side_contact_support.size else 0.0
                if (
                    mean_contact_support < self.min_contact_support_mean
                    or double_contact_ratio < self.min_double_contact_ratio
                ):
                    rejected_segments.append(
                        {
                            "side": group["side"],
                            "start_frame": start,
                            "end_frame": end,
                            "num_frames": seg["num_frames"],
                            "directional_consistency": world_metrics["directional_consistency"],
                            "net_displacement_m": world_metrics["net_displacement_m"],
                            "turn_count": world_metrics["turn_count"],
                            "mean_contact_support": mean_contact_support,
                            "double_contact_ratio": double_contact_ratio,
                            "reason": "insufficient_planted_support",
                        }
                    )
                    continue
                if (
                    mean_root_horiz_vel >= mean_world_horiz_vel * 0.6
                    and mean_local_horiz_vel <= self.translation_only_local_vel_m_per_frame
                    and mean_leg_chain_local_vel <= self.leg_chain_local_vel_m_per_frame
                ):
                    cause_type = "translation_only"
                    joint_ids = [0]
                elif mean_leg_chain_local_vel >= max(self.leg_chain_local_vel_m_per_frame, mean_local_horiz_vel * 0.75):
                    cause_type = "full_leg_chain"
                    joint_ids = SIDE_REPAIR_JOINT_IDS[group["side"]]
                else:
                    cause_type = "foot_plus_translation"
                    joint_ids = [0] + list(group["joint_ids"])
                if mean_root_ang_vel >= 5.0 and mean_root_horiz_vel < self.sliding_velocity_m_per_frame:
                    root_motion_source = "global_orientation"
                elif mean_root_horiz_vel >= self.sliding_velocity_m_per_frame and mean_root_ang_vel < 5.0:
                    root_motion_source = "translation"
                elif mean_root_horiz_vel >= self.sliding_velocity_m_per_frame or mean_root_ang_vel >= 5.0:
                    root_motion_source = "translation+global_orientation"
                else:
                    root_motion_source = "limb_local_motion"
                root_driven = cause_type != "full_leg_chain"
                frame_joint_hits: List[Dict[str, object]] = []
                for frame_idx in range(start, min(end, joint_contact.shape[0])):
                    local_focus_joint_ids = [
                        group["joint_ids"][local_idx]
                        for local_idx, foot_local_idx in enumerate(group_indices)
                        if bool(joint_contact[frame_idx, foot_local_idx])
                        and float(foot_horiz_vel[frame_idx, foot_local_idx]) >= self.sliding_velocity_m_per_frame
                    ]
                    if not local_focus_joint_ids:
                        local_focus_joint_ids = list(group["joint_ids"])
                    if cause_type == "translation_only":
                        mask_joint_ids = sorted(set([0] + local_focus_joint_ids))
                    elif cause_type == "full_leg_chain":
                        mask_joint_ids = sorted(set(SIDE_REPAIR_JOINT_IDS[group["side"]]))
                    else:
                        mask_joint_ids = sorted(set([0] + local_focus_joint_ids))
                    frame_joint_hits.append(
                        {
                            "frame": int(frame_idx),
                            "focus_joint_ids": [int(j) for j in local_focus_joint_ids],
                            "mask_joint_ids": [int(j) for j in mask_joint_ids],
                        }
                    )
                sliding_segments.append(
                    {
                        "side": group["side"],
                        "joint_id": group["joint_ids"][rep_offset],
                        "joint_name": group["joint_names"][rep_offset],
                        "joint_ids": joint_ids,
                        "focus_joint_ids": list(group["joint_ids"]),
                        "joint_names": list(group["joint_names"]),
                        "start_frame": start,
                        "end_frame": end,
                        "num_frames": seg["num_frames"],
                        "cause_type": cause_type,
                        "root_motion_source": root_motion_source,
                        "mean_horiz_vel": mean_world_horiz_vel,
                        "mean_horiz_vel_root_stabilized": mean_local_horiz_vel,
                        "mean_root_horiz_vel": mean_root_horiz_vel,
                        "mean_hip_local_vel": mean_hip_local_vel,
                        "mean_knee_local_vel": mean_knee_local_vel,
                        "mean_leg_chain_local_vel": mean_leg_chain_local_vel,
                        "max_horiz_vel": float(np.max(side_horiz_vel[start:end])),
                        "max_horiz_vel_root_stabilized": float(np.max(side_horiz_vel_local[start:end])),
                        "mean_root_ang_vel_deg": mean_root_ang_vel,
                        "directional_consistency": world_metrics["directional_consistency"],
                        "net_displacement_m": world_metrics["net_displacement_m"],
                        "path_length_m": world_metrics["path_length_m"],
                        "trajectory_turns": world_metrics["turn_count"],
                        "root_stabilized_directional_consistency": local_metrics["directional_consistency"],
                        "root_stabilized_net_displacement_m": local_metrics["net_displacement_m"],
                        "root_driven": bool(root_driven),
                        "contact_frames": int(np.sum(side_contact[start:end])),
                        "contact_support_mean": mean_contact_support,
                        "double_contact_ratio": double_contact_ratio,
                        "frame_joint_hits": frame_joint_hits,
                    }
                )
                for hit in frame_joint_hits:
                    fi = int(hit["frame"])
                    joint_ids_this_frame = [int(j) for j in (hit.get("mask_joint_ids") or []) if 0 <= int(j) < NUM_BODY_JOINTS]
                    if 0 <= fi < invalid_mask.shape[0] and joint_ids_this_frame:
                        invalid_mask[fi, joint_ids_this_frame] = True
                total_sliding_frames += seg["num_frames"]
                longest_sliding_segment = max(longest_sliding_segment, seg["num_frames"])

        if (
            total_sliding_frames < self.min_total_sliding_frames
            and longest_sliding_segment < self.strong_segment_frames
        ):
            if sliding_segments:
                borderline_mask = invalid_mask.copy()
                reason = (
                    f"Mild foot sliding detected in {len(sliding_segments)} segment(s), "
                    f"{total_sliding_frames} total frames below low-quality threshold"
                )
                return CheckResult(
                    is_valid=True,
                    invalid_reason=reason,
                    invalid_mask=borderline_mask,
                    details={
                        "has_sliding": False,
                        "has_borderline_sliding": True,
                        "sliding_segments": sliding_segments,
                        "rejected_segments": rejected_segments,
                        "total_sliding_frames": total_sliding_frames,
                        "longest_sliding_segment": longest_sliding_segment,
                        "reason": reason,
                    },
                    severity="borderline",
                )
            return CheckResult(
                is_valid=True,
                invalid_reason="No foot sliding detected",
                invalid_mask=np.zeros((F, NUM_BODY_JOINTS), dtype=bool),
                details={
                    "has_sliding": False,
                    "sliding_segments": [],
                    "rejected_segments": rejected_segments,
                    "total_sliding_frames": total_sliding_frames,
                    "longest_sliding_segment": longest_sliding_segment,
                    "reason": "No foot sliding detected",
                },
                severity="pass",
            )

        reason = (
            f"Foot sliding in {len(sliding_segments)} segment(s), "
            f"{total_sliding_frames} total frames"
        )
        details = {
            "has_sliding": True,
            "sliding_segments": sliding_segments,
            "rejected_segments": rejected_segments,
            "total_sliding_frames": total_sliding_frames,
            "longest_sliding_segment": longest_sliding_segment,
            "joint_ground_y": {name: float(joint_ground_y[idx]) for idx, name in enumerate(FOOT_JOINT_NAMES)},
            "contact_height_margin_m": float(self.contact_height_margin_m),
            "contact_vertical_velocity_m_per_frame": float(self.contact_vertical_velocity_m_per_frame),
            "sliding_velocity_m_per_frame": float(self.sliding_velocity_m_per_frame),
            "min_directional_consistency": float(self.min_directional_consistency),
            "min_net_displacement_m": float(self.min_net_displacement_m),
            "max_sliding_turns": int(self.max_sliding_turns),
            "strong_segment_frames": int(self.strong_segment_frames),
            "min_contact_support_mean": float(self.min_contact_support_mean),
            "min_double_contact_ratio": float(self.min_double_contact_ratio),
            "reason": reason,
        }
        return CheckResult(
            is_valid=False,
            invalid_reason=reason,
            invalid_mask=invalid_mask,
            details=details,
            severity="fail",
        )


_CHECKER_CACHE: Dict[str, FootSlidingChecker] = {}


def detect_foot_sliding(data: Dict, device: str = "cpu", **kwargs) -> Dict:
    """Legacy API for filter scripts."""
    cache_key = device
    if cache_key not in _CHECKER_CACHE:
        _CHECKER_CACHE[cache_key] = FootSlidingChecker(device=device, **kwargs)
    checker = _CHECKER_CACHE[cache_key]
    result = checker.check(data)
    details = result.get("details") or {}
    return {
        "has_sliding": details.get("has_sliding", False),
        "sliding_segments": details.get("sliding_segments", []),
        "total_sliding_frames": details.get("total_sliding_frames", 0),
        "reason": details.get("reason", result.get("invalid_reason", "")),
    }
