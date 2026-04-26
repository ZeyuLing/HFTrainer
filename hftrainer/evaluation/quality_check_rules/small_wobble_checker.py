"""
Small wobble checker: detects unwanted micro-motion during stable segments.

The expensive part of the previous implementation was repeated per-window,
per-joint median/percentile/PCA work in Python loops. This version keeps the
same high-level rule but batches the window statistics and only loops over the
small set of flagged windows to assemble human-readable details.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from .base_checker import BaseQualityChecker, CheckResult, normalize_betas_array
from .root_motion_utils import (
    apply_inverse_root_rotation,
    root_angular_velocity_deg_per_frame,
    root_rotation_matrices_from_poses,
)

from ._model_compat import SmplxLiteJ24

NUM_BODY_JOINTS = 22
PELVIS_JOINT = 0
STABLE_JOINT_INDICES = [0, 7, 8, 10, 11]
STABLE_JOINT_NAMES = ["Pelvis", "LFoot", "RFoot", "LToeBase", "RToeBase"]

DEFAULT_STABLE_ROOT_VELOCITY_M_PER_FRAME = 0.005
DEFAULT_STABLE_ROOT_ANGULAR_VELOCITY_DEG_PER_FRAME = 3.0
DEFAULT_STABLE_JOINT_VELOCITY_M_PER_FRAME = 0.005
DEFAULT_MILD_WOBBLE_JOINT_VELOCITY_M_PER_FRAME = 0.0025
DEFAULT_WOBBLE_AMPLITUDE_M = 0.03
DEFAULT_ROOT_RELATIVE_WOBBLE_AMPLITUDE_M = 0.04
DEFAULT_MILD_WOBBLE_AMPLITUDE_M = 0.006
DEFAULT_WINDOW_SIZE = 30
DEFAULT_MIN_WOBBLE_WINDOWS = 6
DEFAULT_MIN_OSCILLATION_TURNS = 3
DEFAULT_ENABLE_MILD_OSCILLATORY_RULE = False
DEFAULT_MAX_DRIFT_RATIO = 0.75


def _get_joints_from_pose(
    poses: np.ndarray,
    trans: np.ndarray,
    betas: Optional[np.ndarray] = None,
    body_model: Optional[Any] = None,
    device: str = "cpu",
) -> np.ndarray:
    if body_model is None or SmplxLiteJ24 is None:
        raise RuntimeError("SmallWobbleChecker requires body_model (SmplxLiteJ24) for FK.")
    num_frames = poses.shape[0]
    poses_t = torch.as_tensor(poses, dtype=torch.float32, device=device)
    trans_t = torch.as_tensor(trans, dtype=torch.float32, device=device)
    if betas is None:
        betas_t = torch.zeros((1, 16), dtype=torch.float32, device=device)
    else:
        betas_t = torch.as_tensor(betas, dtype=torch.float32, device=device)
        if betas_t.ndim == 1:
            betas_t = betas_t.unsqueeze(0)
    global_orient = poses_t[:, 0, :]
    body_pose = poses_t[:, 1:22, :].reshape(num_frames, 63)
    with torch.no_grad():
        joints = body_model(
            body_pose=body_pose,
            betas=betas_t,
            global_orient=global_orient,
            transl=trans_t,
            rotation_mode="aa",
        )
    return joints.cpu().numpy()


def _window_view_axis0(arr: np.ndarray, window_size: int) -> np.ndarray:
    view = sliding_window_view(arr, window_shape=window_size, axis=0)
    return np.moveaxis(view, -1, 1)


def _root_velocity_per_frame(joints: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(joints[:, 0, :], axis=0), axis=1)


def _build_eval_positions(joints: np.ndarray, root_rot_mats: np.ndarray) -> np.ndarray:
    positions = joints[:, STABLE_JOINT_INDICES, :].copy()
    positions[:, 1:, :] = apply_inverse_root_rotation(positions[:, 1:, :] - joints[:, :1, :], root_rot_mats)
    return positions


def _smooth_projected(projected: np.ndarray) -> np.ndarray:
    if projected.shape[1] < 3:
        return projected
    smooth = projected.copy()
    smooth[:, 1:-1, :] = (
        0.25 * projected[:, :-2, :]
        + 0.50 * projected[:, 1:-1, :]
        + 0.25 * projected[:, 2:, :]
    )
    smooth[:, 0, :] = 0.75 * projected[:, 0, :] + 0.25 * projected[:, 1, :]
    smooth[:, -1, :] = 0.25 * projected[:, -2, :] + 0.75 * projected[:, -1, :]
    return smooth


def _dominant_axis_projection(centered_windows: np.ndarray) -> np.ndarray:
    axis_ranges = np.ptp(centered_windows, axis=1)
    dominant_axis = np.argmax(axis_ranges, axis=-1)
    projected = np.take_along_axis(
        centered_windows,
        dominant_axis[:, None, :, None],
        axis=-1,
    )
    return projected[..., 0]


def _count_oscillation_turns_batch(projected: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    smooth = _smooth_projected(projected)
    delta = np.diff(smooth, axis=1)
    amp95 = np.percentile(np.abs(smooth), 95, axis=1)
    eps = np.maximum(1e-4, 0.04 * amp95)

    signs = np.where(
        np.abs(delta) >= eps[:, None, :],
        np.where(delta > 0.0, 1, -1),
        0,
    )

    flat = signs.transpose(0, 2, 1).reshape(-1, signs.shape[1])
    turns = np.zeros((flat.shape[0],), dtype=np.int32)
    for idx, seq in enumerate(flat):
        non_zero = seq[seq != 0]
        if non_zero.size >= 2:
            turns[idx] = int(np.sum(non_zero[1:] != non_zero[:-1]))
    return turns.reshape(projected.shape[0], projected.shape[2]), amp95


def _edge_drift_ratio_batch(projected: np.ndarray, amp95: np.ndarray) -> np.ndarray:
    edge_disp = np.abs(projected[:, -1, :] - projected[:, 0, :])
    return edge_disp / np.maximum(2.0 * amp95, 1e-6)


class SmallWobbleChecker(BaseQualityChecker):
    name = "small_wobble"

    def __init__(
        self,
        body_model: Optional[Any] = None,
        device: str = "cuda",
        stable_root_velocity_m_per_frame: float = DEFAULT_STABLE_ROOT_VELOCITY_M_PER_FRAME,
        stable_root_angular_velocity_deg_per_frame: float = DEFAULT_STABLE_ROOT_ANGULAR_VELOCITY_DEG_PER_FRAME,
        stable_joint_velocity_m_per_frame: float = DEFAULT_STABLE_JOINT_VELOCITY_M_PER_FRAME,
        mild_wobble_joint_velocity_m_per_frame: float = DEFAULT_MILD_WOBBLE_JOINT_VELOCITY_M_PER_FRAME,
        wobble_amplitude_m: float = DEFAULT_WOBBLE_AMPLITUDE_M,
        root_relative_wobble_amplitude_m: float = DEFAULT_ROOT_RELATIVE_WOBBLE_AMPLITUDE_M,
        mild_wobble_amplitude_m: float = DEFAULT_MILD_WOBBLE_AMPLITUDE_M,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_wobble_windows: int = DEFAULT_MIN_WOBBLE_WINDOWS,
        min_oscillation_turns: int = DEFAULT_MIN_OSCILLATION_TURNS,
        enable_mild_oscillatory_rule: bool = DEFAULT_ENABLE_MILD_OSCILLATORY_RULE,
        max_drift_ratio: float = DEFAULT_MAX_DRIFT_RATIO,
    ) -> None:
        super().__init__(body_model=body_model, device=device)
        if self.body_model is None and SmplxLiteJ24 is not None:
            self.body_model = SmplxLiteJ24(gender="neutral").to(self.device)
            self.body_model.eval()
        self.stable_root_velocity_m_per_frame = stable_root_velocity_m_per_frame
        self.stable_root_angular_velocity_deg_per_frame = stable_root_angular_velocity_deg_per_frame
        self.stable_joint_velocity_m_per_frame = stable_joint_velocity_m_per_frame
        self.mild_wobble_joint_velocity_m_per_frame = mild_wobble_joint_velocity_m_per_frame
        self.wobble_amplitude_m = wobble_amplitude_m
        self.root_relative_wobble_amplitude_m = root_relative_wobble_amplitude_m
        self.mild_wobble_amplitude_m = mild_wobble_amplitude_m
        self.window_size = window_size
        self.min_wobble_windows = min_wobble_windows
        self.min_oscillation_turns = min_oscillation_turns
        self.enable_mild_oscillatory_rule = bool(enable_mild_oscillatory_rule)
        self.max_drift_ratio = max_drift_ratio

    def get_required_keys(self) -> list:
        return ["poses", "trans"]

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
                details={"has_wobble": False, "reason": err},
            )

        poses = np.array(data["poses"])
        trans = np.array(data["trans"])
        if len(poses) < self.window_size:
            reason = f"Too short (need at least {self.window_size} frames)"
            return CheckResult(
                is_valid=False,
                invalid_reason=reason,
                invalid_mask=None,
                details={"has_wobble": False, "reason": reason},
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
                details={"has_wobble": False, "reason": str(e)},
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
                details={"has_wobble": False, "reason": f"FK failed: {str(e)}"},
            )

        num_frames = joints.shape[0]
        window_size = int(self.window_size)
        num_windows = num_frames - window_size + 1
        if num_windows <= 0:
            reason = f"Too short (need at least {self.window_size} frames)"
            return CheckResult(
                is_valid=False,
                invalid_reason=reason,
                invalid_mask=None,
                details={"has_wobble": False, "reason": reason},
            )

        root_rot_mats = (
            np.asarray(data.get("_cached_root_rot_mats_22"))
            if data.get("_cached_root_rot_mats_22") is not None
            else root_rotation_matrices_from_poses(poses_3d, device=self.device)
        )
        eval_positions = _build_eval_positions(joints, root_rot_mats)
        root_vel = _root_velocity_per_frame(joints)
        root_ang_vel = root_angular_velocity_deg_per_frame(root_rot_mats)
        joint_vel = np.linalg.norm(np.diff(eval_positions, axis=0), axis=2)

        root_vel_windows = _window_view_axis0(root_vel[:, None], window_size - 1)[:, :, 0]
        root_ang_vel_windows = _window_view_axis0(root_ang_vel[:, None], window_size - 1)[:, :, 0]
        joint_vel_windows = _window_view_axis0(joint_vel, window_size - 1)
        pos_windows = _window_view_axis0(eval_positions, window_size)

        stable_window_mask = (
            (np.mean(root_vel_windows, axis=1) <= self.stable_root_velocity_m_per_frame)
            & (np.mean(root_ang_vel_windows, axis=1) <= self.stable_root_angular_velocity_deg_per_frame)
        )
        if not np.any(stable_window_mask):
            empty_mask = np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool)
            return CheckResult(
                is_valid=True,
                invalid_reason="No small wobble detected",
                invalid_mask=empty_mask,
                details={
                    "has_wobble": False,
                    "wobble_windows": [],
                    "reason": "No small wobble detected",
                },
                severity="pass",
            )

        joint_vel_p95 = np.percentile(joint_vel_windows, 95, axis=1)
        centers = np.median(pos_windows, axis=1)
        centered = pos_windows - centers[:, None, :, :]
        frame_offsets = np.linalg.norm(centered, axis=-1)
        offset_p95 = np.percentile(frame_offsets, 95, axis=1)
        projected = _dominant_axis_projection(centered)
        turns, amp95 = _count_oscillation_turns_batch(projected)
        drift_ratio = _edge_drift_ratio_batch(projected, amp95)

        strong_thresholds = np.asarray(
            [
                self.wobble_amplitude_m,
                self.root_relative_wobble_amplitude_m,
                self.root_relative_wobble_amplitude_m,
                self.root_relative_wobble_amplitude_m,
                self.root_relative_wobble_amplitude_m,
            ],
            dtype=np.float64,
        )

        strong_hit = (
            stable_window_mask[:, None]
            & (joint_vel_p95 <= self.stable_joint_velocity_m_per_frame)
            & (offset_p95 >= strong_thresholds[None, :])
        )
        per_joint_min_turns = np.asarray(
            [
                self.min_oscillation_turns,
                max(2, self.min_oscillation_turns - 1),
                max(2, self.min_oscillation_turns - 1),
                max(2, self.min_oscillation_turns - 1),
                max(2, self.min_oscillation_turns - 1),
            ],
            dtype=np.int32,
        )
        strong_hit &= turns >= per_joint_min_turns[None, :]
        strong_hit &= drift_ratio <= self.max_drift_ratio

        mild_hit = np.zeros_like(strong_hit, dtype=bool)
        if self.enable_mild_oscillatory_rule:
            mild_hit = (
                stable_window_mask[:, None]
                & (joint_vel_p95 <= self.mild_wobble_joint_velocity_m_per_frame)
                & (offset_p95 >= self.mild_wobble_amplitude_m)
                & (turns >= self.min_oscillation_turns)
                & (drift_ratio <= self.max_drift_ratio)
            )

        candidate_mask = strong_hit | mild_hit
        candidate_mask[:, 0] |= False
        if not np.any(candidate_mask):
            empty_mask = np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool)
            return CheckResult(
                is_valid=True,
                invalid_reason="No small wobble detected",
                invalid_mask=empty_mask,
                details={
                    "has_wobble": False,
                    "wobble_windows": [],
                    "reason": "No small wobble detected",
                },
                severity="pass",
            )

        score = offset_p95 + 0.002 * turns + 0.05 * strong_hit.astype(np.float64)
        score[~candidate_mask] = -np.inf
        flagged_window_indices = np.where(np.any(candidate_mask, axis=1))[0].astype(int).tolist()

        invalid_mask = np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool)
        wobble_windows: List[Dict[str, Any]] = []
        for window_idx in flagged_window_indices:
            local_joint_idx = int(np.argmax(score[window_idx]))
            joint_id = STABLE_JOINT_INDICES[local_joint_idx]
            joint_name = STABLE_JOINT_NAMES[local_joint_idx]
            start = int(window_idx)
            end = int(window_idx + window_size)
            strong_mode = bool(strong_hit[window_idx, local_joint_idx])
            mode = "strong" if strong_mode else "mild"
            threshold = (
                float(strong_thresholds[local_joint_idx])
                if strong_mode
                else float(self.mild_wobble_amplitude_m)
            )
            local_offsets = frame_offsets[window_idx, :, local_joint_idx]
            active_level = max(threshold * 0.75, float(np.percentile(local_offsets, 70)))
            active_local = np.where(local_offsets >= active_level)[0].astype(int)
            if active_local.size == 0:
                active_local = np.asarray([int(np.argmax(local_offsets))], dtype=int)
            active_frames = (active_local + start).astype(int).tolist()
            invalid_mask[active_frames, joint_id] = True

            wobble_windows.append(
                {
                    "start": start,
                    "end": end,
                    "joint": joint_name,
                    "joint_id": joint_id,
                    "mode": mode,
                    "space": "world" if joint_id == PELVIS_JOINT else "root_stabilized",
                    "wobble_score_m": float(offset_p95[window_idx, local_joint_idx]),
                    "joint_velocity_p95_m": float(joint_vel_p95[window_idx, local_joint_idx]),
                    "oscillation_turns": int(turns[window_idx, local_joint_idx]),
                    "dominant_axis_amp95_m": float(amp95[window_idx, local_joint_idx]),
                    "drift_ratio": float(drift_ratio[window_idx, local_joint_idx]),
                    "mean_root_vel": float(np.mean(root_vel_windows[window_idx])),
                    "mean_root_ang_vel_deg": float(np.mean(root_ang_vel_windows[window_idx])),
                    "active_frames": active_frames,
                    "active_frame_count": int(len(active_frames)),
                    "strong_wobble_threshold_m": float(strong_thresholds[local_joint_idx]),
                    "mild_wobble_threshold_m": float(self.mild_wobble_amplitude_m),
                    "oscillation_turns_threshold": int(self.min_oscillation_turns),
                }
            )

        if len(wobble_windows) < self.min_wobble_windows:
            reason = (
                f"Mild wobble detected in {len(wobble_windows)} stable segment(s); "
                f"low-quality threshold is {self.min_wobble_windows}"
            )
            return CheckResult(
                is_valid=True,
                invalid_reason=reason,
                invalid_mask=invalid_mask,
                details={
                    "has_wobble": False,
                    "has_borderline_wobble": True,
                    "wobble_windows": wobble_windows,
                    "reason": reason,
                    "min_wobble_windows": int(self.min_wobble_windows),
                },
                severity="borderline",
            )

        reason = (
            f"Small wobble in {len(wobble_windows)} stable segment(s) "
            f"(pelvis/world >= {self.wobble_amplitude_m * 100:.1f} cm, "
            f"foot-rootrel >= {self.root_relative_wobble_amplitude_m * 100:.1f} cm"
            + (
                f", mild >= {self.mild_wobble_amplitude_m * 100:.1f} cm with >= {self.min_oscillation_turns} turns"
                if self.enable_mild_oscillatory_rule else ""
            )
            + ")"
        )
        details = {
            "has_wobble": True,
            "wobble_windows": wobble_windows,
            "wobble_amplitude_m": self.wobble_amplitude_m,
            "root_relative_wobble_amplitude_m": self.root_relative_wobble_amplitude_m,
            "mild_wobble_amplitude_m": self.mild_wobble_amplitude_m,
            "stable_velocity_threshold": self.stable_root_velocity_m_per_frame,
            "stable_root_angular_velocity_deg_per_frame": self.stable_root_angular_velocity_deg_per_frame,
            "stable_joint_velocity_threshold": self.stable_joint_velocity_m_per_frame,
            "mild_wobble_joint_velocity_threshold": self.mild_wobble_joint_velocity_m_per_frame,
            "min_oscillation_turns": self.min_oscillation_turns,
            "max_drift_ratio": self.max_drift_ratio,
            "enable_mild_oscillatory_rule": self.enable_mild_oscillatory_rule,
            "reason": reason,
        }
        return CheckResult(
            is_valid=False,
            invalid_reason=reason,
            invalid_mask=invalid_mask,
            details=details,
            severity="fail",
        )


_CHECKER_CACHE: Dict[str, SmallWobbleChecker] = {}


def detect_small_wobble(
    data: Dict,
    device: str = "cpu",
    stable_root_velocity_m_per_frame: Optional[float] = None,
    stable_root_angular_velocity_deg_per_frame: Optional[float] = None,
    stable_joint_velocity_m_per_frame: Optional[float] = None,
    mild_wobble_joint_velocity_m_per_frame: Optional[float] = None,
    wobble_amplitude_m: Optional[float] = None,
    root_relative_wobble_amplitude_m: Optional[float] = None,
    mild_wobble_amplitude_m: Optional[float] = None,
    window_size: Optional[int] = None,
    min_wobble_windows: Optional[int] = None,
    min_oscillation_turns: Optional[int] = None,
    enable_mild_oscillatory_rule: Optional[bool] = None,
) -> Dict:
    cache_key = device
    if cache_key not in _CHECKER_CACHE:
        kwargs = {}
        if stable_root_velocity_m_per_frame is not None:
            kwargs["stable_root_velocity_m_per_frame"] = stable_root_velocity_m_per_frame
        if stable_root_angular_velocity_deg_per_frame is not None:
            kwargs["stable_root_angular_velocity_deg_per_frame"] = stable_root_angular_velocity_deg_per_frame
        if stable_joint_velocity_m_per_frame is not None:
            kwargs["stable_joint_velocity_m_per_frame"] = stable_joint_velocity_m_per_frame
        if mild_wobble_joint_velocity_m_per_frame is not None:
            kwargs["mild_wobble_joint_velocity_m_per_frame"] = mild_wobble_joint_velocity_m_per_frame
        if wobble_amplitude_m is not None:
            kwargs["wobble_amplitude_m"] = wobble_amplitude_m
        if root_relative_wobble_amplitude_m is not None:
            kwargs["root_relative_wobble_amplitude_m"] = root_relative_wobble_amplitude_m
        if mild_wobble_amplitude_m is not None:
            kwargs["mild_wobble_amplitude_m"] = mild_wobble_amplitude_m
        if window_size is not None:
            kwargs["window_size"] = window_size
        if min_wobble_windows is not None:
            kwargs["min_wobble_windows"] = min_wobble_windows
        if min_oscillation_turns is not None:
            kwargs["min_oscillation_turns"] = min_oscillation_turns
        if enable_mild_oscillatory_rule is not None:
            kwargs["enable_mild_oscillatory_rule"] = enable_mild_oscillatory_rule
        _CHECKER_CACHE[cache_key] = SmallWobbleChecker(device=device, **kwargs)
    checker = _CHECKER_CACHE[cache_key]
    result = checker.check(data)
    details = result.get("details") or {}
    return {
        "has_wobble": details.get("has_wobble", False),
        "wobble_windows": details.get("wobble_windows", []),
        "reason": details.get("reason", result.get("invalid_reason", "")),
    }
