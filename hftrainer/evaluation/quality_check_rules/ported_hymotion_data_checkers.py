"""Portable checkers migrated from HYMotion_Data filters/operators.

This module ports the directly reusable, non-ML operators first:
  - duration
  - rest_pose
  - ground_penetration
  - first_frame_rotation_velocity
  - knee_x
  - ankle_x
  - neck
  - spine / spine1 / spine2

Operators that depend on classifier checkpoints are intentionally excluded here.
Operators that strongly overlap with existing local checkers (e.g. stationary,
translation, twist) are also kept out of the default chain for now to avoid
double-counting and semantic conflicts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch

from ._geometry_compat import axis_angle_to_matrix

from .base_checker import BaseQualityChecker, CheckResult, normalize_poses_array
from .tbs_utils import extract_joint_tbs_metrics

NUM_BODY_JOINTS = 22
DEFAULT_FPS = 30.0

JOINT_NAMES = {
    0: "Pelvis",
    1: "L_Hip",
    2: "R_Hip",
    3: "Spine1",
    4: "L_Knee",
    5: "R_Knee",
    6: "Spine2",
    7: "L_Ankle",
    8: "R_Ankle",
    9: "Spine3",
    10: "L_Foot",
    11: "R_Foot",
    12: "Neck",
    13: "L_Collar",
    14: "R_Collar",
    15: "Head",
    16: "L_Shoulder",
    17: "R_Shoulder",
    18: "L_Elbow",
    19: "R_Elbow",
    20: "L_Wrist",
    21: "R_Wrist",
}


def _full_mask(num_frames: int, joint_ids: Optional[Sequence[int]] = None) -> np.ndarray:
    mask = np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool)
    if num_frames <= 0:
        return mask
    if joint_ids is None:
        mask[:, :] = True
    else:
        for joint_id in joint_ids:
            if 0 <= int(joint_id) < NUM_BODY_JOINTS:
                mask[:, int(joint_id)] = True
    return mask


def _frames_mask(frame_mask: np.ndarray, joint_ids: Sequence[int]) -> np.ndarray:
    num_frames = int(frame_mask.shape[0])
    mask = np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool)
    for joint_id in joint_ids:
        if 0 <= int(joint_id) < NUM_BODY_JOINTS:
            mask[:, int(joint_id)] = np.asarray(frame_mask, dtype=bool)
    return mask


def _rotation_angle_deg(rot_mats: torch.Tensor) -> torch.Tensor:
    trace = rot_mats[..., 0, 0] + rot_mats[..., 1, 1] + rot_mats[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos_theta))


def _get_duration_seconds(data: Dict[str, Any], num_frames: int) -> float:
    if "duration" in data:
        try:
            return float(data["duration"])
        except Exception:
            pass
    fps_candidates = ("fps", "mocap_framerate", "framerate")
    fps = None
    for key in fps_candidates:
        if key in data:
            try:
                fps = float(np.asarray(data[key]).reshape(-1)[0])
                break
            except Exception:
                continue
    if fps is None or not np.isfinite(fps) or fps <= 1e-6:
        fps = DEFAULT_FPS
    return float(num_frames / fps)


class DurationChecker(BaseQualityChecker):
    name = "duration"

    def __init__(self, min_duration: float = 0.5, max_duration: float = 60.0, device: str = "cuda") -> None:
        super().__init__(body_model=None, device=device)
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)

    def get_required_keys(self) -> list:
        return ["poses"]

    def check(self, motion) -> CheckResult:
        data = self.load_motion(motion)
        err = self.validate_motion_dict(data)
        if err:
            return CheckResult(is_valid=False, invalid_reason=err, invalid_mask=None, details={"error": err})
        poses = np.asarray(data["poses"])
        num_frames = int(poses.shape[0]) if poses.ndim >= 1 else 0
        duration = _get_duration_seconds(data, num_frames)
        reasons = []
        if duration < self.min_duration:
            reasons.append(f"Duration {duration:.2f}s < {self.min_duration:.2f}s")
        if duration > self.max_duration:
            reasons.append(f"Duration {duration:.2f}s > {self.max_duration:.2f}s")
        passed = not reasons
        return CheckResult(
            is_valid=passed,
            invalid_reason="; ".join(reasons) if reasons else "Duration in range",
            invalid_mask=_full_mask(num_frames) if not passed else np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool),
            details={"duration_sec": duration, "min_duration": self.min_duration, "max_duration": self.max_duration},
            severity="pass" if passed else "fail",
        )


class RestPoseChecker(BaseQualityChecker):
    name = "rest_pose"

    def __init__(self, threshold_deg_var: float = 1.0, device: str = "cuda") -> None:
        super().__init__(body_model=None, device=device)
        self.threshold_deg_var = float(threshold_deg_var)

    def get_required_keys(self) -> list:
        return ["poses"]

    def check(self, motion) -> CheckResult:
        data = self.load_motion(motion)
        err = self.validate_motion_dict(data)
        if err:
            return CheckResult(is_valid=False, invalid_reason=err, invalid_mask=None, details={"error": err})
        poses_3d = normalize_poses_array(np.asarray(data["poses"]), num_joints=NUM_BODY_JOINTS)
        num_frames = int(poses_3d.shape[0])
        magnitudes = np.rad2deg(np.linalg.norm(poses_3d, axis=-1))
        pose_variance = float(np.var(magnitudes))
        pose_mean_angle = float(np.mean(magnitudes))
        passed = pose_variance >= self.threshold_deg_var
        return CheckResult(
            is_valid=passed,
            invalid_reason=(
                f"Pose variance {pose_variance:.3f}deg < {self.threshold_deg_var:.3f}deg"
                if not passed
                else "Not a rest pose"
            ),
            invalid_mask=_full_mask(num_frames) if not passed else np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool),
            details={"pose_variance_deg": pose_variance, "pose_mean_angle_deg": pose_mean_angle},
            severity="pass" if passed else "fail",
        )


class GroundPenetrationChecker(BaseQualityChecker):
    name = "ground_penetration"

    def __init__(self, threshold_m: float = 0.05, body_model: Optional[Any] = None, device: str = "cuda") -> None:
        super().__init__(body_model=body_model, device=device)
        self.threshold_m = float(threshold_m)

    def get_required_keys(self) -> list:
        return ["poses", "trans"]

    def check(self, motion) -> CheckResult:
        data = self.load_motion(motion)
        err = self.validate_motion_dict(data)
        if err:
            return CheckResult(is_valid=False, invalid_reason=err, invalid_mask=None, details={"error": err})
        joints = np.asarray(data.get("_cached_joints_22")) if data.get("_cached_joints_22") is not None else None
        if joints is None:
            return CheckResult(
                is_valid=False,
                invalid_reason="Missing FK joints for ground penetration",
                invalid_mask=None,
                details={"error": "missing_fk"},
            )
        foot_joint_ids = [7, 8, 10, 11]
        foot_y = joints[:, foot_joint_ids, 1]
        min_foot_y = float(np.min(foot_y))
        penetration_depth = float(max(0.0, -min_foot_y))
        frame_mask = np.any(foot_y < -self.threshold_m, axis=1)
        passed = not bool(frame_mask.any())
        return CheckResult(
            is_valid=passed,
            invalid_reason=(
                f"Ground penetration depth {penetration_depth:.3f}m > {self.threshold_m:.3f}m"
                if not passed
                else "No ground penetration"
            ),
            invalid_mask=_frames_mask(frame_mask, foot_joint_ids),
            details={"penetration_depth_m": penetration_depth, "min_foot_y": min_foot_y, "joint_ids": foot_joint_ids},
            severity="pass" if passed else "fail",
        )


class FirstFrameRotationVelocityChecker(BaseQualityChecker):
    name = "first_frame_rotation_velocity"

    def __init__(self, threshold_deg: float = 30.0, device: str = "cuda") -> None:
        super().__init__(body_model=None, device=device)
        self.threshold_deg = float(threshold_deg)

    def get_required_keys(self) -> list:
        return ["poses"]

    def check(self, motion) -> CheckResult:
        data = self.load_motion(motion)
        err = self.validate_motion_dict(data)
        if err:
            return CheckResult(is_valid=False, invalid_reason=err, invalid_mask=None, details={"error": err})
        poses_3d = normalize_poses_array(np.asarray(data["poses"]), num_joints=NUM_BODY_JOINTS)
        num_frames = int(poses_3d.shape[0])
        if num_frames < 2:
            return CheckResult(
                is_valid=True,
                invalid_reason="Too short for first/last frame rotation velocity",
                invalid_mask=np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool),
                details={"skipped": True},
                severity="pass",
            )
        rot_mats = axis_angle_to_matrix(torch.as_tensor(poses_3d, dtype=torch.float32))
        rel_first = torch.matmul(rot_mats[1], rot_mats[0].transpose(-1, -2))
        rel_last = torch.matmul(rot_mats[-1], rot_mats[-2].transpose(-1, -2))
        first_angle = float(_rotation_angle_deg(rel_first).mean())
        last_angle = float(_rotation_angle_deg(rel_last).mean())
        reasons = []
        mask = np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool)
        if first_angle > self.threshold_deg:
            reasons.append(f"First-frame rotation spike {first_angle:.1f}deg > {self.threshold_deg:.1f}deg")
            mask[:2, :] = True
        if last_angle > self.threshold_deg:
            reasons.append(f"Last-frame rotation spike {last_angle:.1f}deg > {self.threshold_deg:.1f}deg")
            mask[-2:, :] = True
        passed = not reasons
        return CheckResult(
            is_valid=passed,
            invalid_reason="; ".join(reasons) if reasons else "No first/last frame rotation spike",
            invalid_mask=mask,
            details={"first_frame_rotation_angle_deg": first_angle, "last_frame_rotation_angle_deg": last_angle},
            severity="pass" if passed else "fail",
        )


class _JointXRangeChecker(BaseQualityChecker):
    l_joint_id: int = -1
    r_joint_id: int = -1
    l_joint_name: str = ""
    r_joint_name: str = ""
    name = "joint_x_range"
    use_absolute_metric: bool = False
    enforce_min_bound: bool = True

    def __init__(self, min_angle_deg: float, max_angle_deg: float, device: str = "cuda") -> None:
        super().__init__(body_model=None, device=device)
        self.min_angle_deg = float(min_angle_deg)
        self.max_angle_deg = float(max_angle_deg)

    def get_required_keys(self) -> list:
        return ["poses"]

    def check(self, motion) -> CheckResult:
        data = self.load_motion(motion)
        err = self.validate_motion_dict(data)
        if err:
            return CheckResult(is_valid=False, invalid_reason=err, invalid_mask=None, details={"error": err})
        poses_3d = normalize_poses_array(np.asarray(data["poses"]), num_joints=NUM_BODY_JOINTS)
        num_frames = int(poses_3d.shape[0])
        mask = np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool)
        reasons = []
        details: Dict[str, Any] = {"joint_ranges_deg": {}}
        for joint_id, joint_name in (
            (self.l_joint_id, self.l_joint_name),
            (self.r_joint_id, self.r_joint_name),
        ):
            bend_deg = extract_joint_tbs_metrics(poses_3d[:, joint_id, :], joint_id)["bend_deg"].astype(np.float32)
            metric_deg = np.abs(bend_deg) if self.use_absolute_metric else bend_deg
            joint_mask = metric_deg > self.max_angle_deg
            if self.enforce_min_bound:
                joint_mask |= metric_deg < self.min_angle_deg
            mask[:, joint_id] = joint_mask
            joint_min = float(np.min(metric_deg))
            joint_max = float(np.max(metric_deg))
            details["joint_ranges_deg"][joint_name] = {
                "metric": "tbs_abs_bend_deg" if self.use_absolute_metric else "tbs_bend_deg",
                "signed_bend_min": float(np.min(bend_deg)),
                "signed_bend_max": float(np.max(bend_deg)),
                "min": joint_min,
                "max": joint_max,
            }
            if joint_max > self.max_angle_deg:
                reasons.append(f"{joint_name} bend max {joint_max:.1f}deg > {self.max_angle_deg:.1f}deg")
            if self.enforce_min_bound and joint_min < self.min_angle_deg:
                reasons.append(f"{joint_name} bend min {joint_min:.1f}deg < {self.min_angle_deg:.1f}deg")
        passed = not reasons
        return CheckResult(
            is_valid=passed,
            invalid_reason="; ".join(reasons) if reasons else f"{self.name} in range",
            invalid_mask=mask,
            details=details,
            severity="pass" if passed else "fail",
        )


class KneeXChecker(_JointXRangeChecker):
    name = "knee_x"
    l_joint_id = 4
    r_joint_id = 5
    l_joint_name = "L_Knee"
    r_joint_name = "R_Knee"
    use_absolute_metric = True
    enforce_min_bound = False

    def __init__(self, min_angle_deg: float = -10.0, max_angle_deg: float = 160.0, device: str = "cuda") -> None:
        super().__init__(min_angle_deg=min_angle_deg, max_angle_deg=max_angle_deg, device=device)


class AnkleXChecker(_JointXRangeChecker):
    name = "ankle_x"
    l_joint_id = 7
    r_joint_id = 8
    l_joint_name = "L_Ankle"
    r_joint_name = "R_Ankle"
    use_absolute_metric = True
    enforce_min_bound = False

    def __init__(self, min_angle_deg: float = -60.0, max_angle_deg: float = 80.0, device: str = "cuda") -> None:
        super().__init__(min_angle_deg=min_angle_deg, max_angle_deg=max_angle_deg, device=device)


class NeckChecker(BaseQualityChecker):
    name = "neck"

    def __init__(
        self,
        neck_x_deg: float = 95.0,
        neck_y_deg: float = 80.0,
        neck_z_deg: float = 70.0,
        head_x_deg: float = 85.0,
        head_y_deg: float = 100.0,
        head_z_deg: float = 60.0,
        min_consecutive_frames: int = 4,
        device: str = "cuda",
    ) -> None:
        super().__init__(body_model=None, device=device)
        self.min_consecutive_frames = max(1, int(min_consecutive_frames))
        self.thresholds = {
            12: {
                "name": "Neck",
                "bend": float(neck_x_deg),
                "twist": float(neck_y_deg),
                "spread": float(neck_z_deg),
            },
            15: {
                "name": "Head",
                "bend": float(head_x_deg),
                "twist": float(head_y_deg),
                "spread": float(head_z_deg),
            },
        }

    def get_required_keys(self) -> list:
        return ["poses"]

    @staticmethod
    def _keep_consecutive(mask: np.ndarray, min_run: int) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if min_run <= 1 or mask.size == 0:
            return mask
        filtered = np.zeros_like(mask, dtype=bool)
        start = None
        for idx, value in enumerate(mask):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                if idx - start >= min_run:
                    filtered[start:idx] = True
                start = None
        if start is not None and mask.size - start >= min_run:
            filtered[start:] = True
        return filtered

    def check(self, motion) -> CheckResult:
        data = self.load_motion(motion)
        err = self.validate_motion_dict(data)
        if err:
            return CheckResult(is_valid=False, invalid_reason=err, invalid_mask=None, details={"error": err})
        poses_3d = normalize_poses_array(np.asarray(data["poses"]), num_joints=NUM_BODY_JOINTS)
        num_frames = int(poses_3d.shape[0])
        mask = np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool)
        reasons = []
        details: Dict[str, Any] = {"joint_metrics": {}}
        for joint_id, cfg in self.thresholds.items():
            tbs = extract_joint_tbs_metrics(poses_3d[:, joint_id, :], joint_id)
            bend_deg = np.abs(tbs["bend_deg"])
            twist_deg = np.abs(tbs["twist_deg"])
            spread_deg = np.abs(tbs["spread_deg"])
            bend_mask = self._keep_consecutive(bend_deg > cfg["bend"], self.min_consecutive_frames)
            twist_mask = self._keep_consecutive(twist_deg > cfg["twist"], self.min_consecutive_frames)
            spread_mask = self._keep_consecutive(spread_deg > cfg["spread"], self.min_consecutive_frames)
            joint_mask = bend_mask | twist_mask | spread_mask
            mask[:, joint_id] = joint_mask
            metrics = {
                "bend_max_deg": float(np.max(bend_deg)),
                "twist_max_deg": float(np.max(twist_deg)),
                "spread_max_deg": float(np.max(spread_deg)),
                "bend_fail_frames": int(np.count_nonzero(bend_mask)),
                "twist_fail_frames": int(np.count_nonzero(twist_mask)),
                "spread_fail_frames": int(np.count_nonzero(spread_mask)),
                "thresholds_deg": {
                    "bend": cfg["bend"],
                    "twist": cfg["twist"],
                    "spread": cfg["spread"],
                },
                "min_consecutive_frames": self.min_consecutive_frames,
                "coordinate_system": "tbs",
            }
            details["joint_metrics"][cfg["name"]] = metrics
            if metrics["bend_fail_frames"] > 0:
                reasons.append(f"{cfg['name']} bend {metrics['bend_max_deg']:.1f}deg > {cfg['bend']:.1f}deg")
            if metrics["twist_fail_frames"] > 0:
                reasons.append(f"{cfg['name']} twist {metrics['twist_max_deg']:.1f}deg > {cfg['twist']:.1f}deg")
            if metrics["spread_fail_frames"] > 0:
                reasons.append(f"{cfg['name']} spread {metrics['spread_max_deg']:.1f}deg > {cfg['spread']:.1f}deg")
        passed = not reasons
        return CheckResult(
            is_valid=passed,
            invalid_reason="; ".join(reasons) if reasons else "Neck/head TBS rotations in range",
            invalid_mask=mask,
            details=details,
            severity="pass" if passed else "fail",
        )


class _SpineXChecker(BaseQualityChecker):
    joint_id: int = -1
    joint_name: str = ""
    name = "spine_x"

    def __init__(self, threshold_deg: float, device: str = "cuda") -> None:
        super().__init__(body_model=None, device=device)
        self.threshold_deg = float(threshold_deg)

    def get_required_keys(self) -> list:
        return ["poses"]

    def check(self, motion) -> CheckResult:
        data = self.load_motion(motion)
        err = self.validate_motion_dict(data)
        if err:
            return CheckResult(is_valid=False, invalid_reason=err, invalid_mask=None, details={"error": err})
        poses_3d = normalize_poses_array(np.asarray(data["poses"]), num_joints=NUM_BODY_JOINTS)
        num_frames = int(poses_3d.shape[0])
        tbs = extract_joint_tbs_metrics(poses_3d[:, self.joint_id, :], self.joint_id)
        bend_deg = np.abs(tbs["bend_deg"])
        frame_mask = bend_deg > self.threshold_deg
        passed = not bool(frame_mask.any())
        return CheckResult(
            is_valid=passed,
            invalid_reason=(
                f"{self.joint_name} bend {float(np.max(bend_deg)):.1f}deg > {self.threshold_deg:.1f}deg"
                if not passed
                else f"{self.joint_name} TBS bend in range"
            ),
            invalid_mask=_frames_mask(frame_mask, [self.joint_id]),
            details={
                "joint_id": self.joint_id,
                "joint_name": self.joint_name,
                "metric": "tbs_abs_bend_deg",
                "bend_angle_max_deg": float(np.max(bend_deg)),
                "signed_bend_min_deg": float(np.min(tbs["bend_deg"])),
                "signed_bend_max_deg": float(np.max(tbs["bend_deg"])),
                "threshold_deg": self.threshold_deg,
                "coordinate_system": "tbs",
            },
            severity="pass" if passed else "fail",
        )


class SpineChecker(_SpineXChecker):
    name = "spine"
    joint_id = 3
    joint_name = "Spine1"

    def __init__(self, threshold_deg: float = 90.0, device: str = "cuda") -> None:
        super().__init__(threshold_deg=threshold_deg, device=device)


class Spine1Checker(_SpineXChecker):
    name = "spine1"
    joint_id = 6
    joint_name = "Spine2"

    def __init__(self, threshold_deg: float = 60.0, device: str = "cuda") -> None:
        super().__init__(threshold_deg=threshold_deg, device=device)


class Spine2Checker(_SpineXChecker):
    name = "spine2"
    joint_id = 9
    joint_name = "Spine3"

    def __init__(self, threshold_deg: float = 45.0, device: str = "cuda") -> None:
        super().__init__(threshold_deg=threshold_deg, device=device)
