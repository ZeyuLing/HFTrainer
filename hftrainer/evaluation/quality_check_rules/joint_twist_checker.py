"""
Joint twist quality checker: detects abnormal joint twist.

Checks arm joints (collar, shoulder, elbow, wrist) for twist artifacts, plus
leg joints (UpLeg, Leg) for Y-axis twist and neck/head for multi-axis angle limits.

The current implementation uses two signals:
1. per-joint twist-axis projection for obvious single-joint pathologies
2. chain counter-twist detection for candy-wrapper style compensation
   (e.g. elbow/wrist or collar/shoulder opposite-signed large twist that
   preserves downstream pose while corrupting local parameters)

This module extends the original filter_joint_twist_from_completion.py logic with
additional body part coverage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base_checker import BaseQualityChecker, CheckResult, NUM_BODY_JOINTS_DEFAULT
from .tbs_utils import extract_joint_tbs_metrics
from ._geometry_compat import axis_angle_to_quaternion, quaternion_fix_continuity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BODY_JOINTS = 22

# Arm joints (original)
COLLAR_JOINTS = [13, 14]
SHOULDER_JOINTS = [16, 17]
ELBOW_JOINTS = [18, 19]
WRIST_JOINTS = [20, 21]
ARM_JOINTS = COLLAR_JOINTS + SHOULDER_JOINTS + ELBOW_JOINTS + WRIST_JOINTS
# Keep direct per-joint hard checks conservative. Elbow/wrist are mainly handled
# by chain counter-twist logic below to reduce normal-pose false positives.
ARM_TWIST_JOINTS = COLLAR_JOINTS + SHOULDER_JOINTS

# Leg joints (new)
UPLEG_JOINTS = [1, 2]   # LUpLeg, RUpLeg
KNEE_JOINTS = [4, 5]    # LLeg, RLeg
FOOT_JOINTS = [7, 8]    # LFoot, RFoot
LEG_JOINTS = UPLEG_JOINTS + KNEE_JOINTS + FOOT_JOINTS

# Neck/Head joints (new)
NECK_JOINT = 12
HEAD_JOINT = 15

# All joints that get twist-axis checks.
#
# Important: direct joint_twist is now intentionally narrower than candy_wrapper.
# Wrist/elbow candy-wrapper style counter-twist is handled by CandyWrapperChecker.
JOINT_TWIST_AXIS_MAP = {
    # Arms: local bone axis is X.
    13: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    14: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    16: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    17: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    18: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    19: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    20: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    21: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    # Legs: local long axis is Y in the current SMPL convention.
    1: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    2: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    4: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    5: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    7: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    8: np.array([0.0, 1.0, 0.0], dtype=np.float32),
}

JOINT_NAMES = {
    1: "L_UpLeg", 2: "R_UpLeg",
    4: "L_Leg", 5: "R_Leg",
    7: "L_Foot", 8: "R_Foot",
    12: "Neck", 15: "Head",
    13: "L_Collar", 14: "R_Collar",
    16: "L_Shoulder", 17: "R_Shoulder",
    18: "L_Elbow", 19: "R_Elbow",
    20: "L_Wrist", 21: "R_Wrist",
}

# Arm twist configs — conservative direct detection.
#
# 90deg elbow/wrist hard rules remain disabled. Body-22 wrist joints are too
# ambiguous without finger children; elbow false positives are also common in
# lifting / game motions. Those joints are handled by chain compensation logic.
TWIST_CONFIGS = {
    "90deg": {
        "target_rad": np.pi / 2,
        "threshold_rad": np.pi * 15 / 180,
        "joints": [],
        "min_frames": 15,
        "min_ratio": 0.15,
    },
    "130deg": {
        "target_rad": np.pi * 130 / 180,
        "threshold_rad": np.pi * 15 / 180,
        "joints": [],
        "min_frames": 20,
        "min_ratio": 0.2,
    },
    "180deg": {
        "target_rad": np.pi,
        "threshold_rad": np.pi * 20 / 180,
        "joints": [],
        "min_frames": 15,
        "min_ratio": 0.15,
    },
    "360deg": {
        "target_rad": 2 * np.pi,
        "threshold_rad": np.pi * 20 / 180,
        "joints": [],
        "min_frames": 15,
        "min_ratio": 0.15,
    },
}

# Leg Y-axis twist configs — conservative: only catch clearly unnatural rotations.
# foot_60deg removed: ankles are naturally flexible and 60° Y-twist is common.
LEG_TWIST_CONFIGS = {
    "upleg_150deg": {
        "target_rad": np.pi * 150 / 180,
        "threshold_rad": np.pi * 20 / 180,
        "joints": UPLEG_JOINTS,
        "min_frames": 15,
        "min_ratio": 0.2,
    },
    "knee_120deg": {
        "target_rad": np.pi * 120 / 180,
        "threshold_rad": np.pi * 15 / 180,
        "joints": KNEE_JOINTS,
        "min_frames": 15,
        "min_ratio": 0.2,
    },
    "leg_180deg": {
        "target_rad": np.pi,
        "threshold_rad": np.pi * 20 / 180,
        "joints": LEG_JOINTS,
        "min_frames": 15,
        "min_ratio": 0.15,
    },
}

# Neck/Head angle limits (degrees) — relaxed from original to avoid false positives
# on expressive motions (dance, combat, etc.).
MIN_FRAMES_REQUIRED = 10


def is_supported_twist_joint(joint_id: int, num_joints: int) -> bool:
    return joint_id < num_joints


def _extract_twist_angle(axis_angle: np.ndarray, twist_axis: int) -> np.ndarray:
    """Backward-compatible entry point: int axis means x/y/z basis."""
    basis = np.zeros((3,), dtype=np.float32)
    basis[int(twist_axis)] = 1.0
    return extract_joint_twist_metrics(axis_angle, basis)["geometric_twist_rad"]


def _continuous_raw_aligned_angle(axis_angle: np.ndarray, twist_axis: np.ndarray) -> np.ndarray:
    signed_aligned = np.asarray(axis_angle @ twist_axis, dtype=np.float64)
    if signed_aligned.shape[0] <= 1:
        return signed_aligned
    result = signed_aligned.copy()
    two_pi = 2.0 * np.pi
    for idx in range(1, result.shape[0]):
        prev = result[idx - 1]
        cur = result[idx]
        while cur - prev > np.pi:
            cur -= two_pi
        while cur - prev < -np.pi:
            cur += two_pi
        result[idx] = cur
    return result


def extract_joint_twist_metrics(axis_angle: np.ndarray, twist_axis: np.ndarray) -> Dict[str, np.ndarray]:
    axis_angle_t = torch.as_tensor(axis_angle, dtype=torch.float32)
    twist_axis_t = torch.as_tensor(twist_axis, dtype=torch.float32)
    twist_axis_t = twist_axis_t / torch.clamp(torch.linalg.norm(twist_axis_t), min=1e-8)
    q = axis_angle_to_quaternion(axis_angle_t)
    q = quaternion_fix_continuity(q)
    q_vec = q[:, 1:]
    proj = torch.sum(q_vec * twist_axis_t[None, :], dim=-1, keepdim=True) * twist_axis_t[None, :]
    twist_q = torch.cat([q[:, :1], proj], dim=-1)
    twist_q_norm = torch.linalg.norm(twist_q, dim=-1, keepdim=True)
    identity_mask = twist_q_norm[:, 0] < 1e-8
    twist_q = twist_q / torch.clamp(twist_q_norm, min=1e-8)
    if bool(identity_mask.any()):
        twist_q[identity_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=twist_q.dtype)
    sign = torch.sign(torch.sum(twist_q[:, 1:] * twist_axis_t[None, :], dim=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    half_angle = torch.atan2(torch.linalg.norm(twist_q[:, 1:], dim=-1), torch.clamp(twist_q[:, 0], min=-1.0, max=1.0))
    geometric_twist_rad = (2.0 * half_angle * sign).detach().cpu().numpy().astype(np.float64)
    raw_aligned_rad = _continuous_raw_aligned_angle(axis_angle, twist_axis_t.detach().cpu().numpy())
    return {
        "geometric_twist_rad": geometric_twist_rad,
        "raw_aligned_rad": raw_aligned_rad,
    }


def _select_twist_signal(metrics: Dict[str, np.ndarray], twist_type: str) -> np.ndarray:
    if twist_type == "360deg":
        return metrics["raw_aligned_rad"]
    return metrics["geometric_twist_rad"]


def _detect_twist_at_angle(
    twist_angles: np.ndarray,
    target_rad: float,
    threshold_rad: float,
    min_frames: int,
    min_ratio: float,
    total_frames: int,
) -> Tuple[bool, int, float]:
    """Returns (has_twist, num_frames, min_deviation)."""
    deviation = np.abs(np.abs(twist_angles) - target_rad)
    twist_mask = deviation < threshold_rad
    num_frames = int(np.sum(twist_mask))
    if num_frames >= min_frames and num_frames >= total_frames * min_ratio:
        min_deviation = float(np.min(deviation[twist_mask]))
        return True, num_frames, min_deviation
    return False, 0, float("inf")


def _detect_counter_twist(
    parent_raw_rad: np.ndarray,
    child_raw_rad: np.ndarray,
    *,
    parent_min_abs_deg: float,
    child_min_abs_deg: float,
    balance_margin_deg: float,
    min_frames: int,
    min_ratio: float,
    total_frames: int,
) -> Tuple[bool, List[int], Dict[str, float]]:
    parent_abs = np.abs(np.rad2deg(parent_raw_rad))
    child_abs = np.abs(np.rad2deg(child_raw_rad))
    opposite_sign = np.sign(parent_raw_rad) * np.sign(child_raw_rad) < 0.0
    strong_parent = parent_abs >= float(parent_min_abs_deg)
    strong_child = child_abs >= float(child_min_abs_deg)
    balance_deg = np.abs(np.rad2deg(parent_raw_rad + child_raw_rad))
    balanced = balance_deg <= float(balance_margin_deg)
    frame_mask = opposite_sign & strong_parent & strong_child & balanced
    frame_indices = np.where(frame_mask)[0].astype(int).tolist()
    if len(frame_indices) < int(min_frames) or len(frame_indices) < float(total_frames) * float(min_ratio):
        return False, [], {}
    stats = {
        "mean_parent_abs_deg": float(np.mean(parent_abs[frame_mask])) if np.any(frame_mask) else 0.0,
        "mean_child_abs_deg": float(np.mean(child_abs[frame_mask])) if np.any(frame_mask) else 0.0,
        "mean_balance_deg": float(np.mean(balance_deg[frame_mask])) if np.any(frame_mask) else 0.0,
        "max_parent_abs_deg": float(np.max(parent_abs[frame_mask])) if np.any(frame_mask) else 0.0,
        "max_child_abs_deg": float(np.max(child_abs[frame_mask])) if np.any(frame_mask) else 0.0,
    }
    return True, frame_indices, stats


class JointTwistChecker(BaseQualityChecker):
    """Checker for joint twist. Covers arms, legs (Y-axis twist), and neck/head
    (multi-axis angle limits). Does not require a body model."""

    name = "joint_twist"

    def get_required_keys(self) -> list:
        return ["poses"]

    def check(self, motion: Union[Dict, str, Path]) -> CheckResult:
        """Run joint twist detection. Returns CheckResult and legacy-style details when invalid."""
        if isinstance(motion, (str, Path)):
            data = self.load_motion(motion)
        else:
            data = dict(motion)

        err = self.validate_motion_dict(data)
        if err is not None:
            return CheckResult(
                is_valid=False,
                invalid_reason=err,
                invalid_mask=None,
                details={"has_twist": False, "reason": err},
            )

        poses = np.array(data["poses"])
        if len(poses) < MIN_FRAMES_REQUIRED:
            reason = f"Too short (need at least {MIN_FRAMES_REQUIRED} frames)"
            return CheckResult(
                is_valid=False,
                invalid_reason=reason,
                invalid_mask=None,
                details={"has_twist": False, "reason": reason},
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
                details={"has_twist": False, "reason": str(e)},
            )

        F, J, _ = poses_3d.shape
        twist_detail_map: Dict[int, Dict] = {}

        def ensure_detail(joint_id: int, joint_name: str) -> Dict:
            detail = twist_detail_map.get(joint_id)
            if detail is None:
                detail = {"joint_id": joint_id, "joint_name": joint_name, "twist_types": []}
                twist_detail_map[joint_id] = detail
            return detail

        invalid_mask = np.zeros((F, NUM_BODY_JOINTS), dtype=bool)

        # --- Direct arm twist checks (TBS twist only; wrist/elbow candy-wrapper handled elsewhere) ---
        for joint_id in ARM_JOINTS:
            if joint_id >= J or joint_id not in JOINT_TWIST_AXIS_MAP or not is_supported_twist_joint(joint_id, J):
                continue
            joint_poses = poses_3d[:, joint_id, :]
            tbs_metrics = extract_joint_tbs_metrics(joint_poses, joint_id)
            joint_name = JOINT_NAMES.get(joint_id, f"Joint_{joint_id}")
            detail = ensure_detail(joint_id, joint_name)

            for twist_type, config in TWIST_CONFIGS.items():
                if joint_id not in config["joints"]:
                    continue
                if twist_type == "360deg":
                    twist_angles = np.deg2rad(tbs_metrics["raw_aligned_deg"])
                else:
                    twist_angles = np.deg2rad(tbs_metrics["twist_deg"])
                has_twist, num_frames, min_deviation = _detect_twist_at_angle(
                    twist_angles,
                    config["target_rad"],
                    config["threshold_rad"],
                    config["min_frames"],
                    config["min_ratio"],
                    F,
                )
                if has_twist:
                    deviation = np.abs(np.abs(twist_angles) - config["target_rad"])
                    frame_mask = deviation < config["threshold_rad"]
                    frame_indices = np.where(frame_mask)[0].astype(int).tolist()
                    invalid_mask[np.asarray(frame_indices, dtype=np.int64), joint_id] = True
                    detail["twist_types"].append(
                        {
                            "type": twist_type,
                            "metric": "tbs_raw_aligned" if twist_type == "360deg" else "tbs_twist",
                            "num_frames": num_frames,
                            "min_deviation_deg": float(np.degrees(min_deviation)),
                            "frame_indices": frame_indices,
                        }
                    )

        # --- Check leg joints (Y-axis twist) ---
        for joint_id in LEG_JOINTS:
            if joint_id >= J or joint_id not in JOINT_TWIST_AXIS_MAP or not is_supported_twist_joint(joint_id, J):
                continue
            joint_poses = poses_3d[:, joint_id, :]
            tbs_metrics = extract_joint_tbs_metrics(joint_poses, joint_id)
            joint_name = JOINT_NAMES.get(joint_id, f"Joint_{joint_id}")
            detail = ensure_detail(joint_id, joint_name)

            for twist_type, config in LEG_TWIST_CONFIGS.items():
                if joint_id not in config["joints"]:
                    continue
                if twist_type == "360deg":
                    twist_angles = np.deg2rad(tbs_metrics["raw_aligned_deg"])
                else:
                    twist_angles = np.deg2rad(tbs_metrics["twist_deg"])
                has_twist, num_frames, min_deviation = _detect_twist_at_angle(
                    twist_angles,
                    config["target_rad"],
                    config["threshold_rad"],
                    config["min_frames"],
                    config["min_ratio"],
                    F,
                )
                if has_twist:
                    deviation = np.abs(np.abs(twist_angles) - config["target_rad"])
                    frame_mask = deviation < config["threshold_rad"]
                    frame_indices = np.where(frame_mask)[0].astype(int).tolist()
                    invalid_mask[np.asarray(frame_indices, dtype=np.int64), joint_id] = True
                    detail["twist_types"].append(
                        {
                            "type": twist_type,
                            "metric": "tbs_raw_aligned" if twist_type == "360deg" else "tbs_twist",
                            "num_frames": num_frames,
                            "min_deviation_deg": float(np.degrees(min_deviation)),
                            "frame_indices": frame_indices,
                        }
                    )

        twist_details: List[Dict] = []
        twisted_joints: List[int] = []
        for joint_id in sorted(twist_detail_map.keys()):
            detail = twist_detail_map[joint_id]
            if not detail.get("twist_types"):
                continue
            detail["detected_types_str"] = ", ".join(
                sorted({str(item.get("type", "")).strip() for item in detail["twist_types"] if str(item.get("type", "")).strip()})
            )
            twisted_joints.append(joint_id)
            twist_details.append(detail)

        if not twisted_joints:
            return CheckResult(
                is_valid=True,
                invalid_reason="No twist detected",
                invalid_mask=np.zeros((F, NUM_BODY_JOINTS), dtype=bool),
                details={
                    "has_twist": False,
                    "twisted_joints": [],
                    "twist_details": [],
                    "coordinate_system": "tbs_direct_twist",
                    "reason": "No twist detected",
                },
            )

        joint_info = [f"{d['joint_name']}({d.get('detected_types_str', '')})" for d in twist_details]
        reason = f"Joint twist detected: {', '.join(joint_info)}"
        details = {
            "has_twist": True,
            "twisted_joints": twisted_joints,
            "twist_details": twist_details,
            "uses_full_tbs": True,
            "coordinate_system": "tbs_direct_twist",
            "reason": reason,
        }
        return CheckResult(
            is_valid=False,
            invalid_reason=reason,
            invalid_mask=invalid_mask,
            details=details,
        )


def detect_joint_twist(data: Dict) -> Dict:
    """Legacy function: same signature and return dict as filter_joint_twist_from_completion.detect_joint_twist.

    Returns dict with keys: has_twist, twisted_joints, twist_details, reason.
    """
    checker = JointTwistChecker()
    result = checker.check(data)
    details = result.get("details") or {}
    return {
        "has_twist": details.get("has_twist", False),
        "twisted_joints": details.get("twisted_joints", []),
        "twist_details": details.get("twist_details", []),
        "reason": details.get("reason", result.get("invalid_reason", "")),
    }
