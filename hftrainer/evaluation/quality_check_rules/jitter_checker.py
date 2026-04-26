"""
Jitter quality checker: detects motion jitter (repeated velocity direction flips).

Jitter is defined as: for one or more joints, over a time window, velocity direction
repeatedly reverses (angle between consecutive velocity vectors near 180°) or
has a high proportion of opposite-direction frames.

This module mirrors the logic in scripts/m2m/filter_data/filter_jitter_from_completion.py
so that using JitterChecker produces identical results to the original script.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from .base_checker import BaseQualityChecker, CheckResult, NUM_BODY_JOINTS_DEFAULT, normalize_betas_array

# Optional FK model
from ._model_compat import SmplxLiteJ24

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BODY_JOINTS = 22
END_EFFECTOR_JOINTS = {7, 8, 10, 11, 20, 21}
# Angle threshold: count frame as "opposite" when angle > 150 deg (lowered from 165
# to catch moderate jitter that was previously missed).
ANGLE_THRESHOLD_DEG = 150.0
ANGLE_THRESHOLD_RAD = np.deg2rad(ANGLE_THRESHOLD_DEG)
MIN_VELOCITY_THRESHOLD = 0.001
# Minimum per-frame FK joint displacement (m) to count the frame as jittery.
# Filters out tiny imperceptible oscillations that match the angular pattern.
MIN_DISPLACEMENT_M = 0.005
MIN_BODY_DISPLACEMENT_M = 0.003
MIN_CONSECUTIVE_FRAMES = 4
JITTER_WINDOW_SIZE = 10
# Use 50% overlapping windows (stride = window_size // 2) so jitter bursts
# spanning a window boundary are not missed.
JITTER_WINDOW_STRIDE = JITTER_WINDOW_SIZE // 2
MIN_JITTER_RATIO = 0.6
MIN_FRAMES_REQUIRED = JITTER_WINDOW_SIZE + 2
ACCEL_JITTER_STABLE_ROOT_VEL_M_PER_FRAME = 0.01
ACCEL_JITTER_SPIKE_M_PER_FRAME2 = 0.008
ACCEL_JITTER_MIN_SPIKE_FRAMES = 12
ACCEL_JITTER_MAX_VEL_M_PER_FRAME = 0.015
ACCEL_JITTER_MIN_RATIO = 1.2
ACCEL_JITTER_BORDERLINE_SPIKE_M_PER_FRAME2 = 0.0055
ACCEL_JITTER_BORDERLINE_MIN_SPIKE_FRAMES = 6
ACCEL_JITTER_BORDERLINE_MAX_VEL_M_PER_FRAME = 0.02
ACCEL_JITTER_BORDERLINE_MIN_RATIO = 0.9

JOINT_NAMES = [
    "MidHip",
    "LUpLeg",
    "RUpLeg",
    "spine",
    "LLeg",
    "RLeg",
    "spine1",
    "LFoot",
    "RFoot",
    "spine2",
    "LToeBase",
    "RToeBase",
    "Neck",
    "LShoulder",
    "RShoulder",
    "Head",
    "LArm",
    "RArm",
    "LForeArm",
    "RForeArm",
    "LHand",
    "RHand",
]


def _get_joints_from_pose(
    poses: np.ndarray,
    trans: np.ndarray,
    betas: Optional[np.ndarray] = None,
    body_model: Optional[Any] = None,
    device: str = "cpu",
) -> np.ndarray:
    """FK: compute world-space joint positions from poses and trans.

    Args:
        poses: (F, J, 3) axis-angle, J >= 22.
        trans: (F, 3) root translation.
        betas: (1, 16) or None for zero betas.
        body_model: SmplxLiteJ24 instance.
        device: Device for tensors.

    Returns:
        joints: (F, J, 3) with J from model (24); caller may slice to 22.
    """
    if body_model is None or SmplxLiteJ24 is None:
        raise RuntimeError("JitterChecker requires body_model (SmplxLiteJ24) for FK.")
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


def _compute_velocity_angles(
    joint_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-joint velocity vectors, angles between consecutive velocities,
    and per-frame displacement magnitudes.

    Args:
        joint_positions: (F, J, 3).

    Returns:
        velocities: (F-1, J, 3).
        angles: (F-2, J) in degrees.
        displacement_magnitudes: (F-2, J) magnitude of v2 displacement (used for
            amplitude gating — the displacement at the frame where the direction
            change is measured).
    """
    velocities = np.diff(joint_positions, axis=0)  # (F-1, J, 3)
    vel_norms = np.linalg.norm(velocities, axis=2)  # (F-1, J)
    v1 = velocities[:-1]
    v2 = velocities[1:]
    n1 = vel_norms[:-1]
    n2 = vel_norms[1:]
    displacement_magnitudes = n2.copy()

    denom = n1 * n2
    valid = (n1 >= MIN_VELOCITY_THRESHOLD) & (n2 >= MIN_VELOCITY_THRESHOLD) & (denom > 1e-12)
    dot = np.sum(v1 * v2, axis=2)
    cos_angle = np.zeros_like(dot)
    cos_angle[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    angles = np.zeros_like(dot)
    angles[valid] = np.rad2deg(np.arccos(cos_angle[valid]))
    return velocities, angles, displacement_magnitudes


def _min_displacement_gate(joint_id: int) -> float:
    return MIN_DISPLACEMENT_M if joint_id in END_EFFECTOR_JOINTS else MIN_BODY_DISPLACEMENT_M


def _make_root_relative_positions(joint_positions: np.ndarray) -> np.ndarray:
    root_relative = joint_positions.copy()
    root_relative[:, 1:, :] -= joint_positions[:, :1, :]
    return root_relative


def _detect_jitter_in_window(
    angles: np.ndarray,
    displacement_magnitudes: np.ndarray,
    window_start: int,
    window_size: int,
    source_label: str = "world",
    skip_root_joint: bool = False,
) -> Tuple[bool, Dict]:
    """Detect jitter in a single window. Returns (has_jitter, details_dict).

    A frame is counted as "opposite direction" only if:
      1. The angle exceeds ANGLE_THRESHOLD_DEG, AND
      2. The displacement magnitude exceeds MIN_DISPLACEMENT_M (amplitude gate).
    """
    F2, J = angles.shape
    window_end = min(window_start + window_size, F2)
    if window_end - window_start < MIN_CONSECUTIVE_FRAMES:
        return False, {}
    window_angles = angles[window_start:window_end]
    window_displacements = displacement_magnitudes[window_start:window_end]
    jitter_joints = []
    jitter_details = []
    for j in range(J):
        if skip_root_joint and j == 0:
            continue
        joint_angles = window_angles[:, j]
        joint_disps = window_displacements[:, j]
        min_displacement = _min_displacement_gate(j)
        # A frame counts as "opposite" only if angle is large AND displacement is non-trivial
        opposite_mask = (joint_angles > ANGLE_THRESHOLD_DEG) & (joint_disps > min_displacement)
        opposite_count = int(np.sum(opposite_mask))
        opposite_ratio = opposite_count / len(joint_angles)
        consecutive_opposite = 0
        max_consecutive = 0
        for is_opp in opposite_mask:
            if is_opp:
                consecutive_opposite += 1
                max_consecutive = max(max_consecutive, consecutive_opposite)
            else:
                consecutive_opposite = 0
        has_jitter = opposite_ratio >= MIN_JITTER_RATIO or max_consecutive >= MIN_CONSECUTIVE_FRAMES
        if has_jitter:
            opposite_indices = np.where(opposite_mask)[0].astype(int)
            frame_indices = (opposite_indices + window_start + 1).tolist()
            jitter_joints.append(j)
            jitter_details.append(
                {
                    "joint_id": j,
                    "opposite_ratio": float(opposite_ratio),
                    "max_consecutive": max_consecutive,
                    "avg_angle": float(np.mean(joint_angles)),
                    "max_angle": float(np.max(joint_angles)),
                    "source": source_label,
                    "min_displacement_gate_m": float(min_displacement),
                    "frame_indices": frame_indices,
                }
            )
    if jitter_joints:
        return True, {
            "jitter_joints": jitter_joints,
            "jitter_details": jitter_details,
            "window_start": window_start,
            "window_end": window_end,
            "source": source_label,
        }
    return False, {}


def _collect_direction_flip_windows(
    joint_positions: np.ndarray,
    source_label: str,
    skip_root_joint: bool = False,
) -> Tuple[list, set]:
    velocities, angles, displacement_magnitudes = _compute_velocity_angles(joint_positions)
    F2 = angles.shape[0]
    del velocities
    jitter_windows = []
    jitter_joints = set()
    step_size = JITTER_WINDOW_STRIDE
    for window_start in range(0, max(0, F2 - JITTER_WINDOW_SIZE + 1), step_size):
        has_jitter, details = _detect_jitter_in_window(
            angles,
            displacement_magnitudes,
            window_start,
            JITTER_WINDOW_SIZE,
            source_label=source_label,
            skip_root_joint=skip_root_joint,
        )
        if has_jitter:
            jitter_windows.append(details)
            jitter_joints.update(details["jitter_joints"])
    return jitter_windows, jitter_joints


def _detect_acceleration_spike_jitter(joint_positions: np.ndarray) -> Dict[str, Any]:
    """Detect sparse but obvious jitter using root-relative acceleration spikes.

    The original direction-flip heuristic misses cases where a foot/toe keeps
    flickering in place but does not sustain long runs of 180-degree reversals.
    This alternative path looks for joints that:
    1. live in near-stationary root segments,
    2. have low per-frame displacement,
    3. but repeatedly show high acceleration spikes.
    """
    if joint_positions.shape[0] < 3:
        return {"joints": [], "details": []}

    root_rel = joint_positions - joint_positions[:, :1, :]
    velocities = np.diff(root_rel, axis=0)  # (F-1, J, 3)
    accelerations = np.diff(velocities, axis=0)  # (F-2, J, 3)
    vel_mag = np.linalg.norm(velocities, axis=2)  # (F-1, J)
    acc_mag = np.linalg.norm(accelerations, axis=2)  # (F-2, J)
    root_vel = np.linalg.norm(np.diff(joint_positions[:, 0, :], axis=0), axis=1)  # (F-1,)
    stable_mask = root_vel[1:] <= ACCEL_JITTER_STABLE_ROOT_VEL_M_PER_FRAME  # align to accel domain (F-2,)
    stable_frame_indices = np.where(stable_mask)[0]
    if stable_frame_indices.size < ACCEL_JITTER_MIN_SPIKE_FRAMES:
        return {"joints": [], "details": []}

    accel_joints = []
    accel_details = []
    borderline_joints = []
    borderline_details = []
    for joint_id in range(acc_mag.shape[1]):
        stable_acc = acc_mag[:, joint_id][stable_mask]
        stable_vel = vel_mag[1:, joint_id][stable_mask]
        if stable_acc.size < ACCEL_JITTER_BORDERLINE_MIN_SPIKE_FRAMES:
            continue
        acc_p95 = float(np.percentile(stable_acc, 95))
        vel_p95 = float(np.percentile(stable_vel, 95))
        ratio = float(acc_p95 / max(vel_p95, 1e-6))
        local_spike_indices = np.where(stable_acc >= ACCEL_JITTER_SPIKE_M_PER_FRAME2)[0]
        spike_count = int(local_spike_indices.size)
        if (
            acc_p95 < ACCEL_JITTER_SPIKE_M_PER_FRAME2
            or spike_count < ACCEL_JITTER_MIN_SPIKE_FRAMES
            or vel_p95 > ACCEL_JITTER_MAX_VEL_M_PER_FRAME
            or ratio < ACCEL_JITTER_MIN_RATIO
        ):
            if (
                acc_p95 < ACCEL_JITTER_BORDERLINE_SPIKE_M_PER_FRAME2
                or spike_count < ACCEL_JITTER_BORDERLINE_MIN_SPIKE_FRAMES
                or vel_p95 > ACCEL_JITTER_BORDERLINE_MAX_VEL_M_PER_FRAME
                or ratio < ACCEL_JITTER_BORDERLINE_MIN_RATIO
            ):
                continue
            bucket = borderline_details
            borderline_joints.append(joint_id)
        else:
            bucket = accel_details
            accel_joints.append(joint_id)
        spike_frames = (stable_frame_indices[local_spike_indices] + 1).astype(int).tolist()
        bucket.append(
            {
                "joint_id": int(joint_id),
                "joint_name": JOINT_NAMES[joint_id] if joint_id < len(JOINT_NAMES) else f"Joint_{joint_id}",
                "spike_count": spike_count,
                "spike_frames": spike_frames,
                "acceleration_p95_m": acc_p95,
                "velocity_p95_m": vel_p95,
                "accel_to_vel_ratio": ratio,
            }
        )

    return {
        "joints": accel_joints,
        "details": accel_details,
        "borderline_joints": borderline_joints,
        "borderline_details": borderline_details,
    }


class JitterChecker(BaseQualityChecker):
    """Checker for motion jitter (velocity direction flips). Requires body model for FK."""

    name = "jitter"

    def __init__(
        self,
        body_model: Optional[Any] = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(body_model=body_model, device=device)
        if self.body_model is None and SmplxLiteJ24 is not None:
            self.body_model = SmplxLiteJ24(gender="neutral").to(self.device)
            self.body_model.eval()

    def get_required_keys(self) -> list:
        return ["poses", "trans"]

    def check(self, motion: Union[Dict, str, Path]) -> CheckResult:
        """Run jitter detection. Returns CheckResult and legacy-style details when invalid."""
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
                details={"has_jitter": False, "reason": err},
            )

        poses = np.array(data["poses"])
        trans = np.array(data["trans"])
        if len(poses) < MIN_FRAMES_REQUIRED:
            reason = f"Too short (need at least {MIN_FRAMES_REQUIRED} frames)"
            return CheckResult(
                is_valid=False,
                invalid_reason=reason,
                invalid_mask=None,
                details={"has_jitter": False, "reason": reason},
            )

        try:
            poses_3d = self.normalize_poses(poses, NUM_BODY_JOINTS)
        except ValueError as e:
            return CheckResult(
                is_valid=False,
                invalid_reason=str(e),
                invalid_mask=None,
                details={"has_jitter": False, "reason": str(e)},
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
                details={"has_jitter": False, "reason": f"FK calculation failed: {str(e)}"},
            )
        try:
            jitter_windows, all_jitter_joints = _collect_direction_flip_windows(
                joints,
                source_label="world",
            )
            root_relative_windows, root_relative_joints = _collect_direction_flip_windows(
                _make_root_relative_positions(joints),
                source_label="root_relative",
                skip_root_joint=True,
            )
            jitter_windows.extend(root_relative_windows)
            all_jitter_joints.update(root_relative_joints)
        except Exception as e:
            return CheckResult(
                is_valid=False,
                invalid_reason=f"Velocity calculation failed: {e}",
                invalid_mask=None,
                details={"has_jitter": False, "reason": f"Velocity calculation failed: {str(e)}"},
            )

        accel_jitter = _detect_acceleration_spike_jitter(joints)
        accel_jitter_joints = accel_jitter.get("joints", []) or []
        accel_jitter_details = accel_jitter.get("details", []) or []
        borderline_accel_jitter_joints = accel_jitter.get("borderline_joints", []) or []
        borderline_accel_jitter_details = accel_jitter.get("borderline_details", []) or []
        invalid_mask = np.zeros((joints.shape[0], NUM_BODY_JOINTS), dtype=bool)
        borderline_mask = np.zeros((joints.shape[0], NUM_BODY_JOINTS), dtype=bool)

        for window in jitter_windows:
            for detail in window.get("jitter_details", []) or []:
                joint_id = int(detail.get("joint_id", -1))
                if joint_id < 0 or joint_id >= NUM_BODY_JOINTS:
                    continue
                frame_indices = [int(f) for f in (detail.get("frame_indices") or []) if 0 <= int(f) < invalid_mask.shape[0]]
                if frame_indices:
                    invalid_mask[np.asarray(frame_indices, dtype=np.int64), joint_id] = True
                else:
                    start = int(window.get("window_start", 0)) + 1
                    end = int(window.get("window_end", start)) + 1
                    invalid_mask[max(start, 0):min(end, invalid_mask.shape[0]), joint_id] = True

        for item in accel_jitter_details:
            joint_id = int(item.get("joint_id", -1))
            if joint_id < 0 or joint_id >= NUM_BODY_JOINTS:
                continue
            spike_frames = [int(f) for f in (item.get("spike_frames") or []) if 0 <= int(f) < invalid_mask.shape[0]]
            if spike_frames:
                invalid_mask[np.asarray(spike_frames, dtype=np.int64), joint_id] = True

        for item in borderline_accel_jitter_details:
            joint_id = int(item.get("joint_id", -1))
            if joint_id < 0 or joint_id >= NUM_BODY_JOINTS:
                continue
            spike_frames = [int(f) for f in (item.get("spike_frames") or []) if 0 <= int(f) < borderline_mask.shape[0]]
            if spike_frames:
                borderline_mask[np.asarray(spike_frames, dtype=np.int64), joint_id] = True

        if not jitter_windows and not accel_jitter_joints and not borderline_accel_jitter_joints:
            return CheckResult(
                is_valid=True,
                invalid_reason="No jitter detected",
                invalid_mask=np.zeros((joints.shape[0], NUM_BODY_JOINTS), dtype=bool),
                details={
                    "has_jitter": False,
                    "jitter_windows": [],
                    "jitter_joints": [],
                    "accel_jitter_details": [],
                    "borderline_accel_jitter_details": [],
                    "reason": "No jitter detected",
                },
                severity="pass",
            )

        all_jitter_joints.update(accel_jitter_joints)
        jitter_joints_sorted = sorted(all_jitter_joints)
        jitter_joint_names = [JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Joint_{j}" for j in jitter_joints_sorted]
        if jitter_windows or accel_jitter_joints:
            reason_parts = []
            if jitter_windows:
                world_window_count = sum(1 for item in jitter_windows if item.get("source") == "world")
                root_relative_window_count = sum(1 for item in jitter_windows if item.get("source") == "root_relative")
                source_parts = []
                if world_window_count:
                    source_parts.append(f"world={world_window_count}")
                if root_relative_window_count:
                    source_parts.append(f"root-relative={root_relative_window_count}")
                reason_parts.append(
                    f"direction-flip windows={len(jitter_windows)}"
                    + (f" ({', '.join(source_parts)})" if source_parts else "")
                )
            if accel_jitter_details:
                accel_names = ", ".join(d["joint_name"] for d in accel_jitter_details)
                reason_parts.append(f"acceleration-spike joints={accel_names}")
            reason = f"Jitter detected ({'; '.join(reason_parts)}), joints: {', '.join(jitter_joint_names)}"
            details = {
                "has_jitter": True,
                "jitter_windows": jitter_windows,
                "jitter_joints": jitter_joints_sorted,
                "jitter_joint_names": jitter_joint_names,
                "accel_jitter_details": accel_jitter_details,
                "borderline_accel_jitter_details": borderline_accel_jitter_details,
                "reason": reason,
            }
            return CheckResult(
                is_valid=False,
                invalid_reason=reason,
                invalid_mask=invalid_mask,
                details=details,
                severity="fail",
            )

        borderline_joint_ids = sorted(set(int(x) for x in borderline_accel_jitter_joints))
        borderline_joint_names = [
            JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Joint_{j}" for j in borderline_joint_ids
        ]
        borderline_reason = (
            "Mild jitter detected (acceleration spikes below low-quality threshold), joints: "
            + ", ".join(borderline_joint_names)
        )
        return CheckResult(
            is_valid=True,
            invalid_reason=borderline_reason,
            invalid_mask=borderline_mask,
            details={
                "has_jitter": False,
                "has_borderline_jitter": True,
                "jitter_windows": [],
                "jitter_joints": [],
                "jitter_joint_names": [],
                "accel_jitter_details": [],
                "borderline_accel_jitter_details": borderline_accel_jitter_details,
                "borderline_jitter_joints": borderline_joint_ids,
                "borderline_jitter_joint_names": borderline_joint_names,
                "reason": borderline_reason,
            },
            severity="borderline",
        )


# Process-local cache for legacy API: one JitterChecker per (process, device) to avoid
# re-loading the body model on every call (original script used _SMPL_MODEL_CACHE per process).
_CHECKER_CACHE: Dict[str, JitterChecker] = {}


def detect_jitter(data: Dict, device: str = "cpu") -> Dict:
    """Legacy function: same signature and return dict as filter_jitter_from_completion.detect_jitter.

    Returns dict with keys: has_jitter, jitter_windows, jitter_joints, jitter_joint_names, reason.
    Reuses a single JitterChecker per process per device so the body model is loaded only once.
    """
    if device not in _CHECKER_CACHE:
        _CHECKER_CACHE[device] = JitterChecker(device=device)
    checker = _CHECKER_CACHE[device]
    result = checker.check(data)
    details = result.get("details") or {}
    return {
        "has_jitter": details.get("has_jitter", False),
        "jitter_windows": details.get("jitter_windows", []),
        "jitter_joints": details.get("jitter_joints", []),
        "jitter_joint_names": details.get("jitter_joint_names", []),
        "reason": details.get("reason", result.get("invalid_reason", "")),
    }
