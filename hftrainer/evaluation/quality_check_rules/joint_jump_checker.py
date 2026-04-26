"""
Joint jump checker: detects large per-frame displacement of joint positions in *root-stabilized*
local space, using FK and then removing both root translation and root orientation.

Such jumps indicate artifacts (e.g. retargeting or bad interpolation): the limb "pops" with
both large displacement and a sharp change of direction. Fast but smooth motion (e.g. running)
has large displacement but consistent direction, so we require displacement direction to change
substantially between consecutive frame pairs.

Rule:
  - FK -> root-stabilized positions local[t,j]. Displacement vectors disp_vec[t,j] = local[t+1,j]-local[t,j].
  - Frame pair t is a "jump" for joint j if: (1) |disp_vec[t,j]| > jump_threshold_m,
    (2) at the next frame pair t+1 the *same* joint j also has displacement > threshold,
    (3) the angle between disp_vec[t,j] and disp_vec[t+1,j] >= jump_angle_deg (e.g. 90°).
  - ALL joints exceeding threshold are checked per frame pair (not just the worst one).
  - If require_clustered_jumps: only count as invalid when there are >= min_jump_frames such frame
    pairs AND at least two of them are within max_jump_frame_gap (so fast combat with isolated
    direction changes is not flagged).
  - Otherwise: if there are >= min_jump_frames such frame pairs, flag as invalid.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base_checker import BaseQualityChecker, CheckResult, NUM_BODY_JOINTS_DEFAULT, normalize_betas_array
from .root_motion_utils import (
    root_angular_velocity_deg_per_frame,
    root_rotation_matrices_from_poses,
    root_stabilize_positions,
)

from ._model_compat import SmplxLiteJ24

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_BODY_JOINTS = 22
DEFAULT_JUMP_THRESHOLD_M = 0.30
DEFAULT_JUMP_ANGLE_DEG = 115.0
DEFAULT_JUMP_ACCEL_THRESHOLD_M = 0.18
DEFAULT_JUMP_ACCEL_MEDIAN_RATIO = 4.0
DEFAULT_MIN_JUMP_FRAMES = 2
DEFAULT_MAX_JUMP_FRAME_GAP = 6
DEFAULT_ROOT_TRANSLATION_JUMP_THRESHOLD_M = 0.18
DEFAULT_ROOT_ROTATION_JUMP_ANGLE_DEG = 25.0
DEFAULT_ROOT_TRANSLATION_ACCEL_THRESHOLD_M = 0.10
DEFAULT_ROOT_ROTATION_ACCEL_THRESHOLD_DEG = 14.0

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
    """FK: world-space joint positions from poses and trans. Returns (F, 24, 3)."""
    if body_model is None or SmplxLiteJ24 is None:
        raise RuntimeError("JointJumpChecker requires body_model (SmplxLiteJ24) for FK.")
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


def _to_root_relative(joints: np.ndarray) -> np.ndarray:
    """
    Subtract root (joint 0) position from all joints each frame.
    joints: (F, J, 3). Returns (F, J, 3) root-relative positions.
    """
    root = joints[:, 0:1, :]  # (F, 1, 3)
    return joints - root


def _compute_displacement_vectors(local_joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-joint displacement vectors and magnitudes in local space.

    local_joints: (F, J, 3). Returns:
      disp_vec: (F-1, J, 3) displacement vectors
      disp_mag: (F-1, J) magnitudes
    """
    disp_vec = np.diff(local_joints, axis=0)  # (F-1, J, 3)
    disp_mag = np.linalg.norm(disp_vec, axis=2)  # (F-1, J)
    return disp_vec, disp_mag


def _has_clustered_jumps(jump_frame_indices: List[int], gap: int) -> bool:
    """O(n) check: are there at least two jump frames within `gap` of each other?

    Assumes jump_frame_indices is sorted (ascending).
    """
    for i in range(len(jump_frame_indices) - 1):
        if jump_frame_indices[i + 1] - jump_frame_indices[i] <= gap:
            return True
    return False


def _filter_root_anomaly_frames(
    frames: np.ndarray,
    primary_values: np.ndarray,
    secondary_values: np.ndarray,
    *,
    primary_gate: float,
    secondary_gate: float,
    max_gap: int,
    isolated_primary_scale: float = 1.7,
    isolated_secondary_scale: float = 1.5,
) -> np.ndarray:
    """Keep clustered root anomalies, but suppress isolated spikes unless they are clearly extreme."""
    frames = np.asarray(frames, dtype=np.int64)
    if frames.size == 0:
        return frames
    if frames.size >= 2 and _has_clustered_jumps(frames.tolist(), max_gap):
        return frames
    kept: List[int] = []
    for frame in frames.tolist():
        if frame < 0 or frame >= len(primary_values) or frame >= len(secondary_values):
            continue
        if (
            float(primary_values[frame]) >= float(primary_gate) * isolated_primary_scale
            and float(secondary_values[frame]) >= float(secondary_gate) * isolated_secondary_scale
        ):
            kept.append(int(frame))
    return np.asarray(sorted(set(kept)), dtype=np.int64)


class JointJumpChecker(BaseQualityChecker):
    """
    Detects joint position jumps: large root-relative displacement *and* large
    change of displacement direction between consecutive frame pairs (so smooth
    fast motion is not flagged).

    Checks ALL joints exceeding threshold per frame pair (not just the single worst).
    """

    name = "joint_jump"

    def __init__(
        self,
        body_model: Optional[Any] = None,
        device: str = "cuda",
        jump_threshold_m: float = DEFAULT_JUMP_THRESHOLD_M,
        jump_angle_deg: float = DEFAULT_JUMP_ANGLE_DEG,
        jump_accel_threshold_m: float = DEFAULT_JUMP_ACCEL_THRESHOLD_M,
        jump_accel_median_ratio: float = DEFAULT_JUMP_ACCEL_MEDIAN_RATIO,
        min_jump_frames: int = DEFAULT_MIN_JUMP_FRAMES,
        max_jump_frame_gap: int = DEFAULT_MAX_JUMP_FRAME_GAP,
        root_translation_jump_threshold_m: float = DEFAULT_ROOT_TRANSLATION_JUMP_THRESHOLD_M,
        root_rotation_jump_angle_deg: float = DEFAULT_ROOT_ROTATION_JUMP_ANGLE_DEG,
        root_translation_accel_threshold_m: float = DEFAULT_ROOT_TRANSLATION_ACCEL_THRESHOLD_M,
        root_rotation_accel_threshold_deg: float = DEFAULT_ROOT_ROTATION_ACCEL_THRESHOLD_DEG,
        require_clustered_jumps: bool = True,
    ) -> None:
        super().__init__(body_model=body_model, device=device)
        if self.body_model is None and SmplxLiteJ24 is not None:
            self.body_model = SmplxLiteJ24(gender="neutral").to(self.device)
            self.body_model.eval()
        self.jump_threshold_m = jump_threshold_m
        self.jump_angle_deg = jump_angle_deg
        self.jump_accel_threshold_m = jump_accel_threshold_m
        self.jump_accel_median_ratio = jump_accel_median_ratio
        self.min_jump_frames = min_jump_frames
        self.max_jump_frame_gap = max_jump_frame_gap
        self.root_translation_jump_threshold_m = root_translation_jump_threshold_m
        self.root_rotation_jump_angle_deg = root_rotation_jump_angle_deg
        self.root_translation_accel_threshold_m = root_translation_accel_threshold_m
        self.root_rotation_accel_threshold_deg = root_rotation_accel_threshold_deg
        self.require_clustered_jumps = require_clustered_jumps

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
                details={"has_jump": False, "reason": err},
            )

        poses = np.array(data["poses"])
        trans = np.array(data["trans"])
        if len(poses) < 2:
            reason = "Too short (need at least 2 frames)"
            return CheckResult(
                is_valid=False,
                invalid_reason=reason,
                invalid_mask=None,
                details={"has_jump": False, "reason": reason},
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
                details={"has_jump": False, "reason": str(e)},
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
                details={"has_jump": False, "reason": f"FK failed: {str(e)}"},
            )
        root_rot_mats = (
            np.asarray(data.get("_cached_root_rot_mats_22"))
            if data.get("_cached_root_rot_mats_22") is not None
            else root_rotation_matrices_from_poses(poses_3d, device=self.device)
        )
        local_joints = root_stabilize_positions(joints, root_rot_mats)  # (F, 22, 3), remove root translation and yaw/pitch/roll
        disp_vec, disp_mag = _compute_displacement_vectors(local_joints)
        # disp_vec (F-1, J, 3), disp_mag (F-1, J)

        threshold = self.jump_threshold_m
        angle_thresh = self.jump_angle_deg
        accel_threshold = self.jump_accel_threshold_m
        accel_median_ratio = self.jump_accel_median_ratio
        min_frames = self.min_jump_frames
        n_pairs = disp_mag.shape[0]
        J = disp_mag.shape[1]
        jump_details: List[Dict] = []
        if n_pairs >= 2:
            disp_a = disp_vec[:-1]
            disp_b = disp_vec[1:]
            norm_a = np.linalg.norm(disp_a, axis=2)
            norm_b = np.linalg.norm(disp_b, axis=2)
            denom = norm_a * norm_b
            valid = denom > 1e-9
            dot = np.sum(disp_a * disp_b, axis=2)
            cos_angle = np.ones_like(dot)
            cos_angle[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
            angle_deg = np.zeros_like(dot)
            angle_deg[valid] = np.rad2deg(np.arccos(cos_angle[valid]))
            accel_vec = disp_b - disp_a
            accel_mag = np.linalg.norm(accel_vec, axis=2)
            accel_baseline = np.median(accel_mag, axis=0)
            accel_gate = np.maximum(accel_threshold, accel_baseline[None, :] * accel_median_ratio)

            jump_mask = (
                (disp_mag[:-1] > threshold)
                & (disp_mag[1:] > threshold)
                & (angle_deg >= angle_thresh)
                & (accel_mag >= accel_gate)
            )
            jump_t, jump_j = np.where(jump_mask)
            joint_jump_frame_indices = sorted(set(jump_t.astype(int).tolist()))
            for t, j in zip(jump_t.tolist(), jump_j.tolist()):
                jump_details.append(
                    {
                        "frame": int(t),
                        "joint_id": int(j),
                        "joint_name": JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Joint_{j}",
                        "jump_kind": "joint",
                        "displacement_m": float(disp_mag[t, j]),
                        "angle_deg": float(angle_deg[t, j]),
                        "accel_m": float(accel_mag[t, j]),
                        "accel_gate_m": float(accel_gate[t, j]),
                    }
                )
        else:
            joint_jump_frame_indices = []
            accel_mag = np.zeros((0, J), dtype=np.float64)

        root_translation_vel = np.diff(joints[:, 0, :], axis=0)
        root_translation_speed = np.linalg.norm(root_translation_vel, axis=1)
        if root_translation_speed.shape[0] >= 2:
            root_translation_accel = np.linalg.norm(np.diff(root_translation_vel, axis=0), axis=1)
            vel_a = root_translation_vel[:-1]
            vel_b = root_translation_vel[1:]
            vel_a_norm = np.linalg.norm(vel_a, axis=1)
            vel_b_norm = np.linalg.norm(vel_b, axis=1)
            vel_denom = vel_a_norm * vel_b_norm
            root_turn_angle = np.zeros_like(root_translation_accel)
            valid_vel = vel_denom > 1e-9
            if np.any(valid_vel):
                vel_dot = np.sum(vel_a * vel_b, axis=1)
                vel_cos = np.ones_like(root_translation_accel)
                vel_cos[valid_vel] = np.clip(vel_dot[valid_vel] / vel_denom[valid_vel], -1.0, 1.0)
                root_turn_angle[valid_vel] = np.rad2deg(np.arccos(vel_cos[valid_vel]))
            speed_ratio = np.maximum(vel_a_norm, vel_b_norm) / np.maximum(np.minimum(vel_a_norm, vel_b_norm), 1e-4)
            root_translation_mask = (
                (np.maximum(vel_a_norm, vel_b_norm) > self.root_translation_jump_threshold_m)
                & (root_translation_accel > self.root_translation_accel_threshold_m)
                & ((root_turn_angle >= 75.0) | (speed_ratio >= 2.25))
            )
            root_translation_peak_speed = np.maximum(vel_a_norm, vel_b_norm)
            root_translation_frames = np.where(root_translation_mask)[0].astype(int)
            root_translation_frames = _filter_root_anomaly_frames(
                root_translation_frames,
                root_translation_peak_speed,
                root_translation_accel,
                primary_gate=self.root_translation_jump_threshold_m,
                secondary_gate=self.root_translation_accel_threshold_m,
                max_gap=self.max_jump_frame_gap,
            )
        else:
            root_translation_accel = np.zeros((0,), dtype=np.float64)
            root_turn_angle = np.zeros((0,), dtype=np.float64)
            root_translation_peak_speed = np.zeros((0,), dtype=np.float64)
            root_translation_frames = np.zeros((0,), dtype=np.int64)

        root_rotation_speed = root_angular_velocity_deg_per_frame(root_rot_mats)
        if root_rotation_speed.shape[0] >= 2:
            root_rotation_accel = np.abs(np.diff(root_rotation_speed))
            rot_speed_a = root_rotation_speed[:-1]
            rot_speed_b = root_rotation_speed[1:]
            rot_ratio = np.maximum(rot_speed_a, rot_speed_b) / np.maximum(np.minimum(rot_speed_a, rot_speed_b), 1e-3)
            root_rotation_mask = (
                (np.maximum(rot_speed_a, rot_speed_b) > self.root_rotation_jump_angle_deg)
                & (root_rotation_accel > self.root_rotation_accel_threshold_deg)
                & (rot_ratio >= 2.0)
            )
            root_rotation_peak_speed = np.maximum(rot_speed_a, rot_speed_b)
            root_rotation_frames = np.where(root_rotation_mask)[0].astype(int)
            root_rotation_frames = _filter_root_anomaly_frames(
                root_rotation_frames,
                root_rotation_peak_speed,
                root_rotation_accel,
                primary_gate=self.root_rotation_jump_angle_deg,
                secondary_gate=self.root_rotation_accel_threshold_deg,
                max_gap=self.max_jump_frame_gap,
            )
        else:
            root_rotation_accel = np.zeros((0,), dtype=np.float64)
            root_rotation_peak_speed = np.zeros((0,), dtype=np.float64)
            root_rotation_frames = np.zeros((0,), dtype=np.int64)

        for frame in root_translation_frames.tolist():
            jump_details.append(
                {
                    "frame": int(frame),
                    "joint_id": 0,
                    "joint_name": JOINT_NAMES[0],
                    "jump_kind": "root_translation",
                    "displacement_m": float(root_translation_speed[min(frame, root_translation_speed.shape[0] - 1)]),
                    "accel_m": float(root_translation_accel[frame]),
                    "speed_gate_m": float(self.root_translation_jump_threshold_m),
                    "accel_gate_m": float(self.root_translation_accel_threshold_m),
                }
            )
        for frame in root_rotation_frames.tolist():
            jump_details.append(
                {
                    "frame": int(frame),
                    "joint_id": 0,
                    "joint_name": JOINT_NAMES[0],
                    "jump_kind": "root_rotation",
                    "angle_deg": float(root_rotation_speed[min(frame, root_rotation_speed.shape[0] - 1)]),
                    "accel_deg": float(root_rotation_accel[frame]),
                    "speed_gate_deg": float(self.root_rotation_jump_angle_deg),
                    "accel_gate_deg": float(self.root_rotation_accel_threshold_deg),
                }
            )
        root_jump_frame_indices = sorted(set(root_translation_frames.tolist() + root_rotation_frames.tolist()))

        # Require jump frames to be clustered to filter isolated combat direction changes
        if self.require_clustered_jumps and len(joint_jump_frame_indices) >= 2:
            if not _has_clustered_jumps(joint_jump_frame_indices, self.max_jump_frame_gap):
                joint_jump_frame_indices = []
                jump_details = [item for item in jump_details if item.get("jump_kind") != "joint"]

        jump_frame_indices = sorted(set(joint_jump_frame_indices + root_jump_frame_indices))

        if len(joint_jump_frame_indices) < min_frames and not root_jump_frame_indices:
            return CheckResult(
                is_valid=True,
                invalid_reason="No joint jump detected",
                invalid_mask=np.zeros((poses.shape[0], NUM_BODY_JOINTS), dtype=bool),
                details={
                    "has_jump": False,
                    "jump_frames": [],
                    "reason": "No joint jump detected",
                },
            )

        root_jump_kinds = sorted({item["jump_kind"] for item in jump_details if item.get("jump_kind", "").startswith("root_")})
        if joint_jump_frame_indices:
            reason = (
                f"Joint jump in {len(jump_frame_indices)} frame pair(s) "
                f"(disp > {threshold*100:.1f} cm and direction angle >= {angle_thresh}°)"
            )
            if root_jump_kinds:
                reason += f"; root anomalies: {', '.join(root_jump_kinds)}"
        else:
            reason = f"Root anomalies detected: {', '.join(root_jump_kinds)}"
        details = {
            "has_jump": True,
            "jump_frames": jump_frame_indices,
            "jump_details": jump_details,
            "jump_threshold_m": threshold,
            "jump_accel_threshold_m": float(self.jump_accel_threshold_m),
            "jump_accel_median_ratio": float(self.jump_accel_median_ratio),
            "root_translation_jump_threshold_m": float(self.root_translation_jump_threshold_m),
            "root_rotation_jump_angle_deg": float(self.root_rotation_jump_angle_deg),
            "root_translation_accel_threshold_m": float(self.root_translation_accel_threshold_m),
            "root_rotation_accel_threshold_deg": float(self.root_rotation_accel_threshold_deg),
            "reason": reason,
        }
        # Build per-frame per-joint invalid_mask from jump_details
        num_frames = poses.shape[0]
        invalid_mask = np.zeros((num_frames, NUM_BODY_JOINTS), dtype=bool)
        for item in jump_details:
            frame = int(item.get("frame", -1))
            joint_id = int(item.get("joint_id", -1))
            if 0 <= frame < num_frames and 0 <= joint_id < NUM_BODY_JOINTS:
                invalid_mask[frame, joint_id] = True
                # Also mark the next frame (jump spans two consecutive frame pairs)
                if frame + 1 < num_frames:
                    invalid_mask[frame + 1, joint_id] = True

        return CheckResult(
            is_valid=False,
            invalid_reason=reason,
            invalid_mask=invalid_mask,
            details=details,
        )


# Process-local cache for legacy API: one JointJumpChecker per (process, device),
# so body model is loaded only once per worker in multiprocessing (spawn).
_CHECKER_CACHE: Dict[str, JointJumpChecker] = {}


def detect_joint_jump(
    data: Dict,
    device: str = "cpu",
    jump_threshold_m: Optional[float] = None,
    jump_angle_deg: Optional[float] = None,
    jump_accel_threshold_m: Optional[float] = None,
    jump_accel_median_ratio: Optional[float] = None,
    min_jump_frames: Optional[int] = None,
    max_jump_frame_gap: Optional[int] = None,
    root_translation_jump_threshold_m: Optional[float] = None,
    root_rotation_jump_angle_deg: Optional[float] = None,
    root_translation_accel_threshold_m: Optional[float] = None,
    root_rotation_accel_threshold_deg: Optional[float] = None,
    require_clustered_jumps: Optional[bool] = None,
) -> Dict:
    """
    Legacy function for filter scripts.

    Returns dict with keys: has_jump, jump_frames, jump_details, reason.
    Reuses a single JointJumpChecker per process per device (body model loaded once per worker).
    """
    cache_key = device
    if cache_key not in _CHECKER_CACHE:
        kwargs = {}
        if jump_threshold_m is not None:
            kwargs["jump_threshold_m"] = jump_threshold_m
        if jump_angle_deg is not None:
            kwargs["jump_angle_deg"] = jump_angle_deg
        if jump_accel_threshold_m is not None:
            kwargs["jump_accel_threshold_m"] = jump_accel_threshold_m
        if jump_accel_median_ratio is not None:
            kwargs["jump_accel_median_ratio"] = jump_accel_median_ratio
        if min_jump_frames is not None:
            kwargs["min_jump_frames"] = min_jump_frames
        if max_jump_frame_gap is not None:
            kwargs["max_jump_frame_gap"] = max_jump_frame_gap
        if root_translation_jump_threshold_m is not None:
            kwargs["root_translation_jump_threshold_m"] = root_translation_jump_threshold_m
        if root_rotation_jump_angle_deg is not None:
            kwargs["root_rotation_jump_angle_deg"] = root_rotation_jump_angle_deg
        if root_translation_accel_threshold_m is not None:
            kwargs["root_translation_accel_threshold_m"] = root_translation_accel_threshold_m
        if root_rotation_accel_threshold_deg is not None:
            kwargs["root_rotation_accel_threshold_deg"] = root_rotation_accel_threshold_deg
        if require_clustered_jumps is not None:
            kwargs["require_clustered_jumps"] = require_clustered_jumps
        _CHECKER_CACHE[cache_key] = JointJumpChecker(device=device, **kwargs)
    checker = _CHECKER_CACHE[cache_key]
    result = checker.check(data)
    details = result.get("details") or {}
    return {
        "has_jump": details.get("has_jump", False),
        "jump_frames": details.get("jump_frames", []),
        "jump_details": details.get("jump_details", []),
        "reason": details.get("reason", result.get("invalid_reason", "")),
    }
