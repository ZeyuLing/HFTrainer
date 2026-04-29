"""
Limb penetration checker: detects when arms or legs visibly penetrate the torso.

The historical name is kept as ``arm_penetration`` for backward compatibility,
but the implementation now covers torso collisions from both upper and lower
limbs. We still use skeleton-space heuristics instead of mesh collision because
the checker must stay fast enough for large-scale dataset scans.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base_checker import BaseQualityChecker, CheckResult, NUM_BODY_JOINTS_DEFAULT, normalize_betas_array

from ._model_compat import SmplxLiteJ24

SPINE_JOINT_INDICES = [0, 3, 6, 9, 12]
TORSO_SEGMENTS = [(0, 3), (3, 6), (6, 9), (9, 12)]
TORSO_SEGMENT_NAMES = ["Pelvis-Spine1", "Spine1-Spine2", "Spine2-Spine3", "Spine3-Neck"]

CANDIDATE_SEGMENTS: List[Tuple[str, int, int]] = [
    ("L_upper_arm", 16, 18),
    ("L_forearm", 18, 20),
    ("R_upper_arm", 17, 19),
    ("R_forearm", 19, 21),
    ("L_thigh", 1, 4),
    ("L_shin", 4, 7),
    ("R_thigh", 2, 5),
    ("R_shin", 5, 8),
]
SEGMENT_NAME_TO_JOINTS = {name: [j0, j1] for name, j0, j1 in CANDIDATE_SEGMENTS}
SEGMENT_NAMES = [name for name, _, _ in CANDIDATE_SEGMENTS]

DEFAULT_DISTANCE_THRESHOLD_M = 0.06
DEFAULT_MIN_PENETRATION_FRAMES = 3
DEFAULT_MIN_PENETRATION_RATIO = 0.05
DEFAULT_TORSO_HALF_WIDTH_MIN_M = 0.09
DEFAULT_TORSO_HALF_WIDTH_SCALE = 0.38


def _normalize_vectors(vectors: np.ndarray, fallback: Tuple[float, float, float]) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    out = np.zeros_like(vectors, dtype=np.float64)
    valid = norms[..., 0] > 1e-8
    out[valid] = vectors[valid] / norms[valid]
    out[~valid] = np.asarray(fallback, dtype=np.float64)
    return out


def _point_to_segment_distance(points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Broadcasted distance from point(s) to segment(s)."""
    ab = seg_b - seg_a
    ab_sq = np.sum(ab * ab, axis=-1, keepdims=True)
    ap = points - seg_a
    numerator = np.sum(ap * ab, axis=-1, keepdims=True)
    t = np.divide(
        numerator,
        ab_sq,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=ab_sq > 1e-12,
    )
    t = np.clip(t, 0.0, 1.0)
    closest = seg_a + t * ab
    return np.linalg.norm(points - closest, axis=-1)


def _segment_to_segment_distance_batch(
    seg_a0: np.ndarray,
    seg_a1: np.ndarray,
    seg_b0: np.ndarray,
    seg_b1: np.ndarray,
) -> np.ndarray:
    """Approximate min distance between segment batches with endpoint-to-segment tests."""
    d1 = _point_to_segment_distance(seg_a0[:, :, None, :], seg_b0[:, None, :, :], seg_b1[:, None, :, :])
    d2 = _point_to_segment_distance(seg_a1[:, :, None, :], seg_b0[:, None, :, :], seg_b1[:, None, :, :])
    d3 = _point_to_segment_distance(seg_b0[:, None, :, :], seg_a0[:, :, None, :], seg_a1[:, :, None, :])
    d4 = _point_to_segment_distance(seg_b1[:, None, :, :], seg_a0[:, :, None, :], seg_a1[:, :, None, :])
    return np.minimum(np.minimum(d1, d2), np.minimum(d3, d4))


def _compute_torso_context(joints: np.ndarray) -> Dict[str, np.ndarray]:
    pelvis = joints[:, 0]
    neck = joints[:, 12]
    left_shoulder = joints[:, 16]
    right_shoulder = joints[:, 17]
    left_hip = joints[:, 1]
    right_hip = joints[:, 2]

    spine_up = _normalize_vectors(neck - pelvis, fallback=(0.0, 1.0, 0.0))
    side = _normalize_vectors(right_shoulder - left_shoulder, fallback=(1.0, 0.0, 0.0))
    forward = _normalize_vectors(np.cross(spine_up, side), fallback=(0.0, 0.0, 1.0))

    torso_height = np.linalg.norm(neck - pelvis, axis=-1)
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder, axis=-1)
    hip_width = np.linalg.norm(right_hip - left_hip, axis=-1)
    torso_half_width = np.maximum(
        DEFAULT_TORSO_HALF_WIDTH_MIN_M,
        DEFAULT_TORSO_HALF_WIDTH_SCALE * np.maximum(shoulder_width, hip_width),
    )

    return {
        "spine_up": spine_up,
        "side": side,
        "forward": forward,
        "spine_mid": 0.5 * (joints[:, 6] + joints[:, 9]),
        "torso_height": torso_height,
        "torso_half_width": torso_half_width,
    }


def _get_joints_from_pose(
    poses: np.ndarray,
    trans: np.ndarray,
    betas: Optional[np.ndarray] = None,
    body_model: Optional[Any] = None,
    device: str = "cpu",
) -> np.ndarray:
    if body_model is None or SmplxLiteJ24 is None:
        raise RuntimeError("ArmPenetrationChecker requires body_model (SmplxLiteJ24) for FK.")
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


class ArmPenetrationChecker(BaseQualityChecker):
    """
    Detects torso penetration caused by limbs (arms and legs).

    The checker keeps the historical ``arm_penetration`` key so existing quality
    versions and training bindings remain compatible.
    """

    name = "arm_penetration"

    def __init__(
        self,
        body_model: Optional[Any] = None,
        device: str = "cuda",
        distance_threshold_m: float = DEFAULT_DISTANCE_THRESHOLD_M,
        min_penetration_frames: int = DEFAULT_MIN_PENETRATION_FRAMES,
        min_penetration_ratio: float = DEFAULT_MIN_PENETRATION_RATIO,
    ) -> None:
        super().__init__(body_model=body_model, device=device)
        if self.body_model is None and SmplxLiteJ24 is not None:
            self.body_model = SmplxLiteJ24(gender="neutral").to(self.device)
            self.body_model.eval()
        self.distance_threshold_m = distance_threshold_m
        self.min_penetration_frames = min_penetration_frames
        self.min_penetration_ratio = min_penetration_ratio

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
                details={"has_penetration": False, "reason": err},
            )

        poses = np.array(data["poses"])
        trans = np.array(data["trans"])
        if len(poses) < 2:
            reason = "Too short (need at least 2 frames)"
            return CheckResult(
                is_valid=False,
                invalid_reason=reason,
                invalid_mask=None,
                details={"has_penetration": False, "reason": reason},
            )

        try:
            poses_3d = self.normalize_poses(poses, NUM_BODY_JOINTS_DEFAULT)
        except ValueError as e:
            return CheckResult(
                is_valid=False,
                invalid_reason=str(e),
                invalid_mask=None,
                details={"has_penetration": False, "reason": str(e)},
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
                )[:, :22, :]
        except Exception as e:
            return CheckResult(
                is_valid=False,
                invalid_reason=f"FK failed: {e}",
                invalid_mask=None,
                details={"has_penetration": False, "reason": f"FK failed: {str(e)}"},
            )

        num_frames = joints.shape[0]
        threshold = float(self.distance_threshold_m)
        min_frames = int(self.min_penetration_frames)
        min_ratio = float(self.min_penetration_ratio)

        torso_a = joints[:, [a for a, _ in TORSO_SEGMENTS], :]
        torso_b = joints[:, [b for _, b in TORSO_SEGMENTS], :]
        limb_a = joints[:, [a for _, a, _ in CANDIDATE_SEGMENTS], :]
        limb_b = joints[:, [b for _, _, b in CANDIDATE_SEGMENTS], :]
        pairwise_distance = _segment_to_segment_distance_batch(limb_a, limb_b, torso_a, torso_b)
        min_distance_per_segment = np.min(pairwise_distance, axis=2)
        nearest_torso_idx = np.argmin(pairwise_distance, axis=2)

        torso_ctx = _compute_torso_context(joints)
        limb_mid = 0.5 * (limb_a + limb_b)
        offset = limb_mid - torso_ctx["spine_mid"][:, None, :]
        depth = np.sum(offset * torso_ctx["forward"][:, None, :], axis=-1)
        side_offset = np.abs(np.sum(offset * torso_ctx["side"][:, None, :], axis=-1))
        height_offset = np.sum(offset * torso_ctx["spine_up"][:, None, :], axis=-1)

        torso_overlap = (
            (side_offset <= torso_ctx["torso_half_width"][:, None])
            & (height_offset >= (-0.20 * torso_ctx["torso_height"][:, None]))
            & (height_offset <= (0.80 * torso_ctx["torso_height"][:, None]))
            & (depth <= (0.50 * torso_ctx["torso_half_width"][:, None]))
        )
        penetration_mask = (min_distance_per_segment < threshold) & torso_overlap
        penetration_frames = np.where(np.any(penetration_mask, axis=1))[0].astype(int).tolist()
        penetration_ratio = len(penetration_frames) / num_frames if num_frames > 0 else 0.0

        worst_segment_per_frame: List[Optional[str]] = [None] * num_frames
        worst_torso_segment_per_frame: List[Optional[str]] = [None] * num_frames
        penetration_joint_ids_per_frame: List[List[int]] = [[] for _ in range(num_frames)]
        penetration_frame_details: List[Dict[str, Any]] = []
        invalid_mask = np.zeros((num_frames, 22), dtype=bool)

        for frame_idx in penetration_frames:
            active_segment_indices = np.where(penetration_mask[frame_idx])[0].astype(int).tolist()
            if not active_segment_indices:
                continue
            segment_names = [SEGMENT_NAMES[idx] for idx in active_segment_indices]
            segment_distances = min_distance_per_segment[frame_idx, active_segment_indices]
            worst_local = int(active_segment_indices[int(np.argmin(segment_distances))])
            worst_segment_name = SEGMENT_NAMES[worst_local]
            torso_segment_name = TORSO_SEGMENT_NAMES[int(nearest_torso_idx[frame_idx, worst_local])]
            worst_segment_per_frame[frame_idx] = worst_segment_name
            worst_torso_segment_per_frame[frame_idx] = torso_segment_name

            joint_ids = set()
            torso_idx = int(nearest_torso_idx[frame_idx, worst_local])
            torso_joint_ids = list(TORSO_SEGMENTS[torso_idx])
            for seg_idx in active_segment_indices:
                seg_name = SEGMENT_NAMES[seg_idx]
                joint_ids.update(SEGMENT_NAME_TO_JOINTS[seg_name])
                # 2026-04-27: When an arm segment penetrates the torso, the
                # whole shoulder chain (collar → shoulder → elbow → wrist) is
                # typically misposed. Geometric line-line distance often
                # misses the collar/shoulder root, so include the same-side
                # collar in the invalid mask. This lets QC mask propagate
                # repair to the shoulder root, not just the elbow/wrist.
                if seg_name == "L_upper_arm" or seg_name == "L_forearm":
                    joint_ids.add(13)  # left_collar
                elif seg_name == "R_upper_arm" or seg_name == "R_forearm":
                    joint_ids.add(14)  # right_collar
            joint_ids.update(torso_joint_ids)
            joint_ids_sorted = sorted(joint_ids)
            penetration_joint_ids_per_frame[frame_idx] = joint_ids_sorted
            invalid_mask[frame_idx, joint_ids_sorted] = True

            penetration_frame_details.append(
                {
                    "frame": int(frame_idx),
                    "segment_names": segment_names,
                    "joint_ids": joint_ids_sorted,
                    "min_distance_m": float(np.min(segment_distances)),
                    "worst_segment": worst_segment_name,
                    "torso_segment": torso_segment_name,
                    "depth": float(depth[frame_idx, worst_local]),
                    "side_offset_m": float(side_offset[frame_idx, worst_local]),
                    "height_offset_m": float(height_offset[frame_idx, worst_local]),
                }
            )

        if len(penetration_frames) < min_frames or penetration_ratio < min_ratio:
            return CheckResult(
                is_valid=True,
                invalid_reason="No limb penetration detected",
                invalid_mask=invalid_mask,
                details={
                    "has_penetration": False,
                    "penetration_frames": [],
                    "penetration_ratio": penetration_ratio,
                    "min_dist_per_frame": np.min(min_distance_per_segment, axis=1).astype(float).tolist(),
                    "worst_segment_per_frame": worst_segment_per_frame,
                    "worst_torso_segment_per_frame": worst_torso_segment_per_frame,
                    "penetration_joint_ids_per_frame": penetration_joint_ids_per_frame,
                    "penetration_frame_details": [],
                    "reason": "No limb penetration detected",
                },
            )

        reason = (
            f"Limb penetration in {len(penetration_frames)}/{num_frames} frames "
            f"({penetration_ratio * 100:.1f}%, limb-torso distance < {threshold * 100:.1f} cm)"
        )
        details = {
            "has_penetration": True,
            "penetration_frames": penetration_frames,
            "penetration_ratio": penetration_ratio,
            "min_dist_per_frame": np.min(min_distance_per_segment, axis=1).astype(float).tolist(),
            "min_dist_per_segment": {
                SEGMENT_NAMES[idx]: min_distance_per_segment[:, idx].astype(float).tolist()
                for idx in range(len(SEGMENT_NAMES))
            },
            "worst_segment_per_frame": worst_segment_per_frame,
            "worst_torso_segment_per_frame": worst_torso_segment_per_frame,
            "penetration_joint_ids_per_frame": penetration_joint_ids_per_frame,
            "penetration_frame_details": penetration_frame_details,
            "distance_threshold_m": threshold,
            "reason": reason,
        }
        return CheckResult(
            is_valid=False,
            invalid_reason=reason,
            invalid_mask=invalid_mask,
            details=details,
        )


_CHECKER_CACHE: Dict[str, ArmPenetrationChecker] = {}


def detect_arm_penetration(
    data: Dict,
    device: str = "cpu",
    distance_threshold_m: Optional[float] = None,
    min_penetration_frames: Optional[int] = None,
    min_penetration_ratio: Optional[float] = None,
) -> Dict:
    cache_key = device
    if cache_key not in _CHECKER_CACHE:
        kwargs = {}
        if distance_threshold_m is not None:
            kwargs["distance_threshold_m"] = distance_threshold_m
        if min_penetration_frames is not None:
            kwargs["min_penetration_frames"] = min_penetration_frames
        if min_penetration_ratio is not None:
            kwargs["min_penetration_ratio"] = min_penetration_ratio
        _CHECKER_CACHE[cache_key] = ArmPenetrationChecker(device=device, **kwargs)
    checker = _CHECKER_CACHE[cache_key]
    result = checker.check(data)
    details = result.get("details") or {}
    return {
        "has_penetration": details.get("has_penetration", False),
        "penetration_frames": details.get("penetration_frames", []),
        "reason": details.get("reason", result.get("invalid_reason", "")),
    }
