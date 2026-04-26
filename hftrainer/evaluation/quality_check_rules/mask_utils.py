"""
Helpers for reconstructing per-frame / per-joint invalid masks for quality checkers.

Most legacy checkers expose rich details but do not populate ``invalid_mask``.
This module normalizes checker outputs into a boolean mask of shape ``[T, 22]``
so downstream tools (web UI, reports, versioned JSON exports) can visualize
failed frames / joints consistently.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

NUM_BODY_JOINTS = 22

PENETRATION_SEGMENT_TO_JOINTS = {
    "L_upper_arm": [16, 18],
    "L_forearm": [18, 20],
    "R_upper_arm": [17, 19],
    "R_forearm": [19, 21],
    "L_thigh": [1, 4],
    "L_shin": [4, 7],
    "R_thigh": [2, 5],
    "R_shin": [5, 8],
}

WOBBLE_JOINT_NAME_TO_ID = {
    "Pelvis": 0,
    "LFoot": 7,
    "RFoot": 8,
    "LToeBase": 10,
    "RToeBase": 11,
}


def _num_frames_from_motion(data: Dict[str, Any]) -> int:
    if "poses" in data:
        poses = np.asarray(data["poses"])
        if poses.ndim >= 1:
            return int(poses.shape[0])
    if "trans" in data:
        trans = np.asarray(data["trans"])
        if trans.ndim >= 1:
            return int(trans.shape[0])
    if "transl" in data:
        trans = np.asarray(data["transl"])
        if trans.ndim >= 1:
            return int(trans.shape[0])
    return 0


def empty_invalid_mask(num_frames: int, num_joints: int = NUM_BODY_JOINTS) -> np.ndarray:
    return np.zeros((max(int(num_frames), 0), int(num_joints)), dtype=bool)


def normalize_invalid_mask(
    mask: Optional[np.ndarray],
    num_frames: int,
    num_joints: int = NUM_BODY_JOINTS,
) -> Optional[np.ndarray]:
    if mask is None:
        return None

    arr = np.asarray(mask)
    if arr.size == 0:
        return empty_invalid_mask(num_frames=num_frames, num_joints=num_joints)

    if arr.ndim == 1:
        arr = arr.astype(bool).reshape(-1, 1)
        arr = np.repeat(arr, num_joints, axis=1)
    elif arr.ndim != 2:
        return None

    if arr.shape[1] == 1 and num_joints > 1:
        arr = np.repeat(arr, num_joints, axis=1)
    elif arr.shape[1] != num_joints:
        if arr.shape[0] == num_joints and arr.shape[1] == num_frames:
            arr = arr.T
        elif arr.shape[1] > num_joints:
            arr = arr[:, :num_joints]
        else:
            pad = np.zeros((arr.shape[0], num_joints - arr.shape[1]), dtype=bool)
            arr = np.concatenate([arr.astype(bool), pad], axis=1)

    if arr.shape[0] > num_frames:
        arr = arr[:num_frames]
    elif arr.shape[0] < num_frames:
        pad = np.zeros((num_frames - arr.shape[0], arr.shape[1]), dtype=bool)
        arr = np.concatenate([arr.astype(bool), pad], axis=0)

    return arr.astype(bool)


def merge_invalid_masks(
    masks: Iterable[Optional[np.ndarray]],
    num_frames: int,
    num_joints: int = NUM_BODY_JOINTS,
) -> np.ndarray:
    merged = empty_invalid_mask(num_frames=num_frames, num_joints=num_joints)
    for mask in masks:
        norm = normalize_invalid_mask(mask, num_frames=num_frames, num_joints=num_joints)
        if norm is None:
            continue
        merged |= norm
    return merged


def mark_frames(mask: np.ndarray, start: int, end: int, joints: Optional[Iterable[int]] = None) -> None:
    if mask.size == 0:
        return
    lo = max(int(start), 0)
    hi = min(int(end), mask.shape[0])
    if hi <= lo:
        return
    if joints is None:
        mask[lo:hi, :] = True
        return
    valid_joints = [int(j) for j in joints if 0 <= int(j) < mask.shape[1]]
    if not valid_joints:
        return
    mask[lo:hi, valid_joints] = True


def mark_specific_frames(mask: np.ndarray, frames: Iterable[int], joints: Optional[Iterable[int]] = None) -> None:
    if mask.size == 0:
        return
    valid_joints = None
    if joints is not None:
        valid_joints = [int(j) for j in joints if 0 <= int(j) < mask.shape[1]]
        if not valid_joints:
            return
    for frame in frames:
        fi = int(frame)
        if fi < 0 or fi >= mask.shape[0]:
            continue
        if valid_joints is None:
            mask[fi, :] = True
        else:
            mask[fi, valid_joints] = True


def mask_to_sparse_dict(mask: Optional[np.ndarray]) -> Dict[str, Any]:
    if mask is None:
        return {
            "num_frames": 0,
            "num_joints": 0,
            "invalid_frame_count": 0,
            "invalid_joint_count": 0,
            "frames": {},
        }

    arr = np.asarray(mask, dtype=bool)
    frames: Dict[str, List[int]] = {}
    invalid_frame_indices = np.where(np.any(arr, axis=1))[0].tolist()
    for frame_idx in invalid_frame_indices:
        joint_ids = np.where(arr[frame_idx])[0].astype(int).tolist()
        if joint_ids:
            frames[str(int(frame_idx))] = joint_ids

    return {
        "num_frames": int(arr.shape[0]),
        "num_joints": int(arr.shape[1]) if arr.ndim == 2 else 0,
        "invalid_frame_count": int(len(invalid_frame_indices)),
        "invalid_joint_count": int(np.sum(arr)),
        "frames": frames,
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _build_jitter_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    mask = empty_invalid_mask(num_frames=num_frames)
    details = result.get("details") or {}
    for window in details.get("jitter_windows", []) or []:
        joints = window.get("jitter_joints", []) or []
        # window_start/window_end are defined on the F-2 angle domain.
        frame_start = int(window.get("window_start", 0)) + 1
        frame_end = int(window.get("window_end", frame_start)) + 2
        mark_frames(mask, frame_start, frame_end, joints=joints)
    for item in details.get("accel_jitter_details", []) or []:
        joint_id = int(item.get("joint_id", -1))
        if joint_id < 0:
            continue
        spike_frames = item.get("spike_frames", []) or []
        for frame in spike_frames:
            fi = int(frame)
            mark_specific_frames(mask, [fi - 1, fi, fi + 1], joints=[joint_id])
    for item in details.get("borderline_accel_jitter_details", []) or []:
        joint_id = int(item.get("joint_id", -1))
        if joint_id < 0:
            continue
        spike_frames = item.get("spike_frames", []) or []
        for frame in spike_frames:
            fi = int(frame)
            mark_specific_frames(mask, [fi - 1, fi, fi + 1], joints=[joint_id])
    return mask


def _build_joint_twist_mask(checker: Any, data: Dict[str, Any], _result: Dict[str, Any], num_frames: int) -> np.ndarray:
    from .joint_twist_checker import (
        ARM_JOINTS,
        JOINT_TWIST_AXIS_MAP,
        LEG_JOINTS,
        LEG_TWIST_CONFIGS,
        NECK_HEAD_ANGLE_LIMITS,
        NUM_BODY_JOINTS,
        TWIST_CONFIGS,
        _select_twist_signal,
        extract_joint_twist_metrics,
        is_supported_twist_joint,
    )

    result = _result or {}
    mask = empty_invalid_mask(num_frames=num_frames)
    details = result.get("details") or {}
    twist_details = details.get("twist_details") or []
    if twist_details:
        for detail in twist_details:
            try:
                joint_id = int(detail.get("joint_id", -1))
            except (TypeError, ValueError):
                joint_id = -1
            if joint_id < 0 or joint_id >= mask.shape[1]:
                continue
            for twist_info in detail.get("twist_types") or []:
                frame_indices = [int(f) for f in (twist_info.get("frame_indices") or []) if isinstance(f, (int, np.integer))]
                if frame_indices:
                    mark_specific_frames(mask, frame_indices, joints=[joint_id])
        if np.any(mask):
            return mask

    poses = np.asarray(data.get("poses"))
    if poses.size == 0:
        return mask

    try:
        poses_3d = checker.normalize_poses(poses, NUM_BODY_JOINTS)
    except Exception:
        return mask

    F, J, _ = poses_3d.shape
    if F <= 0:
        return mask

    for joint_id in ARM_JOINTS:
        if joint_id >= J or joint_id not in JOINT_TWIST_AXIS_MAP or not is_supported_twist_joint(joint_id, J):
            continue
        twist_metrics = extract_joint_twist_metrics(poses_3d[:, joint_id, :], JOINT_TWIST_AXIS_MAP[joint_id])
        for twist_type, config in TWIST_CONFIGS.items():
            if joint_id not in config["joints"]:
                continue
            twist_angles = _select_twist_signal(twist_metrics, twist_type)
            deviation = np.abs(np.abs(twist_angles) - config["target_rad"])
            frame_mask = deviation < config["threshold_rad"]
            num_bad = int(np.sum(frame_mask))
            if num_bad >= config["min_frames"] and num_bad >= F * config["min_ratio"]:
                mask[:F, joint_id] |= frame_mask[:F]

    for joint_id in LEG_JOINTS:
        if joint_id >= J or joint_id not in JOINT_TWIST_AXIS_MAP or not is_supported_twist_joint(joint_id, J):
            continue
        twist_metrics = extract_joint_twist_metrics(poses_3d[:, joint_id, :], JOINT_TWIST_AXIS_MAP[joint_id])
        for twist_type, config in LEG_TWIST_CONFIGS.items():
            if joint_id not in config["joints"]:
                continue
            twist_angles = _select_twist_signal(twist_metrics, twist_type)
            deviation = np.abs(np.abs(twist_angles) - config["target_rad"])
            frame_mask = deviation < config["threshold_rad"]
            num_bad = int(np.sum(frame_mask))
            if num_bad >= config["min_frames"] and num_bad >= F * config["min_ratio"]:
                mask[:F, joint_id] |= frame_mask[:F]

    aa_deg = np.rad2deg(poses_3d)
    for joint_id, limits in NECK_HEAD_ANGLE_LIMITS.items():
        if joint_id >= J:
            continue
        frame_mask = (
            (np.abs(aa_deg[:, joint_id, 0]) > limits["x"])
            | (np.abs(aa_deg[:, joint_id, 1]) > limits["y"])
            | (np.abs(aa_deg[:, joint_id, 2]) > limits["z"])
        )
        mask[:F, joint_id] |= frame_mask[:F]

    return mask


def _build_joint_jump_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    mask = empty_invalid_mask(num_frames=num_frames)
    details = result.get("details") or {}
    for item in details.get("jump_details", []) or []:
        frame = int(item.get("frame", -1))
        joint_id = int(item.get("joint_id", -1))
        if joint_id < 0:
            continue
        # A jump is defined across two consecutive displacement vectors:
        # visually highlight the local three-frame span.
        mark_specific_frames(mask, [frame, frame + 1, frame + 2], joints=[joint_id])
    return mask


def _build_arm_penetration_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    mask = empty_invalid_mask(num_frames=num_frames)
    details = result.get("details") or {}
    frame_joint_ids = details.get("penetration_joint_ids_per_frame") or []
    if frame_joint_ids:
        for frame_idx, joint_ids in enumerate(frame_joint_ids):
            if not joint_ids:
                continue
            mark_specific_frames(mask, [frame_idx], joints=joint_ids)
        return mask
    for item in details.get("penetration_frame_details", []) or []:
        frame_idx = int(item.get("frame", -1))
        joint_ids = [int(j) for j in (item.get("joint_ids") or []) if int(j) >= 0]
        if frame_idx < 0 or not joint_ids:
            continue
        mark_specific_frames(mask, [frame_idx], joints=joint_ids)
    if np.any(mask):
        return mask
    worst_segment_per_frame = details.get("worst_segment_per_frame", []) or []
    for frame in details.get("penetration_frames", []) or []:
        fi = int(frame)
        segment = worst_segment_per_frame[fi] if 0 <= fi < len(worst_segment_per_frame) else None
        joints = PENETRATION_SEGMENT_TO_JOINTS.get(segment, [16, 17, 18, 19, 20, 21])
        mark_specific_frames(mask, [fi], joints=joints)
    return mask


def _build_small_wobble_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    mask = empty_invalid_mask(num_frames=num_frames)
    details = result.get("details") or {}
    for window in details.get("wobble_windows", []) or []:
        joint_name = window.get("joint")
        joint_id = WOBBLE_JOINT_NAME_TO_ID.get(joint_name)
        if joint_id is None:
            continue
        active_frames = [int(f) for f in (window.get("active_frames") or []) if int(f) >= 0]
        if active_frames:
            mark_specific_frames(mask, active_frames, joints=[joint_id])
        else:
            mark_frames(mask, int(window.get("start", 0)), int(window.get("end", 0)), joints=[joint_id])
    return mask


def _build_foot_sliding_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    mask = empty_invalid_mask(num_frames=num_frames)
    details = result.get("details") or {}
    for segment in details.get("sliding_segments", []) or []:
        frame_joint_hits = segment.get("frame_joint_hits") or []
        if frame_joint_hits:
            for hit in frame_joint_hits:
                frame = int(hit.get("frame", -1))
                if frame < 0:
                    continue
                joint_ids = [int(j) for j in (hit.get("mask_joint_ids") or []) if int(j) >= 0]
                if joint_ids:
                    mark_specific_frames(mask, [frame], joints=joint_ids)
            continue
        joint_ids = [int(j) for j in (segment.get("joint_ids") or []) if int(j) >= 0]
        if not joint_ids:
            joint_id = int(segment.get("joint_id", -1))
            if joint_id >= 0:
                joint_ids = [joint_id]
        if not joint_ids:
            continue
        start = int(segment.get("start_frame", 0))
        end = int(segment.get("end_frame", start))
        # Segment is defined on velocity indices; include the trailing pose frame.
        mark_frames(mask, start, end + 1, joints=joint_ids)
    return mask


def _build_candy_wrapper_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    existing = normalize_invalid_mask(result.get("invalid_mask"), num_frames=num_frames, num_joints=NUM_BODY_JOINTS)
    if existing is not None:
        return existing
    mask = empty_invalid_mask(num_frames=num_frames)
    details = result.get("details") or {}
    for event in details.get("events", []) or []:
        frame_indices = [int(f) for f in (event.get("frame_indices") or []) if int(f) >= 0]
        joint_ids = [int(j) for j in (event.get("joint_ids") or []) if 0 <= int(j) < NUM_BODY_JOINTS]
        if frame_indices and joint_ids:
            mark_specific_frames(mask, frame_indices, joints=joint_ids)
    return mask


def _build_rotation_velocity_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    mask = empty_invalid_mask(num_frames=num_frames)
    details = result.get("details") or {}
    for violated in details.get("violated_joints", []) or []:
        joint_id = int(violated.get("joint_id", -1))
        if joint_id < 0:
            continue
        spike_frames = violated.get("spike_frames", []) or []
        for frame in spike_frames:
            fi = int(frame)
            mark_specific_frames(mask, [fi, fi + 1], joints=[joint_id])
    return mask


def _build_translation_velocity_mask(checker: Any, data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    mask = empty_invalid_mask(num_frames=num_frames)
    trans = data.get("trans")
    if trans is None and "transl" in data:
        trans = data.get("transl")
    if trans is None:
        return mask

    trans_arr = np.asarray(trans)
    if trans_arr.ndim != 2 or trans_arr.shape[0] < 2 or trans_arr.shape[1] < 3:
        return mask

    delta = trans_arr[1:] - trans_arr[:-1]
    spike_frames = np.where(
        (np.abs(delta[:, 0]) > float(getattr(checker, "threshold_x", 1.0)))
        | (np.abs(delta[:, 1]) > float(getattr(checker, "threshold_y", 1.0)))
        | (np.abs(delta[:, 2]) > float(getattr(checker, "threshold_z", 1.0)))
    )[0].tolist()
    for frame in spike_frames:
        fi = int(frame)
        mark_specific_frames(mask, [fi, fi + 1], joints=None)

    details = result.get("details") or {}
    outlier_frame = int(details.get("outlier_frame", -1))
    if outlier_frame >= 0:
        mark_specific_frames(mask, [outlier_frame, outlier_frame + 1], joints=None)

    return mask


def _build_rotation_validity_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    mask = empty_invalid_mask(num_frames=num_frames)
    details = result.get("details") or {}
    failed_joint_ids = [int(j) for j in (details.get("failed_joint_ids") or []) if int(j) >= 0]
    existing = normalize_invalid_mask(result.get("invalid_mask"), num_frames=num_frames, num_joints=NUM_BODY_JOINTS)
    if existing is not None:
        return existing
    if failed_joint_ids:
        mark_frames(mask, 0, num_frames, joints=failed_joint_ids)
    return mask


def _build_duration_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    existing = normalize_invalid_mask(result.get("invalid_mask"), num_frames=num_frames, num_joints=NUM_BODY_JOINTS)
    if existing is not None:
        return existing
    mask = empty_invalid_mask(num_frames=num_frames)
    severity = str(result.get("severity") or ("fail" if not result.get("is_valid", True) else "pass")).strip().lower()
    if severity in {"fail", "borderline"}:
        mark_frames(mask, 0, num_frames, joints=None)
    return mask


def _build_rest_pose_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    return _build_duration_mask(_checker, _data, result, num_frames)


def _build_ground_penetration_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    existing = normalize_invalid_mask(result.get("invalid_mask"), num_frames=num_frames, num_joints=NUM_BODY_JOINTS)
    if existing is not None:
        return existing
    mask = empty_invalid_mask(num_frames=num_frames)
    details = result.get("details") or {}
    joint_ids = [int(j) for j in (details.get("joint_ids") or [7, 8, 10, 11]) if 0 <= int(j) < NUM_BODY_JOINTS]
    if joint_ids:
        mark_frames(mask, 0, num_frames, joints=joint_ids)
    return mask


def _build_first_last_rotation_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    existing = normalize_invalid_mask(result.get("invalid_mask"), num_frames=num_frames, num_joints=NUM_BODY_JOINTS)
    if existing is not None:
        return existing
    mask = empty_invalid_mask(num_frames=num_frames)
    reason = str(result.get("invalid_reason") or "")
    if "First-frame" in reason or "首帧" in reason:
        mark_frames(mask, 0, min(2, num_frames), joints=None)
    if "Last-frame" in reason or "末帧" in reason:
        mark_frames(mask, max(num_frames - 2, 0), num_frames, joints=None)
    if not np.any(mask):
        details = result.get("details") or {}
        if float(details.get("first_frame_rotation_angle_deg", 0.0)) > 0.0:
            mark_frames(mask, 0, min(2, num_frames), joints=None)
        if float(details.get("last_frame_rotation_angle_deg", 0.0)) > 0.0:
            mark_frames(mask, max(num_frames - 2, 0), num_frames, joints=None)
    return mask


def _build_joint_range_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int, joint_ids: List[int]) -> np.ndarray:
    existing = normalize_invalid_mask(result.get("invalid_mask"), num_frames=num_frames, num_joints=NUM_BODY_JOINTS)
    if existing is not None:
        return existing
    mask = empty_invalid_mask(num_frames=num_frames)
    severity = str(result.get("severity") or ("fail" if not result.get("is_valid", True) else "pass")).strip().lower()
    if severity in {"fail", "borderline"}:
        mark_frames(mask, 0, num_frames, joints=joint_ids)
    return mask


def _build_knee_x_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    return _build_joint_range_mask(_checker, _data, result, num_frames, [4, 5])


def _build_ankle_x_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    return _build_joint_range_mask(_checker, _data, result, num_frames, [7, 8])


def _build_neck_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int) -> np.ndarray:
    return _build_joint_range_mask(_checker, _data, result, num_frames, [12, 15])


def _build_spine_mask(_checker: Any, _data: Dict[str, Any], result: Dict[str, Any], num_frames: int, joint_id: int) -> np.ndarray:
    return _build_joint_range_mask(_checker, _data, result, num_frames, [joint_id])


MASK_BUILDERS = {
    "jitter": _build_jitter_mask,
    "joint_twist": _build_joint_twist_mask,
    "joint_jump": _build_joint_jump_mask,
    "arm_penetration": _build_arm_penetration_mask,
    "small_wobble": _build_small_wobble_mask,
    "foot_sliding": _build_foot_sliding_mask,
    "rotation_velocity": _build_rotation_velocity_mask,
    "translation_velocity": _build_translation_velocity_mask,
    "rotation_validity": _build_rotation_validity_mask,
    "duration": _build_duration_mask,
    "rest_pose": _build_rest_pose_mask,
    "ground_penetration": _build_ground_penetration_mask,
    "first_frame_rotation_velocity": _build_first_last_rotation_mask,
    "knee_x": _build_knee_x_mask,
    "ankle_x": _build_ankle_x_mask,
    "neck": _build_neck_mask,
    "spine": lambda checker, data, result, num_frames: _build_spine_mask(checker, data, result, num_frames, 3),
    "spine1": lambda checker, data, result, num_frames: _build_spine_mask(checker, data, result, num_frames, 6),
    "spine2": lambda checker, data, result, num_frames: _build_spine_mask(checker, data, result, num_frames, 9),
    "candy_wrapper": _build_candy_wrapper_mask,
}


def build_invalid_mask(
    checker_name: str,
    checker: Any,
    data: Dict[str, Any],
    result: Dict[str, Any],
    num_joints: int = NUM_BODY_JOINTS,
) -> np.ndarray:
    num_frames = _num_frames_from_motion(data)
    existing = normalize_invalid_mask(
        result.get("invalid_mask"),
        num_frames=num_frames,
        num_joints=num_joints,
    )
    if existing is not None:
        return existing

    builder = MASK_BUILDERS.get(checker_name)
    if builder is None:
        severity = str(result.get("severity") or ("fail" if not result.get("is_valid", True) else "pass")).strip().lower()
        if severity in {"fail", "borderline"}:
            fallback = empty_invalid_mask(num_frames=num_frames, num_joints=num_joints)
            details = result.get("details") or {}
            failed_joint_ids = [int(j) for j in (details.get("failed_joint_ids") or []) if 0 <= int(j) < num_joints]
            if failed_joint_ids:
                mark_frames(fallback, 0, num_frames, joints=failed_joint_ids)
                return fallback
            mark_frames(fallback, 0, num_frames, joints=None)
            return fallback
        return empty_invalid_mask(num_frames=num_frames, num_joints=num_joints)

    try:
        mask = builder(checker, data, result, num_frames)
    except Exception:
        mask = None

    normalized = normalize_invalid_mask(mask, num_frames=num_frames, num_joints=num_joints)
    if normalized is None:
        return empty_invalid_mask(num_frames=num_frames, num_joints=num_joints)
    return normalized
