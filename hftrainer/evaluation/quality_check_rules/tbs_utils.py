from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

NUM_BODY_JOINTS = 22

SMPL22_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int32,
)

# Kept aligned with the rest-body offsets used by candy-wrapper corruptors.
REST_OFFSETS_BODY_22 = np.zeros((NUM_BODY_JOINTS, 3), dtype=np.float64)
REST_OFFSETS_BODY_22[1] = [-0.09, -0.22, 0.02]
REST_OFFSETS_BODY_22[2] = [0.09, -0.22, 0.02]
REST_OFFSETS_BODY_22[3] = [0.0, 0.22, 0.0]
REST_OFFSETS_BODY_22[4] = [0.0, -0.43, 0.0]
REST_OFFSETS_BODY_22[5] = [0.0, -0.43, 0.0]
REST_OFFSETS_BODY_22[6] = [0.0, 0.22, 0.0]
REST_OFFSETS_BODY_22[7] = [0.05, -0.42, 0.0]
REST_OFFSETS_BODY_22[8] = [-0.05, -0.42, 0.0]
REST_OFFSETS_BODY_22[9] = [0.0, 0.22, 0.0]
REST_OFFSETS_BODY_22[10] = [0.05, -0.10, 0.05]
REST_OFFSETS_BODY_22[11] = [-0.05, -0.10, 0.05]
REST_OFFSETS_BODY_22[12] = [0.0, 0.15, 0.0]
REST_OFFSETS_BODY_22[13] = [-0.18, 0.08, 0.0]
REST_OFFSETS_BODY_22[14] = [0.18, 0.08, 0.0]
REST_OFFSETS_BODY_22[15] = [0.0, 0.20, 0.0]
REST_OFFSETS_BODY_22[16] = [-0.28, 0.0, 0.0]
REST_OFFSETS_BODY_22[17] = [0.28, 0.0, 0.0]
REST_OFFSETS_BODY_22[18] = [-0.25, 0.0, 0.0]
REST_OFFSETS_BODY_22[19] = [0.25, 0.0, 0.0]
REST_OFFSETS_BODY_22[20] = [-0.12, 0.0, 0.0]
REST_OFFSETS_BODY_22[21] = [0.12, 0.0, 0.0]

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


@dataclass(frozen=True)
class TbsJointSpec:
    joint_id: int
    joint_name: str
    parent_id: int
    twist_child_id: int
    rest_bone_dir: np.ndarray
    u_rest: np.ndarray
    v_rest: np.ndarray


def _normalize(vec: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        if fallback is None:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        arr = np.asarray(fallback, dtype=np.float64)
        n = float(np.linalg.norm(arr))
        if n < 1e-12:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return arr / n


def wrap_angle_deg(deg: np.ndarray) -> np.ndarray:
    arr = np.asarray(deg, dtype=np.float64)
    wrapped = np.mod(arr + 180.0, 360.0) - 180.0
    wrapped[np.isclose(wrapped, -180.0)] = 180.0
    return wrapped


def wrap_angle_rad(rad: np.ndarray) -> np.ndarray:
    arr = np.asarray(rad, dtype=np.float64)
    wrapped = np.mod(arr + np.pi, 2.0 * np.pi) - np.pi
    wrapped[np.isclose(wrapped, -np.pi)] = np.pi
    return wrapped


def _build_rest_world_positions() -> np.ndarray:
    rest_pos = np.zeros((NUM_BODY_JOINTS, 3), dtype=np.float64)
    for joint_id in range(1, NUM_BODY_JOINTS):
        parent_id = int(SMPL22_PARENTS[joint_id])
        rest_pos[joint_id] = rest_pos[parent_id] + REST_OFFSETS_BODY_22[joint_id]
    return rest_pos


def _build_body_frame(rest_world_pos: np.ndarray) -> Dict[str, np.ndarray]:
    pelvis = rest_world_pos[0]
    neck = rest_world_pos[12]
    left_anchor = rest_world_pos[16]
    right_anchor = rest_world_pos[17]

    up = _normalize(neck - pelvis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    left = _normalize(left_anchor - right_anchor, np.array([1.0, 0.0, 0.0], dtype=np.float64))

    forward = np.cross(left, up)
    forward = _normalize(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if np.dot(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64)) < 0.0:
        forward = -forward

    left_ortho = np.cross(up, forward)
    left_ortho = _normalize(left_ortho, left)
    if np.dot(left_ortho, left) < 0.0:
        left_ortho = -left_ortho

    return {
        "up": up,
        "down": -up,
        "left": left_ortho,
        "right": -left_ortho,
        "forward": forward,
        "back": -forward,
    }


def _project_preferred_bend_to_rest_plane(rest_bone: np.ndarray, preferred_dir: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if preferred_dir is None:
        return None
    projected = preferred_dir - rest_bone * float(np.dot(preferred_dir, rest_bone))
    n = float(np.linalg.norm(projected))
    if n < 1e-12:
        return None
    return projected / n


def _preferred_side_axis_for_joint(joint_id: int, body_frame: Dict[str, np.ndarray]) -> np.ndarray:
    if joint_id in {1, 4, 7, 10, 13, 16, 18, 20}:
        return body_frame["left"]
    if joint_id in {2, 5, 8, 11, 14, 17, 19, 21}:
        return body_frame["right"]
    return body_frame["left"]


def _preferred_bend_direction_for_joint(joint_id: int, body_frame: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if joint_id in {0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17}:
        return body_frame["forward"]
    if joint_id in {4, 5, 7, 8, 10, 11}:
        return _preferred_side_axis_for_joint(joint_id, body_frame)
    if joint_id in {18, 19, 20, 21}:
        return body_frame["up"]
    return None


def _child_indices_from_parents() -> np.ndarray:
    child_indices = np.full((NUM_BODY_JOINTS,), -1, dtype=np.int32)
    for joint_id in range(1, NUM_BODY_JOINTS):
        parent_id = int(SMPL22_PARENTS[joint_id])
        if child_indices[parent_id] < 0:
            child_indices[parent_id] = joint_id
    if int(SMPL22_PARENTS[3]) == 0:
        child_indices[0] = 3
    return child_indices


@lru_cache(maxsize=1)
def get_tbs_joint_specs() -> Dict[int, TbsJointSpec]:
    rest_world_pos = _build_rest_world_positions()
    body_frame = _build_body_frame(rest_world_pos)
    child_indices = _child_indices_from_parents()
    fallback_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    twist_axes: Dict[int, np.ndarray] = {}
    specs: Dict[int, TbsJointSpec] = {}
    basis_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    basis_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    basis_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for joint_id in range(NUM_BODY_JOINTS):
        child_id = int(child_indices[joint_id])
        if child_id >= 0:
            direction = rest_world_pos[child_id] - rest_world_pos[joint_id]
        else:
            parent_id = int(SMPL22_PARENTS[joint_id])
            direction = twist_axes[parent_id].copy() if parent_id >= 0 and parent_id in twist_axes else fallback_axis.copy()
        rest_bone = _normalize(direction, fallback_axis)
        twist_axes[joint_id] = rest_bone

        preferred_dir = _preferred_bend_direction_for_joint(joint_id, body_frame)
        u_rest = _project_preferred_bend_to_rest_plane(rest_bone, preferred_dir)
        if u_rest is None:
            u_rest = np.cross(rest_bone, basis_x)
            if np.linalg.norm(u_rest) < 1e-12:
                u_rest = np.cross(rest_bone, basis_y)
            if np.linalg.norm(u_rest) < 1e-12:
                u_rest = np.cross(rest_bone, basis_z)
            u_rest = _normalize(u_rest, basis_y)
        else:
            u_rest = _normalize(u_rest, basis_y)
        v_rest = _normalize(np.cross(rest_bone, u_rest), basis_z)

        specs[joint_id] = TbsJointSpec(
            joint_id=joint_id,
            joint_name=JOINT_NAMES.get(joint_id, f"Joint_{joint_id}"),
            parent_id=int(SMPL22_PARENTS[joint_id]),
            twist_child_id=child_id,
            rest_bone_dir=rest_bone.astype(np.float64),
            u_rest=u_rest.astype(np.float64),
            v_rest=v_rest.astype(np.float64),
        )
    return specs


def get_tbs_joint_spec(joint_id: int) -> TbsJointSpec:
    specs = get_tbs_joint_specs()
    if joint_id not in specs:
        raise KeyError(f"Unsupported TBS joint id: {joint_id}")
    return specs[joint_id]


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    inv = q.copy()
    inv[..., :3] *= -1.0
    denom = np.sum(q * q, axis=-1, keepdims=True)
    return inv / np.clip(denom, 1e-12, None)


def _quat_from_two_vectors(rest_bone: np.ndarray, rotated_bone: np.ndarray, fallback_axis: np.ndarray) -> np.ndarray:
    a = _normalize(rest_bone)
    b = np.asarray(rotated_bone, dtype=np.float64)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    b = b / np.clip(b_norm, 1e-12, None)
    cross = np.cross(np.broadcast_to(a, b.shape), b)
    dot = np.sum(a[None, :] * b, axis=-1)
    q = np.zeros((b.shape[0], 4), dtype=np.float64)
    regular = dot > (-1.0 + 1e-8)
    if np.any(regular):
        q[regular, :3] = cross[regular]
        q[regular, 3] = 1.0 + dot[regular]
    if np.any(~regular):
        axis = fallback_axis - a * float(np.dot(fallback_axis, a))
        if np.linalg.norm(axis) < 1e-12:
            axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-12:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = _normalize(axis, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        q[~regular, :3] = axis[None, :]
        q[~regular, 3] = 0.0
    q /= np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    return q


def extract_joint_tbs_metrics(axis_angle: np.ndarray, joint_id: int) -> Dict[str, np.ndarray]:
    spec = get_tbs_joint_spec(int(joint_id))
    rotvec = np.asarray(axis_angle, dtype=np.float64).reshape(-1, 3)
    if rotvec.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "twist_deg": empty,
            "bend_deg": empty,
            "spread_deg": empty,
            "swing_axis_deg": empty,
            "swing_mag_deg": empty,
            "raw_aligned_deg": empty,
        }

    rotation = R.from_rotvec(rotvec)
    quat = rotation.as_quat()  # x, y, z, w
    rest_bone = spec.rest_bone_dir
    u_rest = spec.u_rest
    v_rest = spec.v_rest

    rotated_bone = rotation.apply(np.broadcast_to(rest_bone, rotvec.shape))
    rotated_bone = rotated_bone / np.clip(np.linalg.norm(rotated_bone, axis=-1, keepdims=True), 1e-12, None)

    dot_rt = np.clip(np.sum(rotated_bone * rest_bone[None, :], axis=-1), -1.0, 1.0)
    swing_mag_rad = np.arccos(dot_rt)
    swing_axis = np.cross(np.broadcast_to(rest_bone, rotated_bone.shape), rotated_bone)
    swing_axis_norm = np.linalg.norm(swing_axis, axis=-1, keepdims=True)
    swing_axis_unit = np.divide(swing_axis, np.clip(swing_axis_norm, 1e-12, None))

    swing_axis_deg = np.zeros((rotvec.shape[0],), dtype=np.float64)
    valid_axis = swing_axis_norm[:, 0] >= 1e-8
    if np.any(valid_axis):
        du = np.sum(swing_axis_unit[valid_axis] * u_rest[None, :], axis=-1)
        dv = np.sum(swing_axis_unit[valid_axis] * v_rest[None, :], axis=-1)
        swing_axis_deg[valid_axis] = np.degrees(np.arctan2(dv, du))

    qswing = _quat_from_two_vectors(rest_bone, rotated_bone, u_rest)
    qtwist = _quat_multiply(quat, _quat_inverse(qswing))
    dot_twist = np.sum(qtwist[:, :3] * rotated_bone, axis=-1)
    twist_rad = 2.0 * np.arctan2(dot_twist, qtwist[:, 3])
    twist_deg = wrap_angle_deg(np.degrees(wrap_angle_rad(twist_rad)))

    swing_mag_deg = np.clip(np.degrees(swing_mag_rad), 0.0, 180.0)
    bend_deg = swing_mag_deg * np.cos(np.deg2rad(swing_axis_deg))
    spread_deg = swing_mag_deg * np.sin(np.deg2rad(swing_axis_deg))

    raw_aligned_deg = np.degrees(rotvec @ rest_bone)
    if raw_aligned_deg.shape[0] > 1:
        unwrapped = raw_aligned_deg.copy()
        for idx in range(1, unwrapped.shape[0]):
            cur = unwrapped[idx]
            prev = unwrapped[idx - 1]
            while cur - prev > 180.0:
                cur -= 360.0
            while cur - prev < -180.0:
                cur += 360.0
            unwrapped[idx] = cur
        raw_aligned_deg = unwrapped

    return {
        "twist_deg": twist_deg.astype(np.float64),
        "bend_deg": bend_deg.astype(np.float64),
        "spread_deg": spread_deg.astype(np.float64),
        "swing_axis_deg": wrap_angle_deg(swing_axis_deg).astype(np.float64),
        "swing_mag_deg": swing_mag_deg.astype(np.float64),
        "raw_aligned_deg": raw_aligned_deg.astype(np.float64),
    }

