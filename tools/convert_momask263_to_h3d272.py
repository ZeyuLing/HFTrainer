#!/usr/bin/env python3
"""HumanML3D-263 <-> humanml3d_272 bridging utilities.

This module supplies the *decode 263 -> pose* and *encode pose -> 272* halves of
the round-trip pipeline used to evaluate motions under both the HumanML3D
official (263-dim, 20fps) evaluator and the MotionStreamer (272-dim, 30fps)
evaluator.

Convention notes (verified against reference implementations):
  * HumanML3D-263 packs joint rotations as continuous-6D = [col0, col1] of the
    rotation matrix (COLUMN-major), via ``common.quaternion.quaternion_to_cont6d``.
  * humanml3d_272 packs joint rotations as the first two ROWS of the matrix
    (ROW-major), matching pytorch3d ``matrix_to_rotation_6d``.
  * The root orientation in 263 is reconstructed as a pure-Y yaw from the
    half-angle ``rot_velocity`` channel (lossy: pitch/roll of the root are not
    stored in 263).

All bridging is done through full 3x3 rotation matrices so the differing 6D
packings never leak across representations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# --- make MoMask's motion_process / quaternion utilities importable ----------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MOMASK = _REPO_ROOT / "ref_repo" / "Momask" / "momask-codes"
if _MOMASK.is_dir() and str(_MOMASK) not in sys.path:
    sys.path.insert(0, str(_MOMASK))

from common.quaternion import (  # noqa: E402
    quaternion_to_matrix,
    cont6d_to_matrix,
)
from utils.motion_process import (  # noqa: E402
    recover_from_ric,
    recover_root_rot_pos,
)

_NJOINT = 22


# ============================ resampling ====================================

def linear_resample_positions(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linearly resample a time-major array ``(T, ...)`` from ``src_fps`` to ``dst_fps``."""
    arr = np.asarray(arr)
    T = arr.shape[0]
    if T < 2 or abs(src_fps - dst_fps) < 1e-9:
        return arr.copy()
    duration = (T - 1) / src_fps
    new_T = int(round(duration * dst_fps)) + 1
    new_T = max(new_T, 2)
    src_t = np.arange(T) / src_fps
    dst_t = np.arange(new_T) / dst_fps
    dst_t = np.clip(dst_t, src_t[0], src_t[-1])
    flat = arr.reshape(T, -1)
    out = np.empty((new_T, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst_t, src_t, flat[:, c])
    return out.reshape((new_T,) + arr.shape[1:])


def slerp_rotations(R: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Spherical-linear resample rotation matrices ``(T, J, 3, 3)`` between fps."""
    from scipy.spatial.transform import Rotation as _Rot, Slerp as _Slerp

    R = np.asarray(R)
    T = R.shape[0]
    if T < 2 or abs(src_fps - dst_fps) < 1e-9:
        return R.copy()
    duration = (T - 1) / src_fps
    new_T = int(round(duration * dst_fps)) + 1
    new_T = max(new_T, 2)
    src_t = np.arange(T) / src_fps
    dst_t = np.arange(new_T) / dst_fps
    dst_t = np.clip(dst_t, src_t[0], src_t[-1])
    J = R.shape[1]
    out = np.empty((new_T, J, 3, 3), dtype=np.float64)
    for j in range(J):
        key = _Rot.from_matrix(R[:, j])
        out[:, j] = _Slerp(src_t, key)(dst_t).as_matrix()
    return out


# ============================ 263 -> pose ===================================

def decode_263_to_pose(m263: np.ndarray):
    """Decode a HumanML3D-263 vector to global joint positions + local rotations.

    Args:
        m263: ``(T, 263)`` HumanML3D feature (un-normalized).

    Returns:
        positions: ``(T, 22, 3)`` global joint positions.
        R_all:     ``(T, 22, 3, 3)`` per-joint rotation matrices. Joint 0 is the
                   recovered global root yaw; joints 1..21 are parent-relative
                   local rotations (decoded from the column-major 6D block).
        feet:      ``(T, 4)`` foot-contact channel (passthrough).
    """
    m263 = np.asarray(m263, dtype=np.float32)
    data = torch.from_numpy(m263).float().unsqueeze(0)  # (1, T, 263)

    positions = recover_from_ric(data, _NJOINT).squeeze(0).numpy()  # (T, 22, 3)

    r_rot_quat, _ = recover_root_rot_pos(data)
    r_rot_quat = r_rot_quat.squeeze(0)  # (T, 4)  pure-yaw quaternion (w,x,y,z)
    R_root = quaternion_to_matrix(r_rot_quat).numpy()  # (T, 3, 3)

    T = m263.shape[0]
    start = 1 + 2 + 1 + (_NJOINT - 1) * 3            # = 67
    end = start + (_NJOINT - 1) * 6                  # = 193
    cont6d = torch.from_numpy(m263[:, start:end]).float().view(T, _NJOINT - 1, 6)
    R_nonroot = cont6d_to_matrix(cont6d).numpy()     # (T, 21, 3, 3)

    R_all = np.concatenate([R_root[:, None], R_nonroot], axis=1)  # (T, 22, 3, 3)
    feet = m263[:, 259:263].copy()
    return positions, R_all, feet


# ============================ pose -> 272 ===================================

def _rot_yaw(yaw: float) -> np.ndarray:
    cs, sn = np.cos(yaw), np.sin(yaw)
    return np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])


def encode_h3d272(positions: np.ndarray, R_local: np.ndarray) -> np.ndarray:
    """Encode global positions + local rotation matrices into a 272-dim vector.

    Faithful re-implementation of MotionStreamer's ``representation_272`` /
    ``compute_representation_272`` taking rotation matrices directly (instead of
    SMPL axis-angle), so it can be fed decoded poses.

    Args:
        positions: ``(T, 22, 3)`` global joint positions.
        R_local:   ``(T, 22, 3, 3)`` per-joint rotation matrices (root is global
                   orientation, joints 1..21 parent-relative local).

    Returns:
        ``(T, 272)`` representation.
    """
    positions = np.asarray(positions, dtype=np.float64)
    rotations_matrix = np.asarray(R_local, dtype=np.float64).copy()
    nfrm, njoint = positions.shape[0], positions.shape[1]
    assert njoint == _NJOINT, njoint
    root_idx = 0

    position_data = positions.copy()
    # put on floor & center first-frame root at origin
    ori = position_data[0, root_idx].copy()
    ori[1] = np.min(position_data[:, :, 1])
    position_data = position_data - ori

    velocities_root = position_data[1:, root_idx, :] - position_data[:-1, root_idx, :]

    # every frame: root at xz origin
    position_data[:, :, 0] -= position_data[:, 0:1, 0]
    position_data[:, :, 2] -= position_data[:, 0:1, 2]

    # heading from root rotation matrix
    global_heading = -np.arctan2(rotations_matrix[:, root_idx, 0, 2],
                                 rotations_matrix[:, root_idx, 2, 2])
    global_heading_rot = np.array([_rot_yaw(x) for x in global_heading])
    global_heading_diff = global_heading[1:] - global_heading[:-1]
    global_heading_diff_rot = np.array([_rot_yaw(x) for x in global_heading_diff])

    positions_no_heading = np.matmul(
        np.repeat(global_heading_rot[:, None, :, :], njoint, axis=1),
        position_data[..., None]).squeeze(-1)
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]
    velocities_root_xy_no_heading = np.matmul(
        global_heading_rot[:-1], velocities_root[:, :, None]).squeeze(-1)[..., [0, 2]]

    rotations_matrix[:, 0, ...] = np.matmul(global_heading_rot, rotations_matrix[:, 0, ...])

    final_x = np.zeros((nfrm, 8 + njoint * 12))
    final_x[0, 2] = 1
    final_x[0, 6] = 1
    if nfrm > 1:
        final_x[1:, 2:8] = global_heading_diff_rot[:, :2, :].reshape(-1, 6)
        final_x[1:, :2] = velocities_root_xy_no_heading
        final_x[1:, 8 + 3 * njoint:8 + 6 * njoint] = velocities_no_heading.reshape(nfrm - 1, -1)
    final_x[:, 8:8 + 3 * njoint] = positions_no_heading.reshape(nfrm, -1)
    final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = rotations_matrix[:, :, :2, :].reshape(nfrm, -1)
    return final_x
