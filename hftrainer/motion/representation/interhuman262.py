"""InterHuman / InterGen ``interhuman_262`` representation (encode/decode).

Faithful, self-contained re-implementation of the InterGen data pipeline
(``utils.utils.process_motion_np`` + ``rigid_transform`` from the official
InterGen repo, https://github.com/tr3e/InterGen). See
:class:`hftrainer.motion.representation.specs.IH262` for the channel layout.

262 layout (per person, per frame, ``njoint=22``)::

    [0:66]     22 joint positions (22x3), canonical (Y-up, floor at 0,
               first-frame root xz at origin, facing +Z)
    [66:132]   22 joint velocities (22x3), forward difference
    [132:258]  21 NON-root joint local rot6d (21x6), ROW-major
    [258:262]  4 foot-contact flags (L heel/toe, R heel/toe)

Key facts (reverse-engineered + validated against official ``motions_processed``,
matrix MSE ~1e-4, InterCLIP R@3 0.830 vs official 0.835 on the same subset):

- **rotations are SMPL ``body_pose`` local rotations** (NOT IK, NOT global), one
  per non-root body joint, encoded as 6D in **ROW-major** layout
  ``[R00,R01,R10,R11,R20,R21] = [c0x,c1x,c0y,c1y,c0z,c1z]`` (same convention as
  :data:`hftrainer.motion.representation.specs.MS272`, i.e.
  ``rotation.Rot6DConvention.ROW``). This is the historical gotcha: using the
  COLUMN layout (``[c0x,c0y,c0z,c1x,c1y,c1z]``) silently drops ~0.3 R@3.
- positions enter the pipeline in a **z-up raw frame** (official ``.npy[:,:66]``
  ``= Mt @ smplx_joints``); ``process_motion_np`` then applies ``trans_matrix``
  (z->y) plus floor / xz-origin / face-+Z canonicalisation. Rotations are passed
  through UNCHANGED (no canonical rotation applied to the 21-joint block).
- **encode drops the last frame** (output length ``T-1``): positions/rot use
  ``[:-1]`` and velocities are forward differences.
- two-person: person2 is rigidly aligned to person1's first-frame heading+xz via
  :func:`rigid_transform` (see :func:`build_pair`).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from hftrainer.motion.representation.rotation import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

# --------------------------------------------------------------------------- #
# Constants (match InterGen utils.utils)
# --------------------------------------------------------------------------- #
# z-up -> y-up coordinate change applied to positions inside process_motion_np.
TRANS_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], np.float32)
# y-up smplx joints -> z-up raw frame (== TRANS_MATRIX.T); official .npy[:,:66].
_MT = TRANS_MATRIX.T
FACE_JOINT_IDX = [2, 1, 17, 16]  # r_hip, l_hip, sdr_r, sdr_l
FID_L = [7, 10]                  # left foot (ankle, toe)
FID_R = [8, 11]                  # right foot
NUM_JOINTS = 22
FEET_THRE = 0.001


# --------------------------------------------------------------------------- #
# Minimal numpy quaternion helpers (ported from InterGen utils.quaternion)
# quaternions are (w, x, y, z)
# --------------------------------------------------------------------------- #
def _qmul(q, r):
    q = np.asarray(q, np.float32)
    r = np.asarray(r, np.float32)
    w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    # NOTE: InterGen computes outer(r, q) -> q*r; replicate q*r exactly.
    w = w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0
    x = w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0
    y = w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0
    z = w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0
    return np.stack([w, x, y, z], axis=-1)


def _qinv(q):
    q = np.asarray(q, np.float32).copy()
    q[..., 1:] *= -1.0
    return q


def _qrot(q, v):
    q = np.asarray(q, np.float32)
    v = np.asarray(v, np.float32)
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def _qnormalize(q):
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def _qbetween(v0, v1):
    """Quaternion rotating v0 onto v1 (both ``(...,3)``)."""
    v = np.cross(v0, v1)
    w = np.sqrt((v0 ** 2).sum(-1, keepdims=True) * (v1 ** 2).sum(-1, keepdims=True)) + (
        v0 * v1
    ).sum(-1, keepdims=True)
    return _qnormalize(np.concatenate([w, v], axis=-1))


# --------------------------------------------------------------------------- #
# rot6d helpers (ROW-major, component-interleaved [c0x,c1x,c0y,c1y,c0z,c1z])
# --------------------------------------------------------------------------- #
def body_pose_to_rot6d_row(body_pose_aa: np.ndarray) -> np.ndarray:
    """SMPL ``body_pose`` axis-angle ``(...,21,3)`` -> ROW-major rot6d ``(...,21,6)``.

    This is the exact packing stored in the 262 ``body_rot6d`` block.
    """
    aa = np.asarray(body_pose_aa, np.float32)
    shp = aa.shape[:-1]
    R = axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(shp + (3, 3))
    return matrix_to_rotation_6d(R, convention="row").astype(np.float32)


def rot6d_row_to_matrix(rot6d_row: np.ndarray) -> np.ndarray:
    """Inverse of :func:`body_pose_to_rot6d_row` to rotation matrices ``(...,3,3)``."""
    return rotation_6d_to_matrix(np.asarray(rot6d_row, np.float32), convention="row")


# --------------------------------------------------------------------------- #
# Core encode (port of InterGen process_motion_np)
# --------------------------------------------------------------------------- #
def _process_motion(
    positions_zup: np.ndarray,
    rot6d_row: np.ndarray,
    feet_thre: float = FEET_THRE,
    prev_frames: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalise + assemble one person to the 262 vector.

    Args:
        positions_zup: ``(T,22,3)`` joint positions in the z-up raw frame.
        rot6d_row: ``(T,21,6)`` non-root body rot6d, ROW-major (passed through).
        feet_thre: foot-contact velocity threshold.
        prev_frames: index of the reference (init) frame (InterGen uses 0).

    Returns:
        ``(data (T-1,262), root_quat_init (1,4), root_pos_init_xz (1,3))``.
    """
    positions = np.einsum("mn,tjn->tjm", TRANS_MATRIX, positions_zup.astype(np.float32))

    # put on floor
    floor_h = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_h

    # xz at origin (reference frame root)
    root_pos_init = positions[prev_frames]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1], np.float32)
    positions = positions - root_pose_init_xz

    # face +Z
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_IDX
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / (np.linalg.norm(across) + 1e-12)
    forward_init = np.cross(np.array([[0, 1, 0]], np.float32), across)
    forward_init = forward_init / (np.linalg.norm(forward_init, axis=-1, keepdims=True) + 1e-12)
    target = np.array([[0, 0, 1]], np.float32)
    root_quat_init = _qbetween(forward_init, target)
    root_quat_all = np.ones(positions.shape[:-1] + (4,), np.float32) * root_quat_init
    positions = _qrot(root_quat_all, positions)

    # foot contact
    def _foot(pos, fid):
        d = (pos[1:, fid] - pos[:-1, fid]) ** 2
        feet = (d.sum(-1) < feet_thre) & (pos[:-1, fid, 1] < 0.12)
        return feet.astype(np.float32)

    feet_l = _foot(positions, FID_L)
    feet_r = _foot(positions, FID_R)

    joint_pos = positions.reshape(len(positions), -1)
    joint_vel = (positions[1:] - positions[:-1]).reshape(len(positions) - 1, -1)
    rot_flat = rot6d_row.reshape(len(rot6d_row), -1)

    data = joint_pos[:-1]
    data = np.concatenate([data, joint_vel], axis=-1)
    data = np.concatenate([data, rot_flat[:-1]], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    return data.astype(np.float32), root_quat_init, root_pose_init_xz[None]


def rigid_transform(relative: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Align a 262 vector's positions/velocities by ``relative=[angle, dx, dz]``.

    Used to express person2 in person1's first-frame heading + xz frame.
    """
    data = data.copy()
    gp = data[..., : 22 * 3].reshape(data.shape[:-1] + (22, 3))
    gv = data[..., 22 * 3 : 22 * 6].reshape(data.shape[:-1] + (22, 3))
    rel_rot = relative[0]
    rel_t = relative[1:3]
    q = np.zeros(gp.shape[:-1] + (4,), np.float32)
    q[..., 0] = np.cos(rel_rot)
    q[..., 2] = np.sin(rel_rot)
    gp = _qrot(_qinv(q), gp)
    gp[..., [0, 2]] += rel_t
    data[..., : 22 * 3] = gp.reshape(data.shape[:-1] + (-1,))
    gv = _qrot(_qinv(q), gv)
    data[..., 22 * 3 : 22 * 6] = gv.reshape(data.shape[:-1] + (-1,))
    return data


# --------------------------------------------------------------------------- #
# Public encode API
# --------------------------------------------------------------------------- #
def encode_smpl_to_interhuman262(
    joints_world: np.ndarray,
    body_pose_aa: np.ndarray,
    *,
    max_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode one person (SMPL-X) to the interhuman_262 representation.

    Args:
        joints_world: ``(T,22,3)`` world joints, **Y-up** (SMPL-X forward output;
            e.g. ``smplx_dict_to_joints22``). Internally rotated to the z-up raw
            frame before canonicalisation.
        body_pose_aa: ``(T,21,3)`` SMPL ``body_pose`` axis-angle (non-root joints).
        max_len: optional crop applied to both inputs before encoding.

    Returns:
        ``(motion (T-1,262), root_quat_init (1,4), root_pos_init_xz (1,3))``.
    """
    joints_world = np.asarray(joints_world, np.float32)
    body_pose_aa = np.asarray(body_pose_aa, np.float32).reshape(len(joints_world), 21, 3)
    if max_len is not None:
        joints_world = joints_world[:max_len]
        body_pose_aa = body_pose_aa[:max_len]
    pos_zup = np.einsum("ij,tnj->tni", _MT, joints_world)  # y-up -> z-up raw
    rot6d = body_pose_to_rot6d_row(body_pose_aa)            # (T,21,6) ROW
    return _process_motion(pos_zup, rot6d)


def build_pair(
    joints1: np.ndarray,
    body_pose1: np.ndarray,
    joints2: np.ndarray,
    body_pose2: np.ndarray,
    *,
    max_len: Optional[int] = 300,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Encode a two-person clip; align person2 to person1 (InterGen protocol).

    Returns ``(m1 (L,262), m2 (L,262), L)`` with ``L = min(T1,T2,max_len) - 1``.
    """
    m1, rq1, rp1 = encode_smpl_to_interhuman262(joints1, body_pose1, max_len=max_len)
    m2, rq2, rp2 = encode_smpl_to_interhuman262(joints2, body_pose2, max_len=max_len)
    L = min(len(m1), len(m2))
    m1, m2 = m1[:L], m2[:L]
    r_rel = _qmul(rq2, _qinv(rq1))
    angle = np.arctan2(r_rel[:, 2:3], r_rel[:, 0:1])
    xz = _qrot(rq1, rp2 - rp1)[:, [0, 2]]
    relative = np.concatenate([angle, xz], axis=-1)[0]
    m2 = rigid_transform(relative, m2)
    return m1.astype(np.float32), m2.astype(np.float32), L


# --------------------------------------------------------------------------- #
# Decode
# --------------------------------------------------------------------------- #
def interhuman262_to_joints(m262: np.ndarray) -> np.ndarray:
    """Recover canonical joint positions ``(...,22,3)`` from the position block.

    The 262 vector stores canonical world joints directly in ``[0:66]``, so this
    is an exact decode (no FK needed). For SMPL local rotations use
    :func:`rot6d_row_to_matrix` on ``[132:258]``.
    """
    m = np.asarray(m262)
    return m[..., :66].reshape(m.shape[:-1] + (22, 3))


def interhuman262_to_local_rotmat(m262: np.ndarray) -> np.ndarray:
    """Recover the 21 non-root local rotation matrices ``(...,21,3,3)``."""
    m = np.asarray(m262)
    return rot6d_row_to_matrix(m[..., 132:258].reshape(m.shape[:-1] + (21, 6)))


__all__ = [
    "TRANS_MATRIX",
    "FACE_JOINT_IDX",
    "body_pose_to_rot6d_row",
    "rot6d_row_to_matrix",
    "rigid_transform",
    "encode_smpl_to_interhuman262",
    "build_pair",
    "interhuman262_to_joints",
    "interhuman262_to_local_rotmat",
]
