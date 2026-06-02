"""SMPL -> MotionStreamer ``humanml3d_272`` ENCODER (forward representation).

Inverse of :mod:`hftrainer.datasets.motion.representation.humanml_repr` decode
functions; a faithful re-implementation of the official MotionStreamer forward
script ``representation_272.py``
(https://github.com/Li-xingXiao/272-dim-Motion-Representation).

272 layout (per frame, ``njoint=22``)::

    [0:2]      root local xz velocity, heading-removed
    [2:8]      heading angular velocity, 6D rotation (frame 0 = identity)
    [8:74]     22 joint positions, heading-removed, per-frame xz origin (66)
    [74:140]   22 joint velocities, heading-removed (66)
    [140:272]  22 joint LOCAL rotations, 6D ROW-major (first two rows) (132)

The forward needs two physical inputs:
  * ``joints_world``  ``(T, 22, 3)`` world joint positions (Y-up, metres)
  * ``local_rotmat``  ``(T, 22, 3, 3)`` SMPL local rotation matrices
    (joint 0 = global root orientation; joints 1..21 parent-relative)

For the M2M model output (135-dim), :func:`motion135_to_272` runs the SMPL-H
forward kinematics (same skeleton / convention used to build the GT 263 set)
to obtain both inputs, then encodes.

Rotation block is ROW-major (``matrix[:2, :]``), matching
``humanml_repr._rotation_6d_to_matrix`` and the official decoder.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_NJOINT = 22


def _rot_yaw(yaw: np.ndarray) -> np.ndarray:
    """Pure yaw (around +Y) rotation matrices ``(...,3,3)`` for angles ``(...,)``.

    Matches ``representation_272.rot_yaw``::

        [[ cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
    """
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    z = np.zeros_like(yaw)
    o = np.ones_like(yaw)
    return np.stack([
        np.stack([cs, z, sn], axis=-1),
        np.stack([z, o, z], axis=-1),
        np.stack([-sn, z, cs], axis=-1),
    ], axis=-2)


def _matrix_to_rotation_6d_rows(mat: np.ndarray) -> np.ndarray:
    """ROW-major 6D = first two ROWS of the matrix, flattened.

    ``representation_272`` stores ``rotations_matrix[..., :2, :]`` (rows 0,1),
    read back by the ROW-major ``_rotation_6d_to_matrix`` decoder.
    """
    return mat[..., :2, :].reshape(*mat.shape[:-2], 6)


def encode_smpl_to_272(joints_world: np.ndarray,
                       local_rotmat: np.ndarray) -> np.ndarray:
    """Encode one clip to the 272 representation.

    Faithful re-implementation of the official ``representation_272.py`` main
    loop (SMPL-FK joint extraction assumed done by the caller).

    Args:
        joints_world: ``(T, 22, 3)`` world joint positions (Y-up, metres).
        local_rotmat: ``(T, 22, 3, 3)`` SMPL local rotation matrices.

    Returns:
        ``(T, 272)`` representation (float64).
    """
    pos = np.array(joints_world, dtype=np.float64)
    rot = np.array(local_rotmat, dtype=np.float64)
    nfrm, njoint = pos.shape[0], pos.shape[1]
    assert njoint == _NJOINT, f"expected 22 joints, got {njoint}"
    root_idx = 0

    # put on floor + root xz origin for the FIRST frame
    ori = pos[0, root_idx].copy()
    ori[1] = np.min(pos[:, :, 1])
    pos = pos - ori

    velocities_root = pos[1:, root_idx, :] - pos[:-1, root_idx, :]

    # per-frame xz origin (all joints relative to that frame's root)
    pos[:, :, 0] -= pos[:, root_idx:root_idx + 1, 0]
    pos[:, :, 2] -= pos[:, root_idx:root_idx + 1, 2]

    # heading from root rotation matrix
    R0 = rot[:, root_idx]
    global_heading = -np.arctan2(R0[:, 0, 2], R0[:, 2, 2])
    global_heading_rot = _rot_yaw(global_heading)               # (T,3,3)
    global_heading_diff = global_heading[1:] - global_heading[:-1]
    global_heading_diff_rot = _rot_yaw(global_heading_diff)     # (T-1,3,3)

    positions_no_heading = np.matmul(
        np.repeat(global_heading_rot[:, None, :, :], njoint, axis=1),
        pos[..., None]).squeeze(-1)                             # (T,22,3)
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]

    velocities_root_xy_no_heading = np.matmul(
        global_heading_rot[:-1], velocities_root[:, :, None]
    ).squeeze(-1)[..., [0, 2]]                                  # (T-1,2)

    rot = rot.copy()
    rot[:, root_idx] = np.matmul(global_heading_rot, rot[:, root_idx])

    size_frame = 8 + njoint * 3 + njoint * 3 + njoint * 6
    final_x = np.zeros((nfrm, size_frame), dtype=np.float64)
    final_x[0, 2] = 1.0   # frame-0 heading 6D = identity rows [1,0,0,0,1,0]
    final_x[0, 6] = 1.0
    final_x[1:, 2:8] = _matrix_to_rotation_6d_rows(global_heading_diff_rot)
    final_x[1:, :2] = velocities_root_xy_no_heading
    final_x[:, 8:8 + 3 * njoint] = positions_no_heading.reshape(nfrm, -1)
    final_x[1:, 8 + 3 * njoint:8 + 6 * njoint] = velocities_no_heading.reshape(nfrm - 1, -1)
    final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = _matrix_to_rotation_6d_rows(rot).reshape(nfrm, -1)
    return final_x


# ---------------------------------------------------------------------------
# Round-trip helpers (Gate A validation)
# ---------------------------------------------------------------------------

def reencode_272_via_stored_positions(m272: np.ndarray) -> np.ndarray:
    """GT272 -> decode (stored positions + local rotations) -> re-encode.

    Isolates the *encoding math* from any FK / body-model mismatch: uses the
    EXACT stored joint positions, so a correct encoder reproduces ``m272`` to
    float precision.
    """
    from hftrainer.datasets.motion.representation.humanml_repr import (
        recover_local_rotations_and_root, recover_272_stored_positions,
    )
    rot, _root = recover_local_rotations_and_root(m272)
    joints = recover_272_stored_positions(m272)
    return encode_smpl_to_272(joints, rot)


def reencode_272_via_fk(m272: np.ndarray,
                        smplh_model: Optional[str] = None) -> np.ndarray:
    """GT272 -> decode (rot, root) -> SMPL-H FK joints -> re-encode.

    Mirrors the *prediction* pipeline (FK-derived joints); the residual vs
    ``m272`` quantifies the SMPL-H-rest-FK vs native-SMPL-X-betas domain gap.
    """
    from hftrainer.datasets.motion.representation.humanml_repr import (
        recover_local_rotations_and_root, fk_smplh_joints, DEFAULT_PATHS,
    )
    rot, root = recover_local_rotations_and_root(m272)
    model_path = smplh_model or DEFAULT_PATHS.resolve("smplh_model")
    joints = fk_smplh_joints(rot, root, model_path)
    return encode_smpl_to_272(joints, rot)


# ---------------------------------------------------------------------------
# Prediction path: M2M 135-dim model output -> 272
# ---------------------------------------------------------------------------

import os as _os

# Canonical parent-relative bone offsets (22,3) of the **GT humanml3d_272 body**,
# extracted directly from the GT 272 set (offset[j] = Rg[parent]^T @ (pos[j]-pos[p]),
# averaged over ~400 clips / 87k frames; see scripts/eval/_extract_canon_skeleton).
#
# WHY THIS MATTERS (bug fixed 2026-06-02): the GT 272 set is built from **SMPL-X**
# joints, whose collar/neck/head/shoulder rest positions differ a LOT from the
# **SMPL-H** rest skeleton we previously FK'd with (collar 0.10 vs 0.15 m, head
# 0.17 vs 0.08 m, ...). FK'ing M2M/HY rotations on the SMPL-H rest skeleton put
# the upper body ~210 mm off the GT joints (errors accumulate down the chain),
# inflating the 272 FID by ~1.4 (HY-Lite 11.52 -> 10.15). Using this GT-matched
# canonical skeleton drops the stored-vs-FK joint error to ~23 mm (the residual
# is SMPL-X pose-blendshape deformation, which a rigid skeleton cannot reproduce).
_CANON_OFFSETS_PATH = _os.path.join(_os.path.dirname(__file__),
                                    "assets", "bone_offsets_canon272.npy")


def _canonical_272_offsets() -> np.ndarray:
    return np.load(_CANON_OFFSETS_PATH).astype(np.float64)


def motion135_to_272(motion_135: np.ndarray, *,
                     rotation_space: str = "local",
                     bone_offsets: Optional[np.ndarray] = None,
                     skeleton: str = "canon272",
                     smplh_model: Optional[str] = None) -> np.ndarray:
    """Convert a 135-dim M2M model output (trans3 + 22x6D rot6d, 30 fps) to 272.

    Runs SMPL-22 forward kinematics (``differentiable_fk``) then encodes via
    :func:`encode_smpl_to_272`.

    Args:
        motion_135: ``(T, >=135)`` motion (only first 135 dims used).
        rotation_space: ``"local"`` or ``"global"`` (model rot6d convention).
        bone_offsets: optional ``(22,3)`` parent-relative offsets override. If
            given, takes precedence over ``skeleton``.
        skeleton: which rest skeleton to FK with when ``bone_offsets`` is None:
            ``"canon272"`` (DEFAULT) = the GT-humanml3d_272 SMPL-X canonical body
            (matches the evaluator's body, ~23 mm joint error); ``"smplh"`` =
            legacy SMPL-H neutral rest (~210 mm off the GT body, do NOT use for
            the 272 evaluator -- inflates FID by ~1.4).

    Returns:
        ``(T, 272)`` representation (float64).
    """
    import torch
    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
    from hftrainer.datasets.motion.representation.humanml_repr import (
        _smplh_bone_offsets, DEFAULT_PATHS,
    )

    arr = np.asarray(motion_135, dtype=np.float32)
    m135 = torch.from_numpy(arr[:, :135]).float()
    if bone_offsets is not None:
        bo = torch.as_tensor(bone_offsets).float()
    elif skeleton == "canon272":
        bo = torch.from_numpy(_canonical_272_offsets()).float()
    elif skeleton == "smplh":
        mp = smplh_model or DEFAULT_PATHS.resolve("smplh_model")
        bo = torch.from_numpy(_smplh_bone_offsets(mp)).float()
    else:
        raise ValueError(f"unknown skeleton={skeleton!r}")

    world_pos, _world_rot, _trans, local_rotmat = motion135_to_fk(
        m135, bo, rotation_space=rotation_space)
    joints = world_pos.detach().cpu().numpy().astype(np.float64)
    rot = local_rotmat.detach().cpu().numpy().astype(np.float64)
    return encode_smpl_to_272(joints, rot)
