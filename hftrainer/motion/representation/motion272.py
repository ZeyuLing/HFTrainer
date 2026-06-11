"""MotionStreamer / GoToZero ``humanml3d_272`` representation (encode/decode).

Faithful re-implementation of the official MotionStreamer forward script
``representation_272.py`` (https://github.com/Li-xingXiao/272-dim-Motion-Representation).
See :class:`hftrainer.motion.representation.specs.MS272` for the channel layout.

272 layout (per frame, ``njoint=22``)::

    [0:2]      root local xz velocity, heading-removed
    [2:8]      heading angular velocity, 6D rotation (frame 0 = identity)
    [8:74]     22 joint positions, heading-removed, per-frame xz origin (66)
    [74:140]   22 joint velocities, heading-removed (66)
    [140:272]  22 joint LOCAL rotations, 6D ROW-major (first two rows) (132)

The rotation block is ROW-major (``matrix[:2, :]``), matching the official
decoder and :func:`hftrainer.motion.representation.rotation` ``convention="row"``.

Important: :func:`motion135_to_272` FK's the SMPL-22 rotations on the **GT-272
canonical skeleton** (``bone_offsets_canon272.npy``), NOT the SMPL-H rest pose.
Using the SMPL-H rest skeleton puts the upper body ~210 mm off the GT joints and
inflates the 272 evaluator FID by ~1.4 (bug fixed 2026-06-02).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

_NJOINT = 22

# Canonical parent-relative bone offsets (22,3) of the GT humanml3d_272 body.
# Resolution: in-repo library asset first, then the legacy scripts location.
_ASSET_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "bone_offsets_canon272.npy"),
    os.path.join("scripts", "eval", "assets", "bone_offsets_canon272.npy"),
)


def _canonical_272_offsets() -> np.ndarray:
    for p in _ASSET_CANDIDATES:
        if os.path.isfile(p):
            return np.load(p).astype(np.float64)
    raise FileNotFoundError(
        "bone_offsets_canon272.npy not found. Tried: " + ", ".join(_ASSET_CANDIDATES)
    )


def _rot_yaw(yaw: np.ndarray) -> np.ndarray:
    """Pure yaw (+Y) rotation matrices ``(...,3,3)`` (matches representation_272.rot_yaw)."""
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    z = np.zeros_like(yaw)
    o = np.ones_like(yaw)
    return np.stack(
        [
            np.stack([cs, z, sn], axis=-1),
            np.stack([z, o, z], axis=-1),
            np.stack([-sn, z, cs], axis=-1),
        ],
        axis=-2,
    )


def _matrix_to_rotation_6d_rows(mat: np.ndarray) -> np.ndarray:
    """ROW-major 6D = first two ROWS of the matrix, flattened."""
    return mat[..., :2, :].reshape(*mat.shape[:-2], 6)


def encode_smpl_to_272(joints_world: np.ndarray, local_rotmat: np.ndarray) -> np.ndarray:
    """Encode one clip to the 272 representation.

    Args:
        joints_world: ``(T, 22, 3)`` world joint positions (Y-up, metres).
        local_rotmat: ``(T, 22, 3, 3)`` SMPL local rotation matrices
            (joint 0 = global root orientation; joints 1..21 parent-relative).

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
    pos[:, :, 0] -= pos[:, root_idx : root_idx + 1, 0]
    pos[:, :, 2] -= pos[:, root_idx : root_idx + 1, 2]

    # heading from root rotation matrix
    R0 = rot[:, root_idx]
    global_heading = -np.arctan2(R0[:, 0, 2], R0[:, 2, 2])
    global_heading_rot = _rot_yaw(global_heading)  # (T,3,3)
    global_heading_diff = global_heading[1:] - global_heading[:-1]
    global_heading_diff_rot = _rot_yaw(global_heading_diff)  # (T-1,3,3)

    positions_no_heading = np.matmul(
        np.repeat(global_heading_rot[:, None, :, :], njoint, axis=1), pos[..., None]
    ).squeeze(-1)  # (T,22,3)
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]

    velocities_root_xy_no_heading = np.matmul(
        global_heading_rot[:-1], velocities_root[:, :, None]
    ).squeeze(-1)[..., [0, 2]]  # (T-1,2)

    rot = rot.copy()
    rot[:, root_idx] = np.matmul(global_heading_rot, rot[:, root_idx])

    size_frame = 8 + njoint * 3 + njoint * 3 + njoint * 6
    final_x = np.zeros((nfrm, size_frame), dtype=np.float64)
    final_x[0, 2] = 1.0  # frame-0 heading 6D = identity rows [1,0,0,0,1,0]
    final_x[0, 6] = 1.0
    final_x[1:, 2:8] = _matrix_to_rotation_6d_rows(global_heading_diff_rot)
    final_x[1:, :2] = velocities_root_xy_no_heading
    final_x[:, 8 : 8 + 3 * njoint] = positions_no_heading.reshape(nfrm, -1)
    final_x[1:, 8 + 3 * njoint : 8 + 6 * njoint] = velocities_no_heading.reshape(nfrm - 1, -1)
    final_x[:, 8 + 6 * njoint : 8 + 12 * njoint] = _matrix_to_rotation_6d_rows(rot).reshape(nfrm, -1)
    return final_x


def motion135_to_272(
    motion_135: np.ndarray,
    *,
    rotation_space: str = "local",
    bone_offsets: Optional[np.ndarray] = None,
    skeleton: str = "canon272",
    smplh_model: Optional[str] = None,
) -> np.ndarray:
    """Convert a 135-dim motion (trans3 + 22x6D rot6d, 30 fps) to MS272.

    Runs SMPL-22 forward kinematics then encodes via :func:`encode_smpl_to_272`.

    Args:
        motion_135: ``(T, >=135)`` motion (only the first 135 dims are used).
        rotation_space: ``"local"`` or ``"global"`` (model rot6d convention).
        bone_offsets: optional ``(22,3)`` rest offsets override (takes precedence).
        skeleton: rest skeleton when ``bone_offsets`` is None:
            ``"canon272"`` (DEFAULT, GT-272 SMPL-X canonical body, ~23 mm joint
            error) or ``"smplh"`` (legacy SMPL-H neutral, ~210 mm off, do NOT use
            for the 272 evaluator).
        smplh_model: SMPL-H model path used only when ``skeleton="smplh"``.

    Returns:
        ``(T, 272)`` representation (float64).
    """
    import torch

    from hftrainer.motion.skeleton.fk import motion135_to_fk

    arr = np.asarray(motion_135, dtype=np.float32)
    m135 = torch.from_numpy(arr[:, :135]).float()
    if bone_offsets is not None:
        bo = torch.as_tensor(bone_offsets).float()
    elif skeleton == "canon272":
        bo = torch.from_numpy(_canonical_272_offsets()).float()
    elif skeleton == "smplh":
        from hftrainer.motion.representation.humanml import DEFAULT_PATHS
        from hftrainer.datasets.motion.representation.humanml_repr import _smplh_bone_offsets

        mp = smplh_model or DEFAULT_PATHS.resolve("smplh_model")
        bo = torch.from_numpy(_smplh_bone_offsets(mp)).float()
    else:
        raise ValueError(f"unknown skeleton={skeleton!r}")

    world_pos, _world_rot, _trans, local_rotmat = motion135_to_fk(
        m135, bo, rotation_space=rotation_space
    )
    joints = world_pos.detach().cpu().numpy().astype(np.float64)
    rot = local_rotmat.detach().cpu().numpy().astype(np.float64)
    return encode_smpl_to_272(joints, rot)


# --------------------------------------------------------------------------- #
# Round-trip / decode helpers (delegate decode to humanml bridges)
# --------------------------------------------------------------------------- #
def reencode_272_via_stored_positions(m272: np.ndarray) -> np.ndarray:
    """GT272 -> decode (stored positions + local rotations) -> re-encode.

    Isolates the encoding math from any FK/body-model mismatch.
    """
    from hftrainer.motion.representation.humanml import (
        recover_local_rotations_and_root,
        recover_272_stored_positions,
    )

    rot, _root = recover_local_rotations_and_root(m272)
    joints = recover_272_stored_positions(m272)
    return encode_smpl_to_272(joints, rot)


def reencode_272_via_fk(m272: np.ndarray, smplh_model: Optional[str] = None) -> np.ndarray:
    """GT272 -> decode (rot, root) -> SMPL-H FK joints -> re-encode."""
    from hftrainer.motion.representation.humanml import (
        recover_local_rotations_and_root,
        fk_smplh_joints,
        DEFAULT_PATHS,
    )

    rot, root = recover_local_rotations_and_root(m272)
    model_path = smplh_model or DEFAULT_PATHS.resolve("smplh_model")
    joints = fk_smplh_joints(rot, root, model_path)
    return encode_smpl_to_272(joints, rot)


__all__ = [
    "encode_smpl_to_272",
    "motion135_to_272",
    "reencode_272_via_stored_positions",
    "reencode_272_via_fk",
]
