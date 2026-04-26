"""Differentiable Forward Kinematics for SMPL-22 skeleton.

Computes world-space joint positions and rotations from local rot6d + translation,
using the SMPL-22 kinematic tree. Fully differentiable for use in IK optimization.

Rotation convention: **row-major** rot6d (matching training data and geometry.py).

Usage::

    from hftrainer.pipelines.motion.differentiable_fk import differentiable_fk, motion135_to_fk

    world_pos, world_rot = differentiable_fk(local_rotmat, translation, bone_offsets)
    world_pos, world_rot, transl, local_rotmat = motion135_to_fk(motion_denorm, bone_offsets)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from hftrainer.datasets.motion.motionhub.transforms.fk_utils import SMPL22_PARENTS

NUM_JOINTS = 22


def differentiable_fk(
    local_rotmat: Tensor,
    translation: Tensor,
    bone_offsets: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Differentiable forward kinematics for SMPL-22 skeleton.

    Args:
        local_rotmat: Local rotation matrices, shape ``(*, 22, 3, 3)``.
        translation: Root translation, shape ``(*, 3)``.
        bone_offsets: Bone offsets (relative to parent), shape ``(22, 3)``.
            ``offsets[0]`` is the root offset (T-pose root position, usually near origin).
            ``offsets[j]`` for ``j > 0`` is ``J_template[j] - J_template[parent[j]]``.

    Returns:
        world_positions: World-space joint positions, shape ``(*, 22, 3)``.
        world_rotations: World-space rotation matrices, shape ``(*, 22, 3, 3)``.
    """
    leading_shape = local_rotmat.shape[:-3]  # e.g. (B, T)
    device = local_rotmat.device
    dtype = local_rotmat.dtype

    # Use lists to avoid in-place ops (needed for autograd)
    world_rot_list: list = [None] * NUM_JOINTS
    world_pos_list: list = [None] * NUM_JOINTS

    for j in range(NUM_JOINTS):
        parent = SMPL22_PARENTS[j]
        if parent < 0:
            world_rot_list[j] = local_rotmat[..., j, :, :]
            world_pos_list[j] = translation + bone_offsets[j]
        else:
            world_rot_list[j] = world_rot_list[parent] @ local_rotmat[..., j, :, :]
            offset_rotated = (world_rot_list[parent] @ bone_offsets[j].unsqueeze(-1)).squeeze(-1)
            world_pos_list[j] = world_pos_list[parent] + offset_rotated

    world_pos = torch.stack(world_pos_list, dim=-2)  # (*, 22, 3)
    world_rot = torch.stack(world_rot_list, dim=-3)  # (*, 22, 3, 3)

    return world_pos, world_rot


def rot6d_to_rotmat_row_major(rot6d: Tensor) -> Tensor:
    """Convert row-major rot6d to rotation matrix using geometry.py convention.

    Args:
        rot6d: Row-major 6D rotation, shape ``(*, 6)``.

    Returns:
        Rotation matrix, shape ``(*, 3, 3)``.
    """
    from hftrainer.models.motion.hymotion_m2m.network.geometry import rot6d_to_rotation_matrix
    return rot6d_to_rotation_matrix(rot6d)


def rotmat_to_rot6d_row_major(rotmat: Tensor) -> Tensor:
    """Convert rotation matrix to row-major rot6d.

    Args:
        rotmat: Rotation matrix, shape ``(*, 3, 3)``.

    Returns:
        Row-major 6D rotation, shape ``(*, 6)``.
    """
    from hftrainer.models.motion.hymotion_m2m.network.geometry import rotation_matrix_to_rot6d
    return rotation_matrix_to_rot6d(rotmat)


def motion135_to_fk(
    motion_denorm: Tensor,
    bone_offsets: Tensor,
    rotation_space: str = 'local',
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Parse 135-dim denormalized motion and run FK.

    Args:
        motion_denorm: Denormalized motion tensor, shape ``(*, 135)``.
        bone_offsets: Bone offsets, shape ``(22, 3)``.
        rotation_space: ``'local'`` or ``'global'``. If ``'global'``,
            converts to local first via inverse FK before running FK.

    Returns:
        world_positions: ``(*, 22, 3)`` world-space joint positions.
        world_rotations: ``(*, 22, 3, 3)`` world-space rotation matrices.
        translation: ``(*, 3)`` root translation.
        local_rotmat: ``(*, 22, 3, 3)`` local rotation matrices.
    """
    leading = motion_denorm.shape[:-1]

    # Parse 135-dim: [trans(3), rot6d(22*6=132)]
    translation = motion_denorm[..., :3]  # (*, 3)
    rot6d_flat = motion_denorm[..., 3:135]  # (*, 132)
    rot6d = rot6d_flat.reshape(*leading, 22, 6)  # (*, 22, 6)

    if rotation_space == 'global':
        # Convert global rot6d to local rot6d first
        from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
            global_to_local_rot6d_torch,
        )
        rot6d = global_to_local_rot6d_torch(rot6d)

    # Convert row-major rot6d to rotation matrix
    local_rotmat = rot6d_to_rotmat_row_major(rot6d)  # (*, 22, 3, 3)

    # Run FK
    world_pos, world_rot = differentiable_fk(local_rotmat, translation, bone_offsets)

    return world_pos, world_rot, translation, local_rotmat


def fk_to_motion135(
    local_rotmat: Tensor,
    translation: Tensor,
    rotation_space: str = 'local',
) -> Tensor:
    """Convert local rotation matrices and translation back to 135-dim motion.

    Args:
        local_rotmat: Local rotation matrices, shape ``(*, 22, 3, 3)``.
        translation: Root translation, shape ``(*, 3)``.
        rotation_space: ``'local'`` or ``'global'``. If ``'global'``,
            converts local rotations to global before encoding to rot6d.

    Returns:
        motion: 135-dim motion tensor, shape ``(*, 135)``.
    """
    leading = local_rotmat.shape[:-3]

    rot6d = rotmat_to_rot6d_row_major(local_rotmat)  # (*, 22, 6)

    if rotation_space == 'global':
        from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
            local_to_global_rot6d_torch,
        )
        rot6d = local_to_global_rot6d_torch(rot6d)

    rot6d_flat = rot6d.reshape(*leading, 132)
    motion = torch.cat([translation, rot6d_flat], dim=-1)  # (*, 135)
    return motion
