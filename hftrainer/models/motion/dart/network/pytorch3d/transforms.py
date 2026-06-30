"""Subset of :mod:`pytorch3d.transforms` used by DART inference.

The A100 runtime used for large evaluations does not always have PyTorch3D
installed. DART only needs rotation conversion helpers for inference, so this
shim mirrors the PyTorch3D 6D convention locally inside the vendored DART
runtime.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hftrainer.motion.representation.rotation import (
    axis_angle_to_matrix,
    euler_to_matrix,
    matrix_to_axis_angle,
)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """PyTorch3D 6D -> matrix convention.

    PyTorch3D stores the first two rows of the rotation matrix:
    ``matrix[..., :2, :].reshape(..., 6)``.
    """

    if d6.shape[-1] != 6:
        raise ValueError(f"rotation_6d_to_matrix expects (..., 6), got {tuple(d6.shape)}")
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Matrix -> PyTorch3D 6D convention."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix_to_rotation_6d expects (..., 3, 3), got {tuple(matrix.shape)}")
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    return euler_to_matrix(euler_angles, order=convention, deg=False)


__all__ = [
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "rotation_6d_to_matrix",
    "matrix_to_rotation_6d",
    "euler_angles_to_matrix",
]
