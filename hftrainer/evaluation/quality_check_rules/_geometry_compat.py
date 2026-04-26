"""Geometry / rotation function imports for quality checkers.

All functions come from hftrainer's rotation_convert module.
No fallback to hymotion — import errors surface immediately.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
)

# alias used by some checkers
angle_axis_to_rotation_matrix = axis_angle_to_matrix


def quaternion_fix_continuity(q: Tensor) -> Tensor:
    """Force quaternion continuity across the time dimension.

    Selects the representation (q or -q) with minimal distance
    (maximal dot product) between consecutive frames.

    Args:
        q: Quaternion tensor of shape ``(L, 4)`` or ``(L, J, 4)``,
           real-part-first convention ``(w, x, y, z)``.

    Returns:
        Continuous quaternion tensor with the same shape.
    """
    assert q.ndim in (2, 3), (
        f"Expected 2D (L, 4) or 3D (L, J, 4) tensor, got shape {q.shape}"
    )
    assert q.shape[-1] == 4, f"Last dim should be 4, got {q.shape[-1]}"
    if q.shape[0] <= 1:
        return q.clone()

    result = q.clone()
    dot_products = torch.sum(q[1:] * q[:-1], dim=-1)
    flip_mask = dot_products < 0
    flip_mask = (torch.cumsum(flip_mask.int(), dim=0) % 2).bool()
    result[1:][flip_mask] *= -1
    return result


__all__ = [
    "axis_angle_to_matrix",
    "angle_axis_to_rotation_matrix",
    "axis_angle_to_quaternion",
    "quaternion_fix_continuity",
]
