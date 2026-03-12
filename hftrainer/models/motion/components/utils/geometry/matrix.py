"""Minimal matrix helpers required by vendored motion components."""

from __future__ import annotations

from typing import List

import torch


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)


def get_TRS(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """
    Build homogeneous transforms from rotation (..., 3, 3) and translation (..., 3).
    """
    rot_shape = rotation.shape[:-2]
    out = torch.zeros(*rot_shape, 4, 4, device=rotation.device, dtype=rotation.dtype)
    out[..., :3, :3] = rotation
    out[..., :3, 3] = translation
    out[..., 3, 3] = 1.0
    return out


def forward_kinematics(local_mat: torch.Tensor, parents: List[int]) -> torch.Tensor:
    """
    Compose local transforms along a kinematic chain.
    """
    world = torch.zeros_like(local_mat)
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx < 0:
            world[..., joint_idx, :, :] = local_mat[..., joint_idx, :, :]
        else:
            world[..., joint_idx, :, :] = (
                world[..., parent_idx, :, :] @ local_mat[..., joint_idx, :, :]
            )
    return world


def get_position(mat: torch.Tensor) -> torch.Tensor:
    return mat[..., :3, 3]


def get_rotation(mat: torch.Tensor) -> torch.Tensor:
    return mat[..., :3, :3]


def get_mat_BtoA(mat_b: torch.Tensor, mat_a: torch.Tensor) -> torch.Tensor:
    """
    Compute the relative transform that maps coordinates from B to A.
    """
    return torch.linalg.solve(mat_b, mat_a)
