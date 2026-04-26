from __future__ import annotations

from typing import Union

import numpy as np
import torch

from ._geometry_compat import axis_angle_to_matrix


def root_rotation_matrices_from_poses(
    poses_3d: np.ndarray,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    root_aa = torch.as_tensor(poses_3d[:, 0, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        root_rot = axis_angle_to_matrix(root_aa)
    return root_rot.detach().cpu().numpy()


def apply_inverse_root_rotation(vectors: np.ndarray, root_rot_mats: np.ndarray) -> np.ndarray:
    if vectors.ndim == 2:
        root_rot_inv = np.swapaxes(root_rot_mats, 1, 2)
        return np.einsum("fij,fj->fi", root_rot_inv, vectors)
    if vectors.ndim == 3:
        root_rot_inv = np.swapaxes(root_rot_mats, 1, 2)
        return np.einsum("fij,fkj->fki", root_rot_inv, vectors)
    raise ValueError(f"Unsupported vector rank for root rotation application: {vectors.shape}")


def root_stabilize_positions(joints: np.ndarray, root_rot_mats: np.ndarray) -> np.ndarray:
    root = joints[:, 0:1, :]
    root_relative = joints - root
    return apply_inverse_root_rotation(root_relative, root_rot_mats)


def root_angular_velocity_deg_per_frame(root_rot_mats: np.ndarray) -> np.ndarray:
    if root_rot_mats.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    rel_rot = np.einsum(
        "fij,fjk->fik",
        np.swapaxes(root_rot_mats[:-1], 1, 2),
        root_rot_mats[1:],
    )
    trace = np.trace(rel_rot, axis1=1, axis2=2)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta)).astype(np.float64)
