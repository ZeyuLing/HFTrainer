"""Foot contact detection for 151-dim motion representation.

Extends 147-dim representation with 4-dim foot contact channel:
    - Binary indicators for 4 end-effector joints
    - Computed via velocity-based contact detection
    - Layout: [L_Foot_contact, R_Foot_contact, L_Wrist_contact, R_Wrist_contact]

151-dim layout:
    dims [0:147]    — original 147-dim motion (trans + rot6d + end-effector positions)
    dims [147:151]  — foot contact binary indicators (4)

Contact detection algorithm (from Momask):
    A joint is "in contact" if its velocity < threshold (default 0.002)
    Computed as: sqrt(v_x^2 + v_y^2 + v_z^2) < threshold
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def detect_foot_contact(
    positions: Tensor,
    velocity_threshold: float = 0.002,
    joint_indices: Optional[list] = None,
) -> Tensor:
    """Detect foot contact based on joint velocity.

    Args:
        positions: Joint positions, shape (*, T, J, 3) or (*, J, 3).
                   Assumed to be world-space positions.
        velocity_threshold: Velocity threshold for contact detection (default 0.002).
        joint_indices: List of joint indices to detect contact for.
                       If None, uses [10, 11, 20, 21] (L_Foot, R_Foot, L_Wrist, R_Wrist).

    Returns:
        Contact indicators, shape (*, T-1, len(joint_indices)) or (*, len(joint_indices)).
        Binary tensor (0 or 1) where 1 = in contact, 0 = not in contact.
    """
    if joint_indices is None:
        joint_indices = [10, 11, 20, 21]  # L_Foot, R_Foot, L_Wrist, R_Wrist

    device = positions.device
    dtype = positions.dtype

    # Handle different input shapes
    if positions.dim() == 3:  # (T, J, 3) -> add batch dimension
        positions = positions.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    B, T, J, _ = positions.shape
    
    # Extract velocities between consecutive frames
    # velocity[t] = positions[t+1] - positions[t]
    positions_selected = positions[..., joint_indices, :]  # (B, T, K, 3) where K = len(joint_indices)
    velocity = positions_selected[:, 1:, :, :] - positions_selected[:, :-1, :, :]  # (B, T-1, K, 3)

    # Compute velocity magnitude
    velocity_mag = torch.norm(velocity, dim=-1)  # (B, T-1, K)

    # Detect contact: velocity_mag < threshold
    contact = (velocity_mag < velocity_threshold).float()  # (B, T-1, K)

    if squeeze_batch:
        contact = contact.squeeze(0)

    return contact


def compute_151dim_motion_from_147(
    motion_147: Tensor,
    mean_147: Optional[Tensor] = None,
    std_147: Optional[Tensor] = None,
    bone_offsets: Optional[Tensor] = None,
    rotation_space: str = 'local',
    velocity_threshold: float = 0.002,
    data_mask_temporal: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute 151-dim motion by adding foot contact to 147-dim.

    Args:
        motion_147: 147-dim motion (B, L, 147) in normalized or denormalized space.
        mean_147: (147,) mean for denormalization. If provided, assumes motion_147 is normalized.
        std_147: (147,) std for denormalization. If provided, assumes motion_147 is normalized.
        bone_offsets: (22, 3) bone offsets for FK computation.
        rotation_space: 'local' or 'global' for FK computation.
        velocity_threshold: Velocity threshold for contact detection.
        data_mask_temporal: (B, L) temporal mask where 1=valid, 0=padded.

    Returns:
        motion_151: (B, L, 151) motion with foot contact appended.
        contact_151: (B, L, 4) foot contact indicators (binary).
    """
    from hftrainer.motion.pipeline_utils.differentiable_fk import motion135_to_fk

    B, L, D = motion_147.shape
    assert D == 147, f"Expected 147-dim motion, got {D}"

    device = motion_147.device
    dtype = motion_147.dtype

    # Denormalize if needed
    if mean_147 is not None and std_147 is not None:
        std_safe = torch.where(std_147 < 1e-3, torch.ones_like(std_147), std_147)
        motion_147_denorm = motion_147 * std_safe + mean_147
    else:
        motion_147_denorm = motion_147

    # Extract components
    trans = motion_147_denorm[..., 0:3]        # (B, L, 3)
    rot6d = motion_147_denorm[..., 3:135]     # (B, L, 132)
    pos_pred = motion_147_denorm[..., 135:147]  # (B, L, 12)

    # Construct 135-dim for FK
    motion_135 = torch.cat([trans, rot6d], dim=-1)  # (B, L, 135)

    # Compute FK to get world-space positions
    if bone_offsets is not None:
        world_pos, _, _, _ = motion135_to_fk(
            motion_135.reshape(B * L, 135),
            bone_offsets,
            rotation_space=rotation_space,
        )
        world_pos = world_pos.reshape(B, L, 22, 3)
    else:
        # Fallback: use predicted positions directly (less accurate)
        ee_indices = [20, 21, 10, 11]
        world_pos_ee = pos_pred.reshape(B, L, 4, 3)
        # Pad to 22 joints (approximate)
        world_pos = torch.zeros(B, L, 22, 3, device=device, dtype=dtype)
        world_pos[:, :, ee_indices, :] = world_pos_ee

    # Detect foot contact
    contact = detect_foot_contact(
        world_pos,
        velocity_threshold=velocity_threshold,
        joint_indices=[10, 11, 20, 21],  # L_Foot, R_Foot, L_Wrist, R_Wrist
    )  # (B, L-1, 4)

    # Pad contact to match length L (assume first frame stays same as second)
    contact_padded = torch.cat(
        [contact[:, :1, :], contact],
        dim=1
    )  # (B, L, 4)

    # Mask out padded frames if provided
    if data_mask_temporal is not None:
        mask = data_mask_temporal.unsqueeze(-1).float()  # (B, L, 1)
        contact_padded = contact_padded * mask

    # Combine 147-dim + 4-dim contact -> 151-dim
    motion_151 = torch.cat([motion_147, contact_padded], dim=-1)  # (B, L, 151)

    return motion_151, contact_padded


__all__ = [
    'detect_foot_contact',
    'compute_151dim_motion_from_147',
]
