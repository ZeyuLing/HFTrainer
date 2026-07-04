"""FK consistency loss for 147-dim motion representation.

Compares FK-computed end-effector positions with the 12-dim end-effector
position channels in the 147-dim representation.

Layout of 147-dim:
    dims [0:3]      — translation (3)
    dims [3:135]    — rot6d (22*6 = 132)
    dims [135:138]  — L_Wrist position (xyz)
    dims [138:141]  — R_Wrist position (xyz)
    dims [141:144]  — L_Foot position (xyz)
    dims [144:147]  — R_Foot position (xyz)

FK consistency loss:
    L_fk = smooth_L1(FK(rot_pred) - pos_pred)
    where rot_pred is extracted from the rotation channels (dims 3:135)
    and pos_pred comes from the 12-dim end-effector position channels (dims 135:147)
"""

from typing import Optional

import torch
from torch import Tensor


def motion147_fk_loss(
    motion_147_norm: Tensor,
    mean: Tensor,
    std: Tensor,
    bone_offsets: Tensor,
    rotation_space: str = 'local',
    timesteps: Optional[Tensor] = None,
    data_mask_temporal: Optional[Tensor] = None,
) -> Tensor:
    """Compute FK consistency loss for 147-dim motion.

    Args:
        motion_147_norm: (B, L, 147) predicted motion in normalized space.
        mean: (147,) mean normalization vector.
        std: (147,) std normalization vector.
        bone_offsets: (22, 3) bone offsets tensor.
        rotation_space: 'local' or 'global', determines FK computation.
        timesteps: (B,) timestep values for warmup (unused currently, for future use).
        data_mask_temporal: (B, L) mask where 1=valid frame, 0=padded.

    Returns:
        Scalar FK consistency loss (smooth_L1).
    """
    from hftrainer.motion.pipeline_utils.differentiable_fk import motion135_to_fk

    B, L, D = motion_147_norm.shape
    assert D == 147, f"Expected 147-dim motion, got {D}"

    device = motion_147_norm.device
    dtype = motion_147_norm.dtype

    # Denormalize motion
    std_safe = torch.where(std < 1e-3, torch.ones_like(std), std)
    motion_147 = motion_147_norm * std_safe + mean  # (B, L, 147)

    # Extract components
    trans = motion_147[..., 0:3]                      # (B, L, 3)
    rot6d = motion_147[..., 3:135]                   # (B, L, 132)
    pos_pred = motion_147[..., 135:147]              # (B, L, 12)

    # Construct 135-dim motion for FK (trans + rot6d only)
    motion_135 = torch.cat([trans, rot6d], dim=-1)  # (B, L, 135)

    # Run FK on 135-dim (rotation only)
    # Returns world-space positions: (B, L, 22, 3)
    with torch.no_grad():
        world_pos, _, _, _ = motion135_to_fk(
            motion_135.reshape(B * L, 135),
            bone_offsets,
            rotation_space=rotation_space,
        )
        world_pos = world_pos.reshape(B, L, 22, 3)

    # Extract end-effector positions in order: L_Wrist(20), R_Wrist(21), L_Foot(10), R_Foot(11)
    ee_indices = [20, 21, 10, 11]
    fk_pos = world_pos[:, :, ee_indices, :]  # (B, L, 4, 3)
    fk_pos = fk_pos.reshape(B, L, 12)  # (B, L, 12)

    # Compute smooth L1 loss between FK positions and predicted positions
    loss_per_frame = torch.nn.functional.smooth_l1_loss(
        fk_pos, pos_pred, reduction='none'
    ).mean(dim=-1)  # (B, L)

    # Mask out padded frames
    if data_mask_temporal is not None:
        mask = data_mask_temporal.to(device).to(dtype)  # (B, L)
        loss_per_frame = loss_per_frame * mask
        mask_sum = torch.clamp(mask.sum(), min=1.0)
        return loss_per_frame.sum() / mask_sum
    else:
        return loss_per_frame.mean()


__all__ = ['motion147_fk_loss']
