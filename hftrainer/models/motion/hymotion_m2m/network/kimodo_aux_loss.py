"""KIMODO-style auxiliary losses for HyMotion M2M v2.

This module implements a small set of auxiliary supervision terms on top of
the existing flow-matching ``M2MLoss``, designed to *suppress foot skating
and "slipping" failure modes* by aligning training-time supervision with the
loss formulation used in the KIMODO paper (Eq. 1).

Crucially we **do not change** the motion representation or the network
architecture.  All three auxiliary terms are computed *post-hoc* from the
predicted and ground-truth ``x1`` (the 198-dim motion in normalised space),
by running differentiable forward kinematics on each.

Loss terms (subset of KIMODO Eq. 1 that requires no representation changes):

* ``joint_pos`` (KIMODO γ₃)
    Smooth-L1 between predicted and GT *global* joint positions in world
    space, computed from FK on rotation+translation channels.  This gives
    the model a strong, well-conditioned position signal that does NOT
    depend on the relative-pelvis encoding used inside the 198-dim vector,
    and therefore cannot be "cheated" by translating the pelvis without
    moving the legs.

* ``joint_vel`` (KIMODO γ₄)
    Smooth-L1 between predicted and GT *global* joint velocities, computed
    as the temporal derivative of the FK joint positions above.  The
    KIMODO paper explicitly notes this is computed from *global* positions,
    not the (partially) root-relative ones — so a "slipping" trajectory
    (pelvis translates, legs static) immediately incurs a large velocity
    error at every joint, instead of being hidden under near-zero
    relative-position deltas.

* ``fk_consistency`` (KIMODO γ₇)
    Smooth-L1 between the position channels stored inside the predicted
    198-dim vector and the position derived by running FK on the predicted
    rotation+translation channels.  This is the same loss as the existing
    ``motion198_fk_loss``, replicated here so that the KIMODO-style block
    is self-contained and the original ``M2MLoss.fk_consistency`` can be
    safely turned off when the new class is in use.

All three terms are *padding-aware*: padded frames (as marked by
``data_mask_temporal``) are excluded from the numerator and denominator,
so the replicated tail produced by ``RandomCropPadding(pad_mode='replicate')``
or any zeroed-out tail in the trainer cannot leak into the loss.

The class is intentionally orthogonal to ``M2MLoss``: it is constructed
separately and called separately from the trainer.  To disable, simply set
all three weights to ``0.0`` (or skip instantiation in the bundle).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _safe_std(std: Tensor) -> Tensor:
    """Avoid div-by-zero in denormalisation (matches existing convention)."""
    return torch.where(std < 1e-3, torch.ones_like(std), std)


def _denormalize_198(x_norm: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Denormalize a (B, L, 198) tensor."""
    return x_norm * _safe_std(std) + mean


def _fk_global_positions(
    motion_135_denorm: Tensor,
    bone_offsets: Tensor,
    rotation_space: str,
) -> Tensor:
    """Run FK on a (B, L, 135) denormalised tensor and return world-space joint positions.

    Returns:
        (B, L, 22, 3) world-frame joint positions.
    """
    from hftrainer.motion.pipeline_utils.differentiable_fk import motion135_to_fk

    world_pos, _, _, _ = motion135_to_fk(
        motion_135_denorm, bone_offsets, rotation_space=rotation_space
    )
    return world_pos


def _strict_ric_relative(world_pos: Tensor) -> Tensor:
    """Convert world joint positions to strict non-pelvis RIC (B, L, 21*3).

    XYZ are all relative to the pelvis and the pelvis joint is dropped.  This
    matches the strict 198-dim layout used by ``Compute198DimPosition``.
    """
    pelvis = world_pos[..., 0:1, :]             # (B, L, 1, 3)
    body_rel = world_pos[..., 1:, :] - pelvis   # (B, L, 21, 3)
    leading = body_rel.shape[:-2]
    return body_rel.reshape(*leading, 63)


def _temporal_mean_masked(
    per_frame: Tensor,
    mask: Tensor,
) -> Tensor:
    """Average a per-frame loss tensor under a (B, L) mask.

    Args:
        per_frame: (B, L) loss values.
        mask: (B, L) {0,1} mask, 1 = include in average.

    Returns:
        Scalar loss = sum(per_frame * mask) / clamp(sum(mask), 1).
    """
    m = mask.to(per_frame.device).to(per_frame.dtype)
    denom = torch.clamp(m.sum(), min=1.0)
    return (per_frame * m).sum() / denom


class KimodoStyleAuxLoss(nn.Module):
    """KIMODO-style auxiliary losses (joint_pos / joint_vel / fk_consistency).

    All three terms operate on *denormalised* 198-dim motion via FK.  They
    are padding-aware and do NOT use ``generation_mask`` — KIMODO supervises
    every frame uniformly, and under MAN the known regions automatically
    contribute (near-)zero loss anyway because the predicted x1 there is
    forced to ground truth.

    Parameters
    ----------
    joint_pos_weight : float
        Weight for the global joint position L1 loss (KIMODO γ₃≈10).
    joint_vel_weight : float
        Weight for the global joint velocity L1 loss (KIMODO γ₄≈3).
    fk_consistency_weight : float
        Weight for the FK consistency L1 loss between 198-dim pos channels
        and FK-derived rel-pelvis pos (KIMODO γ₇≈5).
    loss_type : str
        ``'smooth_l1'`` (default) or ``'l1'``.
    motion_dim : int
        Total dim of the motion vector, expected 198 (= 135 trans+rot6d + 63
        pos channels).  The class only runs when ``motion_dim >= 198``.
    fk_consistency_warmup_steps : int
        Linear warm-up over this many steps (matches existing M2MLoss
        convention).  ``0`` disables warm-up.
    timestep_squared_weighting : bool
        If True (default), multiply each term by ``t²`` (matches the
        existing ``motion198_fk_loss`` t-weighting).  This down-weights
        pure-noise samples where FK on noisy x1 is uninformative.
    """

    def __init__(
        self,
        joint_pos_weight: float = 0.0,
        joint_vel_weight: float = 0.0,
        fk_consistency_weight: float = 0.0,
        loss_type: str = "smooth_l1",
        motion_dim: int = 198,
        fk_consistency_warmup_steps: int = 0,
        joint_pos_warmup_steps: int = 0,
        joint_vel_warmup_steps: int = 0,
        timestep_squared_weighting: bool = True,
    ):
        super().__init__()
        self.joint_pos_weight = float(joint_pos_weight)
        self.joint_vel_weight = float(joint_vel_weight)
        self.fk_consistency_weight = float(fk_consistency_weight)
        self.fk_consistency_warmup_steps = int(fk_consistency_warmup_steps)
        self.joint_pos_warmup_steps = int(joint_pos_warmup_steps)
        self.joint_vel_warmup_steps = int(joint_vel_warmup_steps)
        self.motion_dim = int(motion_dim)
        self.timestep_squared_weighting = bool(timestep_squared_weighting)

        if loss_type == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        elif loss_type == "l1":
            self.loss_fn = F.l1_loss
        else:
            raise ValueError(f"Unsupported loss_type for KimodoStyleAuxLoss: {loss_type}")

    @property
    def enabled(self) -> bool:
        return (
            self.joint_pos_weight > 0.0
            or self.joint_vel_weight > 0.0
            or self.fk_consistency_weight > 0.0
        )

    @staticmethod
    def _warmup(weight: float, warmup_steps: int, global_step: Optional[int]) -> float:
        if weight == 0.0 or warmup_steps <= 0 or global_step is None:
            return weight
        if global_step >= warmup_steps:
            return weight
        return weight * (float(global_step) / float(warmup_steps))

    def forward(
        self,
        pred_x1_norm: Tensor,
        gt_x1_norm: Tensor,
        mean: Tensor,
        std: Tensor,
        bone_offsets: Tensor,
        rotation_space: str = "local",
        data_mask_temporal: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Compute the KIMODO-style auxiliary losses.

        Args:
            pred_x1_norm: (B, L, D) predicted x1 in normalised space.
            gt_x1_norm:   (B, L, D) ground-truth x1 in normalised space.
            mean, std:    (D,) normalisation buffers.
            bone_offsets: (22, 3) SMPL-22 bone offsets.
            rotation_space: ``'local'`` or ``'global'`` for the rot6d channels.
            data_mask_temporal: (B, L) padding mask, 1 = valid frame, 0 = padded.
                If None, all frames are considered valid (NOT recommended).
            timesteps: (B,) diffusion timesteps in ``[0, 1]`` (after
                normalising to a continuous flow timestep).  Used for the
                ``t²`` re-weighting if enabled.
            global_step: optional integer step for warm-up scheduling.

        Returns:
            Dict[str, Tensor] with optional keys ``aux_joint_pos``,
            ``aux_joint_vel``, ``aux_fk_consistency``.  Empty dict if no term
            is enabled or motion_dim < 198 (silently skipped).
        """
        out: Dict[str, Tensor] = {}
        if not self.enabled:
            return out

        D = pred_x1_norm.shape[-1]
        if D < self.motion_dim:
            # Silently skip (e.g. 135-dim pretraining run); aux losses are
            # only meaningful for 198-dim models that carry pos channels.
            return out

        if data_mask_temporal is None:
            B, L = pred_x1_norm.shape[:2]
            data_mask_temporal = torch.ones(
                B, L, device=pred_x1_norm.device, dtype=pred_x1_norm.dtype
            )
        else:
            data_mask_temporal = data_mask_temporal.to(pred_x1_norm.device)

        # Align mask length to the motion length (e.g. when ref_pose has
        # been prepended onto tgt_padding_mask but x1 only covers tgt).
        if data_mask_temporal.shape[-1] != pred_x1_norm.shape[1]:
            data_mask_temporal = data_mask_temporal[..., -pred_x1_norm.shape[1]:]

        # ------------------------------------------------------------------
        # Denormalise once.
        # ------------------------------------------------------------------
        pred_denorm = _denormalize_198(pred_x1_norm, mean, std)
        gt_denorm = _denormalize_198(gt_x1_norm, mean, std)

        pred_135 = pred_denorm[..., :135]
        gt_135 = gt_denorm[..., :135]

        # ------------------------------------------------------------------
        # FK on both pred and GT once; reuse below.
        # ------------------------------------------------------------------
        need_fk = (
            self.joint_pos_weight > 0.0
            or self.joint_vel_weight > 0.0
            or self.fk_consistency_weight > 0.0
        )
        if not need_fk:
            return out

        pred_world = _fk_global_positions(pred_135, bone_offsets, rotation_space)  # (B,L,22,3)
        gt_world = _fk_global_positions(gt_135, bone_offsets, rotation_space)

        # Optional t² re-weighting (matches existing motion198_fk_loss).
        if self.timestep_squared_weighting and timesteps is not None:
            t_sq = (timesteps.to(pred_world.device).to(pred_world.dtype) ** 2)  # (B,)
        else:
            t_sq = None

        # ==================================================================
        # 1) joint_pos loss — global joint positions, smooth_l1
        # ==================================================================
        if self.joint_pos_weight > 0.0:
            per_pt = self.loss_fn(pred_world, gt_world, reduction="none")  # (B,L,22,3)
            # Mean over joint and xyz dims; sum/avg over time under mask.
            per_frame = per_pt.mean(dim=(-1, -2))  # (B, L)
            if t_sq is not None:
                per_frame = per_frame * t_sq.unsqueeze(-1)
            base = _temporal_mean_masked(per_frame, data_mask_temporal)
            w = self._warmup(self.joint_pos_weight, self.joint_pos_warmup_steps, global_step)
            out["aux_joint_pos"] = w * base

        # ==================================================================
        # 2) joint_vel loss — global joint velocities (finite difference)
        # ==================================================================
        if self.joint_vel_weight > 0.0:
            pred_vel = pred_world[:, 1:] - pred_world[:, :-1]  # (B, L-1, 22, 3)
            gt_vel = gt_world[:, 1:] - gt_world[:, :-1]
            per_pt = self.loss_fn(pred_vel, gt_vel, reduction="none")
            per_frame = per_pt.mean(dim=(-1, -2))  # (B, L-1)
            # velocity-frame is valid only when both endpoints are valid.
            vel_mask = (
                data_mask_temporal[:, 1:].to(per_frame.dtype)
                * data_mask_temporal[:, :-1].to(per_frame.dtype)
            )
            if t_sq is not None:
                per_frame = per_frame * t_sq.unsqueeze(-1)
            base = _temporal_mean_masked(per_frame, vel_mask)
            w = self._warmup(self.joint_vel_weight, self.joint_vel_warmup_steps, global_step)
            out["aux_joint_vel"] = w * base

        # ==================================================================
        # 3) fk_consistency — pos channels (rel-pelvis) vs FK rel-pelvis pos
        # ==================================================================
        if self.fk_consistency_weight > 0.0:
            pred_pos_chan = pred_denorm[..., 135:]                      # (B, L, 63)
            fk_pos = _strict_ric_relative(pred_world)                   # (B, L, 63)
            per_pt = self.loss_fn(pred_pos_chan, fk_pos, reduction="none")
            per_frame = per_pt.mean(dim=-1)  # (B, L)
            if t_sq is not None:
                per_frame = per_frame * t_sq.unsqueeze(-1)
            base = _temporal_mean_masked(per_frame, data_mask_temporal)
            w = self._warmup(
                self.fk_consistency_weight, self.fk_consistency_warmup_steps, global_step
            )
            out["aux_fk_consistency"] = w * base

        return out
