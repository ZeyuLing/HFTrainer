"""HyMotion-UMO Pipeline: ODE-based inference with UMO temporal fusion.

Standard ODE integration from noise to clean motion, where the velocity
field fn(t, x) uses UMO fusion: E_in(x_t) + E_ctx(source + Emb(τ)).

Supports classifier-free guidance on both text and source motion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from hftrainer.models.motion.hymotion_umo.bundle import (
    META_OP_EDIT,
    META_OP_GENERATE,
    META_OP_PRESERVE,
)
from hftrainer.registry import PIPELINES


def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


def _mask_to_meta_ops(src_mask: Tensor) -> Tensor:
    """Convert per-dim binary mask (B, T, D) to frame-level meta-ops (B, T)."""
    frame_mask_ratio = src_mask.mean(dim=-1)
    frame_is_generate = (frame_mask_ratio > 0.5)
    return torch.where(
        frame_is_generate,
        torch.full_like(frame_is_generate, META_OP_GENERATE, dtype=torch.long),
        torch.full_like(frame_is_generate, META_OP_PRESERVE, dtype=torch.long),
    )


def _snap_mask_to_frame_level(src_mask: Tensor) -> Tensor:
    """Snap per-dim mask to frame-level: each frame is all-0 or all-1."""
    frame_mask_ratio = src_mask.mean(dim=-1, keepdim=True)
    frame_is_masked = (frame_mask_ratio > 0.5).float()
    return frame_is_masked.expand_as(src_mask)


@PIPELINES.register_module()
class HyMotionUMOPipeline:
    """Inference pipeline for HyMotion-UMO.

    Uses ODE integration to solve the flow matching ODE with UMO fusion.

    Parameters
    ----------
    bundle : HyMotionUMOBundle
        The model bundle.
    num_steps : int
        Number of ODE integration steps.
    text_guidance_scale : float
        CFG scale for text conditioning. >1.0 enables text CFG.
    source_guidance_scale : float
        CFG scale for source motion conditioning. >1.0 enables source CFG.
    """

    def __init__(
        self,
        bundle,
        num_steps: int = 50,
        text_guidance_scale: float = 5.0,
        source_guidance_scale: float = 1.0,
    ):
        self.bundle = bundle
        self.num_steps = num_steps
        self.text_guidance_scale = text_guidance_scale
        self.source_guidance_scale = source_guidance_scale

    @torch.no_grad()
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self._inference(batch)

    def _inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device = next(self.bundle.context_encoder.parameters()).device

        # ---- Source motion and mask ----
        src_motion = batch['src_motion'].to(device)
        B, T, D = src_motion.shape

        src_mask = batch.get('src_mask')
        if src_mask is not None:
            src_mask = src_mask.to(device)
        else:
            # No mask = all generate (pure T2M)
            src_mask = torch.ones_like(src_motion)

        # Expand/fix mask for 201-dim
        if src_mask.shape[-1] == D and D == 201:
            # Fix joint position mask (dims 135:201) to follow rotation mask
            joint_rot_mask = src_mask[:, :, 3::6]
            if joint_rot_mask.shape[-1] >= 22:
                joint_rot_mask = joint_rot_mask[:, :, :22]
                joint_pos_mask = joint_rot_mask.unsqueeze(-1).expand(-1, -1, -1, 3).reshape(B, T, 66)
                src_mask = torch.cat([src_mask[:, :, :135], joint_pos_mask], dim=-1)
        elif src_mask.shape[-1] < D:
            mask_d = src_mask.shape[-1]
            if mask_d == 135 and D == 201:
                joint_mask = src_mask[:, :, 3::6][:, :, :22]
                joint_pos_mask = joint_mask.unsqueeze(-1).expand(-1, -1, -1, 3).reshape(B, T, 66)
                src_mask = torch.cat([src_mask, joint_pos_mask], dim=-1)
            else:
                pad = torch.ones(B, T, D - mask_d, device=device)
                src_mask = torch.cat([src_mask, pad], dim=-1)

        tgt_length = batch.get('tgt_length', [T] * B)
        if isinstance(tgt_length, Tensor):
            tgt_length = tgt_length.tolist()

        tgt_padding_mask = _length_to_mask(
            torch.tensor(tgt_length, dtype=torch.long, device=device), T
        )

        # ---- Snap to frame-level, normalize and build meta-ops ----
        src_mask = _snap_mask_to_frame_level(src_mask)
        src_motion_norm = self.bundle.normalize_motion(src_motion)
        src_motion_norm = src_motion_norm * (1 - src_mask)
        meta_ops = _mask_to_meta_ops(src_mask)

        # ---- Text ----
        if 'text_vec_raw' in batch:
            vtxt_input = batch['text_vec_raw'].to(device)
            ctxt_input = batch['text_ctxt_raw'].to(device)
            ctxt_length = batch['text_ctxt_raw_length'].to(device)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])
        else:
            vtxt_input = self.bundle.null_vtxt_feat.expand(B, 1, -1)
            ctxt_input = self.bundle.null_ctxt_input.expand(B, 1, -1)
            ctxt_length = torch.tensor([1], device=device).expand(B)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, 1).expand(B, -1)

        # ---- CFG setup ----
        do_text_cfg = self.text_guidance_scale > 1.0
        do_source_cfg = self.source_guidance_scale > 1.0

        # Null source (for source CFG): zero source + all-generate meta-ops
        null_source = torch.zeros_like(src_motion_norm)
        null_meta_ops = torch.full_like(meta_ops, META_OP_GENERATE)

        def fn(t_val: Tensor, x: Tensor) -> Tensor:
            """ODE velocity field with optional dual CFG."""
            # Conditional prediction (with both text and source)
            pred_cond = self.bundle.predict_flow(
                x_t=x,
                source_motion=src_motion_norm,
                meta_ops=meta_ops,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=t_val.expand(B),
                x_mask_temporal=tgt_padding_mask,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )

            if self.bundle.pred_type == 'x1':
                t_eps = 0.05
                pred_cond = (pred_cond - x) / (1.0 - t_val).clamp_min(t_eps)

            if not do_text_cfg and not do_source_cfg:
                return pred_cond

            # Unconditional prediction (null text, null source)
            null_vtxt = self.bundle.null_vtxt_feat.expand_as(vtxt_input)
            null_ctxt = self.bundle.null_ctxt_input.expand_as(ctxt_input)

            pred_uncond = self.bundle.predict_flow(
                x_t=x,
                source_motion=null_source,
                meta_ops=null_meta_ops,
                ctxt_input=null_ctxt,
                vtxt_input=null_vtxt,
                timesteps=t_val.expand(B),
                x_mask_temporal=tgt_padding_mask,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )
            if self.bundle.pred_type == 'x1':
                pred_uncond = (pred_uncond - x) / (1.0 - t_val).clamp_min(t_eps)

            # Apply dual CFG
            result = pred_uncond

            if do_text_cfg:
                # Text-only conditioned (null source, real text)
                pred_text_only = self.bundle.predict_flow(
                    x_t=x,
                    source_motion=null_source,
                    meta_ops=null_meta_ops,
                    ctxt_input=ctxt_input,
                    vtxt_input=vtxt_input,
                    timesteps=t_val.expand(B),
                    x_mask_temporal=tgt_padding_mask,
                    ctxt_mask_temporal=ctxt_mask_temporal,
                )
                if self.bundle.pred_type == 'x1':
                    pred_text_only = (pred_text_only - x) / (1.0 - t_val).clamp_min(t_eps)

                result = result + self.text_guidance_scale * (pred_text_only - pred_uncond)

                if do_source_cfg:
                    # Full conditional = text + source
                    result = result + self.source_guidance_scale * (pred_cond - pred_text_only)
                else:
                    # Only text CFG: simple interpolation
                    result = pred_uncond + self.text_guidance_scale * (pred_cond - pred_uncond)
            elif do_source_cfg:
                # Only source CFG (no text CFG)
                result = pred_uncond + self.source_guidance_scale * (pred_cond - pred_uncond)

            return result

        # ---- ODE integration (midpoint method) ----
        y0 = torch.randn(B, T, D, device=device, dtype=src_motion.dtype)
        t = torch.linspace(0, 1, self.num_steps + 1, device=device, dtype=src_motion.dtype)

        x = y0
        for i in range(self.num_steps):
            t_curr = t[i]
            dt = t[i + 1] - t[i]

            # Midpoint method
            v_start = fn(t_curr, x)
            x_mid = x + v_start * (dt * 0.5)
            t_mid = t_curr + dt * 0.5
            v_mid = fn(t_mid, x_mid)
            x = x + v_mid * dt

        sampled = x

        # ---- Decode ----
        result = self.bundle.decode_motion_from_latent(sampled)
        result['latent'] = sampled
        return result
