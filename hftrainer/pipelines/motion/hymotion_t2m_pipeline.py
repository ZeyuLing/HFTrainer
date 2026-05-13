"""HyMotion-T2M Pipeline: ODE-based inference for text-to-motion generation.

Unlike HyMotion-M2M, this pipeline does NOT use VACE conditioning.
The transformer receives only x_t (noisy motion) + text conditions.
Supports classifier-free guidance via text_guidance_scale.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from hftrainer.registry import PIPELINES


def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


@PIPELINES.register_module()
class HyMotionT2MPipeline:
    """Inference pipeline for HyMotion-T2M text-to-motion generation.

    Uses ODE integration (torchdiffeq.odeint) to solve the flow matching
    ODE from noise to clean motion, conditioned on text only (no VACE).
    """

    def __init__(
        self,
        bundle,
        num_steps: int = 50,
        text_guidance_scale: float = 5.0,
    ):
        self.bundle = bundle
        self.num_steps = num_steps
        self.text_guidance_scale = text_guidance_scale

    @torch.no_grad()
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on a batch.

        Args:
            batch: dict with keys:
                - tgt_length or num_frames: List[int] — desired motion lengths
                - motion_dim: int (optional, inferred from bundle if not given)
                - text_vec_raw, text_ctxt_raw, text_ctxt_raw_length: optional
                  pre-encoded text embeddings
                - caption: optional list of text strings for online encoding

        Returns:
            Dict with keys: rot6d, transl, keypoints3d (optional), latent.
        """
        device = next(self.bundle.motion_transformer.parameters()).device

        # Determine sequence lengths
        tgt_length = batch.get('tgt_length', batch.get('num_frames'))
        if isinstance(tgt_length, Tensor):
            tgt_length = tgt_length.tolist()
        if isinstance(tgt_length, int):
            tgt_length = [tgt_length]

        B = len(tgt_length)
        L = max(tgt_length)

        # Pad to at least TRAIN_FRAMES (360) to match official HY-Motion-1.0 training.
        # The model was trained on 360-frame sequences; shorter sequences produce
        # different attention patterns and ODE dynamics. Pad noise to TRAIN_FRAMES,
        # run ODE on full padded length, then truncate output to requested length.
        TRAIN_FRAMES = 360
        L_padded = max(L, TRAIN_FRAMES)

        # Infer motion dim from the transformer output_dim
        motion_dim = batch.get('motion_dim', self.bundle.motion_transformer.output_dim)

        tgt_padding_mask = _length_to_mask(
            torch.tensor(tgt_length, dtype=torch.long, device=device), L_padded
        )

        # Prepare text
        if batch.get('text_vec_raw') is not None:
            vtxt_input = batch['text_vec_raw'].to(device)
            ctxt_input = batch['text_ctxt_raw'].to(device)
            ctxt_length = batch['text_ctxt_raw_length'].to(device)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])
        elif batch.get('caption') is not None:
            captions = batch['caption']
            if isinstance(captions, str):
                captions = [captions]
            text_feats = self.bundle.encode_text(captions)
            vtxt_input = text_feats['text_vec_raw'].to(device)
            ctxt_input = text_feats['text_ctxt_raw'].to(device)
            ctxt_length = text_feats['text_ctxt_raw_length'].to(device)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])
        else:
            vtxt_input = self.bundle.null_vtxt_feat.expand(B, 1, -1)
            ctxt_input = self.bundle.null_ctxt_input.expand(B, 1, -1)
            ctxt_length = torch.tensor([1], device=device).expand(B)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, 1).expand(B, -1)

        do_cfg = self.text_guidance_scale > 1.0

        # For CFG: prepare null text embeddings
        if do_cfg:
            null_vtxt = self.bundle.null_vtxt_feat.expand_as(vtxt_input)
            null_ctxt = self.bundle.null_ctxt_input.expand_as(ctxt_input)
            # Stack: [unconditional, conditional]
            vtxt_cfg = torch.cat([null_vtxt, vtxt_input], dim=0)
            ctxt_cfg = torch.cat([null_ctxt, ctxt_input], dim=0)
            ctxt_mask_cfg = torch.cat([
                torch.ones(B, ctxt_mask_temporal.shape[1], dtype=torch.bool, device=device),
                ctxt_mask_temporal,
            ], dim=0)

        # ODE function
        def fn(t_val: Tensor, x: Tensor) -> Tensor:
            if do_cfg:
                x_double = torch.cat([x, x], dim=0)
                x_pred = self.bundle.predict_flow(
                    x_input=x_double,
                    ctxt_input=ctxt_cfg,
                    vtxt_input=vtxt_cfg,
                    timesteps=t_val.expand(2 * B),
                    x_mask_temporal=tgt_padding_mask.repeat(2, 1),
                    ctxt_mask_temporal=ctxt_mask_cfg,
                )
            else:
                x_pred = self.bundle.predict_flow(
                    x_input=x,
                    ctxt_input=ctxt_input,
                    vtxt_input=vtxt_input,
                    timesteps=t_val.expand(B),
                    x_mask_temporal=tgt_padding_mask,
                    ctxt_mask_temporal=ctxt_mask_temporal,
                )

            if self.bundle.pred_type == 'x1':
                t_eps = 0.05
                if do_cfg:
                    x_pred = (x_pred - torch.cat([x, x], dim=0)) / (1.0 - t_val).clamp_min(t_eps)
                else:
                    x_pred = (x_pred - x) / (1.0 - t_val).clamp_min(t_eps)

            if do_cfg:
                pred_uncond, pred_text = x_pred.chunk(2, dim=0)
                x_pred = pred_uncond + self.text_guidance_scale * (pred_text - pred_uncond)
            return x_pred

        # Initial noise
        dtype = next(self.bundle.motion_transformer.parameters()).dtype
        y0 = torch.randn(B, L_padded, motion_dim, device=device, dtype=dtype)
        t = torch.linspace(0, 1, self.num_steps + 1, device=device, dtype=dtype)

        try:
            from torchdiffeq import odeint
            method = self.bundle._noise_scheduler_cfg.get('method', 'euler')
            trajectory = odeint(fn, y0, t, method=method)
        except ImportError:
            # Fallback: simple Euler integration
            trajectory = [y0]
            dt = 1.0 / self.num_steps
            x = y0
            for i in range(self.num_steps):
                t_val = torch.tensor(i * dt, device=device, dtype=dtype)
                v = fn(t_val, x)
                x = x + v * dt
                trajectory.append(x)
            trajectory = torch.stack(trajectory, dim=0)

        sampled = trajectory[-1]

        # Truncate padded frames back to the requested length.
        # ODE was run on L_padded (≥360) to match training dynamics;
        # only the first L frames contain the actual motion.
        sampled = sampled[:, :L, :]

        # Decode to motion
        result = self.bundle.decode_motion_from_latent(sampled)
        result['latent'] = sampled
        return result
