"""HyMotion-UMO Trainer: flow-matching training with UMO temporal fusion.

Training:
  1. Load pretrained T2M weights -> init E_ctx from input_encoder -> freeze backbone
  2. Sample mask using M2M universal mask strategies (M1-M7)
  3. Convert per-dim mask to frame-level meta-ops: P (all unmasked), G (any masked)
  4. Build source_motion: normalized motion where mask=0, zero where mask=1
  5. Flow matching: x_t = (1-t)*noise + t*tgt_motion
  6. Fuse: E_in(x_t) + E_ctx(source + Emb(τ))
  7. Loss: SmoothL1(pred_velocity, gt_velocity)

Only ~0.207M parameters trained (E_ctx + meta_op_embeddings).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from hftrainer.models.motion.hymotion_umo.bundle import (
    META_OP_EDIT,
    META_OP_GENERATE,
    META_OP_PRESERVE,
)
from hftrainer.registry import TRAINERS
from hftrainer.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


def _mask_to_meta_ops(src_mask: Tensor) -> Tensor:
    """Convert per-dim binary mask (B, T, D) to frame-level meta-ops (B, T).

    UMO operates at frame level (whole-body), so we first collapse
    the per-dim mask to per-frame:
      - If MAJORITY of dims are mask=1 for a frame -> G (Generate)
      - If MAJORITY of dims are mask=0 for a frame -> P (Preserve)

    Returns:
        (B, T) long tensor with values in {0=P, 1=G, 2=E}.
    """
    D = src_mask.shape[-1]
    frame_mask_ratio = src_mask.mean(dim=-1)  # (B, T), ratio of masked dims
    frame_is_generate = (frame_mask_ratio > 0.5)  # (B, T) bool
    meta_ops = torch.where(
        frame_is_generate,
        torch.full_like(frame_is_generate, META_OP_GENERATE, dtype=torch.long),
        torch.full_like(frame_is_generate, META_OP_PRESERVE, dtype=torch.long),
    )
    return meta_ops


def _snap_mask_to_frame_level(src_mask: Tensor) -> Tensor:
    """Snap per-dim mask to frame-level: each frame is all-0 or all-1.

    UMO doesn't support per-joint control, so we collapse:
      - If majority of dims are mask=1 -> entire frame mask=1
      - If majority of dims are mask=0 -> entire frame mask=0
    """
    frame_mask_ratio = src_mask.mean(dim=-1, keepdim=True)  # (B, T, 1)
    frame_is_masked = (frame_mask_ratio > 0.5).float()       # (B, T, 1)
    return frame_is_masked.expand_as(src_mask)                # (B, T, D)


@TRAINERS.register_module()
class HyMotionUMOTrainer(BaseTrainer):
    """Trainer for UMO-style temporal fusion on frozen MMDiT backbone.

    On first train_step, initializes E_ctx from pretrained input_encoder
    and freezes the backbone. Only E_ctx + meta_op_embeddings are trained.
    """

    def __init__(
        self,
        bundle,
        val_num_steps: int = 10,
        max_text_len: int = 128,
        source_cond_mask_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__(bundle)
        self.val_num_steps = val_num_steps
        self.max_text_len = max_text_len
        self.source_cond_mask_prob = source_cond_mask_prob
        self._initialized = False

    def _lazy_init(self):
        """One-time initialization: copy E_in -> E_ctx, freeze backbone."""
        if self._initialized:
            return
        self.bundle.init_context_encoder_from_pretrained()
        self.bundle.freeze_backbone()
        self._initialized = True

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self._lazy_init()

        device = next(self.bundle.context_encoder.parameters()).device

        # ---- Extract batch data ----
        # The dataset provides either:
        # (a) UMO-specific: motion + src_mask (from PrepareM2MUniversalMask)
        # (b) T2M-only: motion only (all frames masked -> pure T2M)
        if 'src_motion' in batch:
            # M2M-style dataset with mask
            src_motion = batch['src_motion'].to(device)
            tgt_motion = batch['tgt_motion'].to(device)
            src_mask = batch.get('src_mask')
            if src_mask is not None:
                src_mask = src_mask.to(device)
        elif 'motion' in batch:
            # T2M-style: treat all frames as generate
            motion = batch['motion'].to(device)
            src_motion = motion
            tgt_motion = motion
            # All-ones mask: everything needs generation (pure T2M mode)
            src_mask = torch.ones_like(motion)
        else:
            raise KeyError("Batch must contain 'src_motion' or 'motion'.")

        # ---- Get lengths ----
        tgt_length_list: List[int] = batch.get('tgt_length', [tgt_motion.shape[1]])
        if isinstance(tgt_length_list, Tensor):
            tgt_length_list = tgt_length_list.tolist()
        if not isinstance(tgt_length_list, (list, tuple)):
            tgt_length_list = [tgt_length_list]
        src_length_list = batch.get('src_length', tgt_length_list)
        if isinstance(src_length_list, Tensor):
            src_length_list = src_length_list.tolist()

        B, T, D = tgt_motion.shape

        # ---- Normalize ----
        src_motion_norm = self.bundle.normalize_motion(src_motion)
        tgt_motion_norm = self.bundle.normalize_motion(tgt_motion)

        # ---- Build frame-level mask if needed ----
        if src_mask is None:
            src_mask = torch.ones(B, T, D, device=device)

        # ---- Fix joint position mask for 201-dim data ----
        # PrepareM2MUniversalMask operates on a (T, 23) joint-group grid and
        # expands to (T, D). For D=201, dims [135:201] (joint positions) get
        # padded with 1.0 (generate) by default. We need to fix this so that
        # each joint's 3D position mask follows its rotation mask.
        if D == 201 and src_mask.shape[-1] == 201:
            # Extract per-joint rotation mask: one flag per joint from rot6d dims
            # Joint j's rot6d starts at dim 3+j*6, take first dim as representative
            joint_rot_mask = src_mask[:, :, 3::6]  # (B, T, 22) from 135-dim region
            if joint_rot_mask.shape[-1] >= 22:
                joint_rot_mask = joint_rot_mask[:, :, :22]
                # Rebuild joint position mask (dims [135:201])
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

        # ---- Snap mask to frame-level (UMO doesn't support per-joint) ----
        src_mask = _snap_mask_to_frame_level(src_mask)

        # ---- Zero src_motion in masked regions (for P frames: keep, for G frames: zero) ----
        src_motion_norm = src_motion_norm * (1 - src_mask)

        # ---- Convert to frame-level meta-ops ----
        meta_ops = _mask_to_meta_ops(src_mask)  # (B, T) long

        # ---- Zero out padding frames ----
        for i in range(B):
            tgt_len = int(tgt_length_list[i])
            src_len = int(src_length_list[i])
            if tgt_len < T:
                tgt_motion_norm[i, tgt_len:] = 0.0
            if src_len < T:
                src_motion_norm[i, src_len:] = 0.0
                meta_ops[i, src_len:] = META_OP_PRESERVE  # padding frames = preserve (zero source)

        # ---- Build temporal mask ----
        tgt_padding_mask = _length_to_mask(
            torch.tensor(tgt_length_list, dtype=torch.long, device=device), T
        )

        # ---- Prepare text ----
        vtxt_input, ctxt_input, ctxt_mask_temporal = self._prepare_text(batch, B, device)

        # ---- Flow matching: sample t, build x_t ----
        x1 = tgt_motion_norm
        x0 = torch.randn_like(x1)

        if self.bundle.pred_type == 'x1':
            z = torch.randn(B, dtype=x1.dtype, device=device) * 0.8 + (-0.8)
            timesteps = torch.sigmoid(z)
        else:
            timesteps = torch.rand(B, dtype=x1.dtype, device=device)

        t = timesteps.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        x_t = (1 - t) * x0 + t * x1

        # ---- Source condition dropout (CFG for source motion) ----
        # With probability source_cond_mask_prob, replace source embedding
        # with null source to enable source-motion CFG at inference.
        if self.training and self.source_cond_mask_prob > 0.0:
            source_drop_mask = torch.bernoulli(
                torch.ones(B, device=device) * self.source_cond_mask_prob
            ).bool()
            if source_drop_mask.any():
                # For dropped samples: zero out source, set all meta_ops to G
                src_motion_norm[source_drop_mask] = 0.0
                meta_ops[source_drop_mask] = META_OP_GENERATE

        # ---- Forward: UMO fusion + backbone ----
        pred = self.bundle.predict_flow(
            x_t=x_t,
            source_motion=src_motion_norm,
            meta_ops=meta_ops,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=tgt_padding_mask,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )

        # ---- Compute loss ----
        if self.bundle.pred_type == 'velocity':
            gt_velocity = x1 - x0
            losses = self.bundle.m2m_loss(
                pred_vel=pred,
                gt_vel=gt_velocity,
                pred_x1=None,
                gt_x1=None,
                pred_keypoints3d=None,
                gt_keypoints3d=None,
                data_mask_temporal=tgt_padding_mask,
                global_step=self.get_global_step(),
            )
        elif self.bundle.pred_type == 'x1':
            t_eps = 0.05
            gt_velocity = (x1 - x_t) / (1 - t).clamp_min(t_eps)
            pred_velocity = (pred - x_t) / (1 - t).clamp_min(t_eps)
            losses = self.bundle.m2m_loss(
                pred_vel=pred_velocity,
                gt_vel=gt_velocity,
                pred_x1=pred,
                gt_x1=x1,
                pred_keypoints3d=None,
                gt_keypoints3d=None,
                data_mask_temporal=tgt_padding_mask,
                global_step=self.get_global_step(),
            )
        else:
            raise ValueError(f'Unsupported pred_type: {self.bundle.pred_type}')

        loss = self.sum_train_losses(losses)
        result = {'loss': loss}
        for k, v in losses.items():
            result[f'loss_{k}'] = v.detach()

        # Log mask stats
        with torch.no_grad():
            n_preserve = (meta_ops == META_OP_PRESERVE).float().mean()
            n_generate = (meta_ops == META_OP_GENERATE).float().mean()
            result['meta_op_preserve_ratio'] = n_preserve
            result['meta_op_generate_ratio'] = n_generate

        return result

    def _prepare_text(
        self,
        batch: Dict[str, Any],
        B: int,
        device: torch.device,
    ):
        """Prepare text embeddings from batch (pre-extracted or online)."""
        pad_len = self.max_text_len

        if batch.get('text_vec_raw') is not None:
            vtxt_input = batch['text_vec_raw'].to(device)
            ctxt_raw = batch['text_ctxt_raw']

            if isinstance(ctxt_raw, (list, tuple)):
                feat_dim = ctxt_raw[0].shape[-1]
                ctxt_padded = ctxt_raw[0].new_zeros(len(ctxt_raw), pad_len, feat_dim)
                for i, t in enumerate(ctxt_raw):
                    seq = min(t.shape[0], pad_len)
                    ctxt_padded[i, :seq] = t[:seq]
                ctxt_input = ctxt_padded.to(device)
            else:
                cur_len = ctxt_raw.shape[1]
                if cur_len < pad_len:
                    ctxt_input = F.pad(ctxt_raw, (0, 0, 0, pad_len - cur_len)).to(device)
                else:
                    ctxt_input = ctxt_raw[:, :pad_len].to(device)

            ctxt_length = batch['text_ctxt_raw_length'].to(device).clamp(max=pad_len)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)

            # Force null embeddings for samples without captions
            null_mask = (ctxt_length == 0)
            if null_mask.any():
                null_v = self.bundle.null_vtxt_feat.expand_as(vtxt_input)
                null_c = self.bundle.null_ctxt_input.expand_as(ctxt_input)
                vtxt_input = torch.where(
                    null_mask.view(B, 1, 1).expand_as(vtxt_input), null_v, vtxt_input
                )
                ctxt_input = torch.where(
                    null_mask.view(B, 1, 1).expand_as(ctxt_input), null_c, ctxt_input
                )

            vtxt_input, ctxt_input = self.bundle.mask_text_cond(
                vtxt_input, ctxt_input,
                force_mask=False,
                cond_mask_prob=self.bundle.cond_mask_prob,
            )
        elif 'caption' in batch and batch['caption'] is not None:
            captions = batch['caption']
            if isinstance(captions, torch.Tensor):
                captions = captions.tolist()
            captions = [c if c is not None else '' for c in captions]
            with torch.no_grad():
                text_feats = self.bundle.encode_text(captions)
            vtxt_input = text_feats['text_vec_raw'].to(device)
            ctxt_input = text_feats['text_ctxt_raw'].to(device)
            ctxt_length = text_feats['text_ctxt_raw_length'].to(device)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])
            vtxt_input, ctxt_input = self.bundle.mask_text_cond(
                vtxt_input, ctxt_input,
                force_mask=False,
                cond_mask_prob=self.bundle.cond_mask_prob,
            )
        else:
            vtxt_input = self.bundle.null_vtxt_feat.expand(B, 1, -1)
            ctxt_input = self.bundle.null_ctxt_input.expand(B, 1, -1)
            ctxt_length = torch.tensor([1], device=device).expand(B)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, 1).expand(B, -1)

        return vtxt_input, ctxt_input, ctxt_mask_temporal

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        from hftrainer.pipelines.motion.hymotion_umo_pipeline import HyMotionUMOPipeline

        pipeline = HyMotionUMOPipeline(
            bundle=self.bundle,
            num_steps=self.val_num_steps,
        )
        with torch.no_grad():
            output = pipeline(batch)
        return {'preds': output}
