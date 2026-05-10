"""HyMotion-M2M v3 Trainer: DSCF (Dual-Stream Condition Fusion) training.

Key differences from v1 trainer (hymotion_m2m_trainer.py):
  - No VACE context construction: condition_mask + known_motion passed directly
    to the v3 transformer, which handles MotionCondEncoder + RoleEmbedding internally.
  - Simpler _prepare_and_forward: no prepare_vace_input(), no concat of x_t+vace.
  - edit_mask propagated to forward for EDIT role embedding assignment.
  - Mask-aware noise retained for positional hint consistency.

Training forward:
  1. Prepare padding and masks
  2. Prepare text embeddings (null or encoded)
  3. Sample timesteps, create x_t via flow matching interpolation
  4. Apply mask-aware noise (known regions stay clean)
  5. Forward: bundle.predict_flow(x_t, condition_mask, known_motion, edit_mask, ...)
  6. Compute loss via bundle.m2m_loss
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from hftrainer.registry import TRAINERS
from hftrainer.trainers.base_trainer import BaseTrainer


def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


@TRAINERS.register_module()
class HyMotionM2Mv3Trainer(BaseTrainer):
    """Trainer for HyMotion-M2M v3 (DSCF) flow-matching motion editing.

    Training forward:
      1. Prepare padding and masks
      2. Prepare text embeddings (null or encoded)
      3. Sample timesteps, create x_t via flow matching
      4. Apply mask-aware noise (optional)
      5. Forward: bundle.predict_flow(x_t, condition_mask, known_motion, ...)
      6. Compute loss via bundle.m2m_loss
    """

    def __init__(
        self,
        bundle,
        val_num_steps: int = 10,
        max_text_len: int = 128,
        mask_aware_noise: bool = True,
        **kwargs,
    ):
        super().__init__(bundle)
        self.val_num_steps = val_num_steps
        self.mask_aware_noise = mask_aware_noise
        self.max_text_len = max_text_len

    def _prepare_and_forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs and run a single forward pass.

        Returns a context dict with all intermediate tensors needed to
        compute loss.

        Keys in returned dict:
          device, src_motion, tgt_motion, src_mask, src_length_list,
          tgt_length_list, ref_pose, tgt_padding_mask, vtxt_input,
          ctxt_input, ctxt_mask_temporal, x0, x1, x_t, timesteps, t,
          pred, generation_mask, edit_mask
        """
        device = next(self.bundle.motion_transformer.parameters()).device

        # Source and target motions
        src_motion = batch['src_motion'].to(device)
        tgt_motion = batch['tgt_motion'].to(device)
        src_mask = batch.get('src_mask')
        if src_mask is not None:
            src_mask = src_mask.to(device)

        # Lengths
        tgt_length_list: List[int] = batch['tgt_length']
        if isinstance(tgt_length_list, Tensor):
            tgt_length_list = tgt_length_list.tolist()
        src_length_list = batch.get('src_length', tgt_length_list)
        if isinstance(src_length_list, Tensor):
            src_length_list = src_length_list.tolist()

        # Normalize motions
        src_motion = self.bundle.normalize_motion(src_motion)
        tgt_motion = self.bundle.normalize_motion(tgt_motion)

        # Build edit_mask: (B, L) bool — True where sample is in edit mode
        # and frame is in the mask region. Used for EDIT role assignment.
        edit_flags = batch.get('edit_mode', None)
        edit_mask = None

        if src_mask is not None:
            if edit_flags is not None:
                if isinstance(edit_flags, Tensor):
                    keep = edit_flags.view(-1, 1, 1).float().to(src_motion.device)
                elif isinstance(edit_flags, (list, tuple)):
                    keep = torch.tensor([float(bool(e)) for e in edit_flags],
                                        device=src_motion.device).view(-1, 1, 1)
                else:
                    keep = torch.zeros(1, 1, 1, device=src_motion.device)
                # For completion (keep=0): zero mask regions
                # For edit (keep=1): keep LQ values (reactive channel)
                src_motion = src_motion * (1 - src_mask * (1 - keep))

                # Build per-frame edit_mask: True where edit mode + masked
                # edit_mask: (B, L) bool
                if isinstance(edit_flags, Tensor):
                    is_edit = edit_flags.bool().to(device)
                elif isinstance(edit_flags, (list, tuple)):
                    is_edit = torch.tensor([bool(e) for e in edit_flags], device=device)
                else:
                    is_edit = torch.zeros(src_motion.shape[0], dtype=torch.bool, device=device)
                # Frame is "edit" if edit_mode=True AND frame has mask>0.5
                frame_masked = src_mask.mean(dim=-1) > 0.5  # (B, L)
                edit_mask = is_edit.unsqueeze(-1) & frame_masked  # (B, L)
            else:
                # No edit_mode flag: all completion, zero mask regions
                src_motion = src_motion * (1 - src_mask)

        # Zero out padded frames
        B, L_src, D = src_motion.shape
        L_tgt = tgt_motion.shape[1]
        for i in range(B):
            tgt_len = int(tgt_length_list[i])
            src_len = int(src_length_list[i])
            if tgt_len < L_tgt:
                tgt_motion[i, tgt_len:] = 0.0
            if src_len < L_src:
                src_motion[i, src_len:] = 0.0
                if src_mask is not None:
                    src_mask[i, src_len:] = 0.0

        ref_pose = batch.get('ref_pose')
        if ref_pose is not None and not isinstance(ref_pose, Tensor):
            ref_pose = None
        if ref_pose is not None:
            ref_pose = ref_pose.to(device)

        # 1. Prepare padding
        src_motion, src_mask, tgt_motion, src_length_list, tgt_length_list, tgt_padding_mask = (
            self.bundle.prepare_padding(
                src_motion, tgt_motion, tgt_length_list, src_mask, src_length_list, ref_pose
            )
        )

        # 2. Prepare text: use null embeddings (unconditioned) or batch text
        B = tgt_motion.shape[0]
        if batch.get('text_vec_raw') is not None:
            vtxt_input = batch['text_vec_raw'].to(device)
            ctxt_raw = batch['text_ctxt_raw']
            pad_len = self.max_text_len
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

        # 3. Flow matching: sample t, build x_t
        x1 = tgt_motion
        if ref_pose is not None:
            x1 = torch.cat([ref_pose, x1], dim=1)
        x0 = torch.randn_like(x1)

        if self.bundle.pred_type == 'x1':
            z = torch.randn(B, dtype=x1.dtype, device=device) * 0.8 + (-0.8)
            timesteps = torch.sigmoid(z)
        else:
            timesteps = torch.rand(B, dtype=x1.dtype, device=device)

        t = timesteps.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - t) * x0 + t * x1

        # 4. Mask-aware noise: keep known regions clean in x_t
        # src_mask: (B, L, D), 1=generate, 0=known
        if self.mask_aware_noise and src_mask is not None:
            keep_mask = 1 - src_mask  # 1=known
            x_t = x_t * src_mask + x1 * keep_mask

        # Prepare condition_mask and known_motion for v3 transformer
        # condition_mask = src_mask (B, L, motion_dim): 1=generate, 0=known
        # known_motion = src_motion (B, L, motion_dim): values where mask=0, zero where mask=1
        condition_mask = src_mask if src_mask is not None else torch.zeros_like(x1)
        known_motion = src_motion

        # Handle ref_pose prepend: if ref_pose exists, condition_mask and known_motion
        # need to be prepended with ref_pose info (ref_pose is fully known → mask=0)
        if ref_pose is not None:
            L_ref = ref_pose.shape[1]
            ref_mask = torch.zeros(B, L_ref, D, dtype=condition_mask.dtype, device=device)
            condition_mask = torch.cat([ref_mask, condition_mask], dim=1)
            known_motion = torch.cat([ref_pose, known_motion], dim=1)
            # edit_mask also needs ref prepend (ref frames are not in edit mode)
            if edit_mask is not None:
                ref_edit = torch.zeros(B, L_ref, dtype=torch.bool, device=device)
                edit_mask = torch.cat([ref_edit, edit_mask], dim=1)

        # 5. Forward through v3 transformer
        pred = self.bundle.predict_flow(
            x=x_t,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            condition_mask=condition_mask,
            known_motion=known_motion,
            x_mask_temporal=tgt_padding_mask,
            ctxt_mask_temporal=ctxt_mask_temporal,
            edit_mask=edit_mask,
        )

        # Generation mask for mask-aware loss weighting
        generation_mask = None
        if self.mask_aware_noise and src_mask is not None:
            generation_mask = condition_mask  # (B, L, D), 1=generate

        return {
            'device': device,
            'src_motion': src_motion,
            'tgt_motion': tgt_motion,
            'src_mask': src_mask,
            'condition_mask': condition_mask,
            'known_motion': known_motion,
            'edit_mask': edit_mask,
            'src_length_list': src_length_list,
            'tgt_length_list': tgt_length_list,
            'ref_pose': ref_pose,
            'tgt_padding_mask': tgt_padding_mask,
            'vtxt_input': vtxt_input,
            'ctxt_input': ctxt_input,
            'ctxt_mask_temporal': ctxt_mask_temporal,
            'x0': x0,
            'x1': x1,
            'x_t': x_t,
            'timesteps': timesteps,
            't': t,
            'pred': pred,
            'generation_mask': generation_mask,
        }

    def _compute_base_loss(self, ctx: Dict[str, Any]) -> Dict[str, Tensor]:
        """Compute the standard flow-matching loss dict from context."""
        x0 = ctx['x0']
        x1 = ctx['x1']
        x_t = ctx['x_t']
        t = ctx['t']
        pred = ctx['pred']
        timesteps = ctx['timesteps']
        tgt_padding_mask = ctx['tgt_padding_mask']
        generation_mask = ctx['generation_mask']

        if self.bundle.pred_type == 'velocity':
            gt_velocity = x1 - x0
            pred_velocity = pred
            pred_x1_for_smooth = None
            gt_x1_for_smooth = None
            kimodo_aux = getattr(self.bundle, 'kimodo_aux_loss', None)
            kimodo_enabled = bool(kimodo_aux is not None and kimodo_aux.enabled)
            need_x1 = (
                self.bundle.m2m_loss.motion_smoothness_weight > 0.0
                or self.bundle.m2m_loss.keypoints3d_weight > 0.0
                or self.bundle.m2m_loss.fk_consistency_weight > 0.0
                or kimodo_enabled
            )
            if need_x1:
                pred_x1_for_smooth = x_t + (1 - t) * pred_velocity
                gt_x1_for_smooth = x1

            # FK loss
            pred_kp3d = None
            gt_kp3d = None
            if self.bundle.m2m_loss.keypoints3d_weight > 0.0 and pred_x1_for_smooth is not None:
                pred_kp3d, gt_kp3d = self._compute_fk_keypoints(
                    pred_x1_for_smooth, gt_x1_for_smooth
                )

            # FK consistency loss for 198-dim
            fk_loss = None
            if (self.bundle.m2m_loss.fk_consistency_weight > 0.0
                    and pred_x1_for_smooth is not None
                    and self.bundle.mean.shape[0] >= 198):
                fk_loss = self._compute_fk_consistency_loss(
                    pred_x1_for_smooth, timesteps, tgt_padding_mask
                )

            losses = self.bundle.m2m_loss(
                pred_vel=pred_velocity,
                gt_vel=gt_velocity,
                pred_x1=pred_x1_for_smooth,
                gt_x1=gt_x1_for_smooth,
                pred_keypoints3d=pred_kp3d,
                gt_keypoints3d=gt_kp3d,
                data_mask_temporal=tgt_padding_mask,
                global_step=self.get_global_step(),
                generation_mask=generation_mask,
                fk_consistency_loss=fk_loss,
            )

            # KIMODO-style auxiliary losses
            aux_losses = self._compute_kimodo_aux_loss(
                pred_x1_for_smooth, x1, timesteps, tgt_padding_mask
            )
            if aux_losses:
                losses.update(aux_losses)
        elif self.bundle.pred_type == 'x1':
            t_eps = 0.05
            gt_velocity = (x1 - x_t) / (1 - t).clamp_min(t_eps)
            pred_velocity = (pred - x_t) / (1 - t).clamp_min(t_eps)

            pred_kp3d = None
            gt_kp3d = None
            if self.bundle.m2m_loss.keypoints3d_weight > 0.0:
                pred_kp3d, gt_kp3d = self._compute_fk_keypoints(pred, x1)

            losses = self.bundle.m2m_loss(
                pred_vel=pred_velocity,
                gt_vel=gt_velocity,
                pred_x1=pred,
                gt_x1=x1,
                pred_keypoints3d=pred_kp3d,
                gt_keypoints3d=gt_kp3d,
                data_mask_temporal=tgt_padding_mask,
                global_step=self.get_global_step(),
                generation_mask=generation_mask,
            )

            aux_losses = self._compute_kimodo_aux_loss(
                pred, x1, timesteps, tgt_padding_mask
            )
            if aux_losses:
                losses.update(aux_losses)
        else:
            raise ValueError(f'Unsupported pred_type: {self.bundle.pred_type}')
        return losses

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self._prepare_and_forward(batch)
        losses = self._compute_base_loss(ctx)
        loss = sum(losses.values())
        result = {'loss': loss}
        for k, v in losses.items():
            result[f'loss_{k}'] = v.detach()
        return result

    def _compute_fk_keypoints(
        self,
        pred_x1: Tensor,
        gt_x1: Tensor,
    ):
        """Compute FK to get 3D keypoints for FK loss."""
        body_model = self.bundle.body_model
        if body_model is None:
            return None, None

        device = pred_x1.device
        std = torch.where(self.bundle.std < 1e-3, torch.zeros_like(self.bundle.std), self.bundle.std)

        def _fk(x_norm):
            x = x_norm * std + self.bundle.mean
            B, L = x.shape[:2]
            transl = x[..., 0:3]
            root_rot6d = x[..., 3:9].reshape(B, L, 1, 6)
            body6d = x[..., 9:135].reshape(B, L, 21, 6)
            betas = torch.zeros(1, 16, device=device, dtype=x.dtype)
            kp_list = []
            for b in range(B):
                kp = body_model(
                    body6d[b].to(device),
                    betas,
                    root_rot6d[b].to(device),
                    transl[b].to(device),
                )
                kp_list.append(kp)
            return torch.stack(kp_list, dim=0)

        pred_kp3d = _fk(pred_x1)
        gt_kp3d = _fk(gt_x1)
        return pred_kp3d, gt_kp3d

    def _compute_kimodo_aux_loss(
        self,
        pred_x1_norm: Optional[Tensor],
        gt_x1_norm: Optional[Tensor],
        timesteps: Optional[Tensor],
        tgt_padding_mask: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        """Compute KIMODO-style auxiliary losses."""
        aux = getattr(self.bundle, 'kimodo_aux_loss', None)
        if aux is None or not aux.enabled:
            return {}
        if pred_x1_norm is None or gt_x1_norm is None:
            return {}
        if self.bundle.mean.shape[0] < 198:
            return {}
        try:
            bone_offsets = self.bundle.get_bone_offsets()
        except Exception:
            return {}
        rotation_space = getattr(self.bundle, 'rotation_space', 'local')
        return aux(
            pred_x1_norm=pred_x1_norm,
            gt_x1_norm=gt_x1_norm,
            mean=self.bundle.mean,
            std=self.bundle.std,
            bone_offsets=bone_offsets,
            rotation_space=rotation_space,
            data_mask_temporal=tgt_padding_mask,
            timesteps=timesteps,
            global_step=self.get_global_step(),
        )

    def _compute_fk_consistency_loss(
        self,
        pred_x1_norm: Tensor,
        timesteps: Tensor,
        data_mask_temporal: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Compute FK consistency loss for 198-dim models."""
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            motion198_fk_loss,
        )
        bone_offsets = self.bundle.get_bone_offsets()
        rotation_space = getattr(self.bundle, 'rotation_space', 'local')
        return motion198_fk_loss(
            pred_x1_norm,
            self.bundle.mean,
            self.bundle.std,
            bone_offsets,
            rotation_space=rotation_space,
            timesteps=timesteps,
            data_mask_temporal=data_mask_temporal,
        )
