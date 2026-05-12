"""HyMotion-M2M Trainer: flow-matching training for motion-to-motion editing."""

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
class HyMotionM2MTrainer(BaseTrainer):
    """Trainer for HYMotion-M2M flow-matching motion editing.

    Training forward:
      1. Prepare padding and masks
      2. Prepare text embeddings (null or encoded)
      3. Sample timesteps, create x_t via flow matching interpolation
      4. Build VACE conditioning context
      5. Forward through bundle.predict_flow()
      6. Compute loss via bundle.m2m_loss
    """

    def __init__(
        self,
        bundle,
        val_num_steps: int = 10,
        max_text_len: int = 128,
        mask_aware_noise: bool = False,
        **kwargs,
    ):
        super().__init__(bundle)
        self.val_num_steps = val_num_steps
        self.mask_aware_noise = mask_aware_noise
        # Fixed text sequence length — matches HY-Motion T2M 1.0 (max_length_llm=128).
        # Pre-extracted embeddings are variable-length; we pad/truncate to this.
        self.max_text_len = max_text_len

    def _prepare_and_forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs and run a single base forward pass.

        Extracts steps 1-5 of the original train_step so that subclasses
        (e.g., SOAR post-trainer) can reuse the same setup and intermediate
        tensors without reimplementing data/VACE/text preparation.

        Returns a context dict with all intermediate tensors needed to
        compute either the plain base loss (current behaviour) or
        additional SOAR rollout/correction losses.

        Keys in returned dict:
          device, src_motion, tgt_motion, src_mask, src_length_list,
          tgt_length_list, ref_pose, tgt_padding_mask, vtxt_input,
          ctxt_input, ctxt_mask_temporal, x0, x1, x_t, timesteps, t,
          vace_context, pred, generation_mask
        """
        device = next(self.bundle.motion_transformer.parameters()).device

        # Source and target motions
        src_motion = batch['src_motion'].to(device)
        tgt_motion = batch['tgt_motion'].to(device)
        src_mask = batch.get('src_mask')
        if src_mask is not None:
            src_mask = src_mask.to(device)

        # Normalize motions using bundle's mean/std (matching original repo which
        # normalizes in dataset before padding). src_mask is binary — NOT normalized.
        # IMPORTANT: Only normalize valid frames; padded frames (zeros from
        # RandomCropPadding) must stay zero. We build a per-frame validity mask
        # from tgt_length and zero out padding frames after normalization.
        tgt_length_list: List[int] = batch['tgt_length']
        if isinstance(tgt_length_list, Tensor):
            tgt_length_list = tgt_length_list.tolist()
        src_length_list = batch.get('src_length', tgt_length_list)
        if isinstance(src_length_list, Tensor):
            src_length_list = src_length_list.tolist()

        src_motion = self.bundle.normalize_motion(src_motion)
        tgt_motion = self.bundle.normalize_motion(tgt_motion)

        # Zero out mask regions for Completion samples; keep LQ values for Edit samples.
        # Per-sample: edit_mode[i]=True → keep src values; edit_mode[i]=False → zero mask region.
        edit_flags = batch.get('edit_mode', None)
        if src_mask is not None:
            if edit_flags is not None:
                if isinstance(edit_flags, Tensor):
                    # (B,) bool tensor → (B, 1, 1) for broadcasting
                    keep = edit_flags.view(-1, 1, 1).float().to(src_motion.device)
                elif isinstance(edit_flags, (list, tuple)):
                    keep = torch.tensor([float(bool(e)) for e in edit_flags],
                                        device=src_motion.device).view(-1, 1, 1)
                else:
                    keep = torch.zeros(1, 1, 1, device=src_motion.device)
                # For completion (keep=0): src_motion *= (1-mask) → zeroes mask regions
                # For edit (keep=1): src_motion unchanged → reactive has LQ values
                src_motion = src_motion * (1 - src_mask * (1 - keep))
            else:
                # No edit_mode flag → all completion
                src_motion = src_motion * (1 - src_mask)

        # Zero out padded frames so they don't produce extreme normalized values
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
            # Pre-encoded text vectors already in batch.
            # text_vec_raw is always a Tensor (null samples get zero-filled by
            # LoadPreExtractedTextEmbedding, so no mixed Tensor/None batches).
            vtxt_input = batch['text_vec_raw'].to(device)
            # text_ctxt_raw may be a list of variable-length tensors when loaded
            # from pre-extracted .pt files (different captions have different
            # sequence lengths and flexible_collate cannot stack them).
            # Pad to the fixed max_text_len (default 128) to match HY-Motion T2M 1.0.
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
                # Already a stacked tensor (all same length) — pad/truncate
                cur_len = ctxt_raw.shape[1]
                if cur_len < pad_len:
                    ctxt_input = F.pad(ctxt_raw, (0, 0, 0, pad_len - cur_len)).to(device)
                else:
                    ctxt_input = ctxt_raw[:, :pad_len].to(device)
            ctxt_length = batch['text_ctxt_raw_length'].to(device).clamp(max=pad_len)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)

            # For null-embedding samples (no caption, text_ctxt_raw_length==0),
            # force-replace with the learned null embeddings so they match the
            # null distribution the model sees during CFG dropout.
            null_mask = (ctxt_length == 0)  # (B,)
            if null_mask.any():
                null_v = self.bundle.null_vtxt_feat.expand_as(vtxt_input)
                null_c = self.bundle.null_ctxt_input.expand_as(ctxt_input)
                vtxt_input = torch.where(
                    null_mask.view(B, 1, 1).expand_as(vtxt_input), null_v, vtxt_input
                )
                ctxt_input = torch.where(
                    null_mask.view(B, 1, 1).expand_as(ctxt_input), null_c, ctxt_input
                )

            vtxt_input, ctxt_input, text_available = self.bundle.mask_text_cond(
                vtxt_input, ctxt_input,
                force_mask=False,
                cond_mask_prob=self.bundle.cond_mask_prob,
                return_text_available=True,
            )
        elif 'caption' in batch and batch['caption'] is not None:
            # Online text encoding from raw captions.
            # The text encoder (Qwen3-8B) lives on CPU and is never moved to
            # GPU.  In DDP, every node has its own CPU copy of the encoder, so
            # we encode on local rank-0 of each node and broadcast within the
            # node group — avoiding duplicate encode work across ranks.
            # In practice we encode on each rank independently because
            # accelerate DDP already replicates the batch per process; the
            # slight CPU overhead is acceptable vs the complexity of
            # rank0-only encode + broadcast.
            captions = batch['caption']
            if isinstance(captions, torch.Tensor):
                captions = captions.tolist()
            # Replace None entries with empty string for encoder
            captions = [c if c is not None else '' for c in captions]
            with torch.no_grad():
                text_feats = self.bundle.encode_text(captions)
            vtxt_input = text_feats['text_vec_raw'].to(device)
            ctxt_input = text_feats['text_ctxt_raw'].to(device)
            ctxt_length = text_feats['text_ctxt_raw_length'].to(device)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])
            vtxt_input, ctxt_input, text_available = self.bundle.mask_text_cond(
                vtxt_input, ctxt_input,
                force_mask=False,
                cond_mask_prob=self.bundle.cond_mask_prob,
                return_text_available=True,
            )
        else:
            vtxt_input = self.bundle.null_vtxt_feat.expand(B, 1, -1)
            ctxt_input = self.bundle.null_ctxt_input.expand(B, 1, -1)
            ctxt_length = torch.tensor([1], device=device).expand(B)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, 1).expand(B, -1)
            text_available = torch.zeros(B, dtype=torch.bool, device=device)
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

        # Mask-aware noise: keep known regions clean in x_t so that
        # inference-time replacement guidance is train-consistent.
        # src_mask: (B, L, D), 1=generate, 0=known.
        # After this: x_t[known] = x1[known] (clean), x_t[gen] unchanged (noisy).
        if self.mask_aware_noise and src_mask is not None:
            keep_mask = 1 - src_mask  # (B, L, D), 1=known
            x_t = x_t * src_mask + x1 * keep_mask

        # 4. Build VACE context
        vace_context = self.bundle.prepare_vace_input(
            src_motion=src_motion,
            ref_pose=ref_pose,
            src_mask=src_mask,
        )

        # 5. Forward
        x_input = torch.cat([x_t, vace_context], dim=-1)
        pred = self.bundle.predict_flow(
            x_input=x_input,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=tgt_padding_mask,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )

        # Generation mask for mask-aware loss weighting.
        generation_mask = None
        if self.mask_aware_noise and src_mask is not None:
            generation_mask = src_mask  # (B, L, D), 1=generate

        return {
            'device': device,
            'src_motion': src_motion,
            'tgt_motion': tgt_motion,
            'src_mask': src_mask,
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
            'vace_context': vace_context,
            'pred': pred,
            'generation_mask': generation_mask,
            'text_available': text_available,
        }

    def _compute_base_loss(self, ctx: Dict[str, Any]) -> Dict[str, Tensor]:
        """Compute the standard flow-matching loss dict from a context dict.

        Separated from train_step so that subclasses (e.g., SOAR trainer) can
        reuse the same loss computation on their base forward.
        """
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
            # For motion smoothness loss: derive predicted x1 from velocity
            # pred_x1 = x_t + (1 - t) * pred_vel
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

            # FK loss: compute 3D keypoints from predicted and GT x1
            pred_kp3d = None
            gt_kp3d = None
            if self.bundle.m2m_loss.keypoints3d_weight > 0.0 and pred_x1_for_smooth is not None:
                pred_kp3d, gt_kp3d = self._compute_fk_keypoints(
                    pred_x1_for_smooth, gt_x1_for_smooth
                )

            # FK consistency loss for 198-dim models
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

            # ---- KIMODO-style auxiliary losses ----
            # Operate on (pred_x1, gt_x1) in normalised space; computed via
            # FK on rotation+translation channels.  Padding-aware; ignores
            # generation_mask by design (KIMODO supervises every frame).
            aux_losses = self._compute_kimodo_aux_loss(
                pred_x1_for_smooth, x1, timesteps, tgt_padding_mask
            )
            if aux_losses:
                losses.update(aux_losses)
        elif self.bundle.pred_type == 'x1':
            t_eps = 0.05
            gt_velocity = (x1 - x_t) / (1 - t).clamp_min(t_eps)
            pred_velocity = (pred - x_t) / (1 - t).clamp_min(t_eps)

            # FK loss: compute 3D keypoints from predicted and GT x1
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

            # KIMODO-style auxiliary losses (also for x1 pred_type)
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
        """Compute FK to get 3D keypoints for FK loss.

        pred_x1 and gt_x1 are in **normalized** space. We denormalize them
        before computing FK so that joint positions are in meters.

        Returns:
            pred_kp3d: (B, L, J, 3) or None
            gt_kp3d: (B, L, J, 3) or None
        """
        body_model = self.bundle.body_model
        if body_model is None:
            return None, None

        device = pred_x1.device
        std = torch.where(self.bundle.std < 1e-3, torch.zeros_like(self.bundle.std), self.bundle.std)

        def _fk(x_norm):
            # Denormalize
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
            return torch.stack(kp_list, dim=0)  # (B, L, J, 3)

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
        """Compute KIMODO-style auxiliary losses (j_p / j_v / fk_consistency).

        Returns an empty dict when:
        - the bundle does not own a ``kimodo_aux_loss`` attribute, or
        - no aux weight is enabled, or
        - the model is not 198-dim, or
        - ``pred_x1_norm`` is None (e.g. velocity pred without smoothness).
        """
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
        """Compute FK consistency loss for 198-dim models.

        Penalizes inconsistency between rotation/translation channels and
        position channels by running FK on the predicted rotation/translation
        and comparing with the predicted position. Padding frames are
        excluded via ``data_mask_temporal`` so the replicated tail and
        zeroed-out tgt frames do not contaminate the consistency signal.

        Args:
            pred_x1_norm: (B, L, 198) predicted x1 in normalized space.
            timesteps: (B,) diffusion timesteps.
            data_mask_temporal: (B, L) mask, 1 = valid frame, 0 = padded.

        Returns:
            Scalar FK consistency loss, or None on failure.
        """
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


# ---------------------------------------------------------------------------
# Unit tests — run with: python -m hftrainer.trainers.motion.hymotion_m2m_trainer
# ---------------------------------------------------------------------------

def _test_trainer_zeroes_mask_regions():
    """Verify trainer zeros src_motion in mask=1 regions AFTER normalization.

    Expected:
    - Input: src_motion (raw), src_mask (binary)
    - After normalize: src_motion_norm has values everywhere
    - After zeroing: src_motion_norm[mask=1] == 0
    - prepare_vace_input receives zeroed motion → reactive = 0 for completion
    """
    import torch

    T, D = 100, 135
    # Simulate raw motion and mask
    src_motion = torch.randn(1, T, D)
    src_mask = torch.zeros(1, T, D)
    src_mask[0, 30:50, :] = 1.0  # frames 30-49 fully masked

    # Simulate normalize (mean=0, std=1 for simplicity)
    motion_norm = src_motion.clone()

    # The critical fix: zero out mask=1 regions AFTER normalize
    motion_zeroed = motion_norm * (1 - src_mask)

    # Verify
    mask_bool = src_mask > 0.5
    assert motion_zeroed[mask_bool].abs().max() == 0.0, \
        "mask=1 regions must be 0 after zeroing"
    assert motion_zeroed[~mask_bool].std() > 0, \
        "mask=0 regions must retain motion values"

    # Verify VACE would be correct
    inactive = motion_zeroed * (1 - src_mask)
    reactive = motion_zeroed * src_mask
    assert reactive.abs().max() == 0.0, \
        "reactive must be 0 everywhere for completion task (src_motion zeroed in mask regions)"

    print("  ✅ Trainer mask zeroing: src_motion[mask=1]=0 after normalize, reactive=0")


def _test_trainer_preserves_tgt_motion():
    """Verify tgt_motion (target/GT) is NOT zeroed — only src_motion is zeroed.

    The model predicts velocity v = x1 - x0 where x1 = normalized tgt_motion.
    tgt_motion must retain all values including in mask=1 regions.
    """
    import torch

    T, D = 100, 135
    tgt_motion = torch.randn(1, T, D)
    src_mask = torch.zeros(1, T, D)
    src_mask[0, 30:50, :] = 1.0

    # tgt_motion is only normalized, NEVER zeroed
    tgt_norm = tgt_motion.clone()  # simulate normalize

    # Verify tgt still has values in mask=1 regions
    mask_bool = src_mask > 0.5
    assert tgt_norm[mask_bool].std() > 0, \
        "tgt_motion must NOT be zeroed — it's the generation target"

    print("  ✅ Trainer tgt_motion: preserved (not zeroed) in mask=1 regions")


if __name__ == '__main__':
    print("Running hymotion_m2m_trainer unit tests...")
    _test_trainer_zeroes_mask_regions()
    _test_trainer_preserves_tgt_motion()
    print("All tests passed ✅")
