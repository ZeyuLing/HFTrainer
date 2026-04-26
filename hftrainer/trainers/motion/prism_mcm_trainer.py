"""MCM (Multi-Condition Motion) trainer for audio-conditioned PRISM."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from hftrainer.registry import TRAINERS
from hftrainer.trainers.base_trainer import BaseTrainer


@TRAINERS.register_module()
class PrismMCMTrainer(BaseTrainer):
    """Trainer for MCM audio-conditioned flow-matching on PRISM.

    Same training flow as ``PrismTrainer`` with added audio conditioning:
    audio features (pre-computed or encoded on-the-fly) are passed to the
    control branch via ``bundle.predict_with_control()``.
    """

    def __init__(
        self,
        bundle,
        condition_num_frames: Union[int, List[int]] = 1,
        frame_condition_rate: float = 0.1,
        prompt_drop_rate: float = 0.1,
        audio_drop_rate: float = 0.1,
        max_text_length: int = 128,
        val_prompts: Optional[List[str]] = None,
        num_val_inference_steps: int = 10,
        guidance_scale: float = 5.0,
        **kwargs,
    ):
        super().__init__(bundle)
        self.condition_num_frames = condition_num_frames
        self.frame_condition_rate = frame_condition_rate
        self.prompt_drop_rate = prompt_drop_rate
        self.audio_drop_rate = audio_drop_rate
        self.max_text_length = max_text_length
        self.val_prompts = val_prompts or ['a person dances to music']
        self.num_val_inference_steps = num_val_inference_steps
        self.guidance_scale = guidance_scale

    @staticmethod
    def _pad_and_stack(tensors: list) -> torch.Tensor:
        """Right-pad variable-length tensors along dim 0 and stack.

        Each element is ``(T_i, ...)``; pad to ``max(T_i)`` by repeating
        the last frame (replicate padding).
        """
        max_t = max(t.shape[0] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[0] < max_t:
                repeat = max_t - t.shape[0]
                last_frame = t[-1:].expand(repeat, *t.shape[1:])
                t = torch.cat([t, last_frame], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0)

    def _get_audio_features(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract audio features from batch.

        Priority: pre-computed ``audio_features`` > raw ``audio`` waveform
        > raw ``music`` waveform.  Returns ``None`` when no audio is available.

        Handles both tensor and list-of-tensor inputs (from flexible_collate
        when waveform lengths differ or some samples lack audio).  With
        ``set_dummy_value=True`` in PackInputs, missing entries become ``None``
        in the collated list.
        """
        # Pre-computed features (smoke test path)
        if 'audio_features' in batch and batch['audio_features'] is not None:
            feat = batch['audio_features']
            if isinstance(feat, (list, tuple)):
                feat = [f for f in feat if f is not None]
                if not feat:
                    return None
                return self._pad_and_stack(feat)
            return feat

        if not hasattr(self.bundle, 'audio_encoder'):
            return None

        # Raw waveforms — try audio first, then music, merge per-sample
        audio_list = batch.get('audio')
        music_list = batch.get('music')

        # Normalise to lists for uniform handling
        if audio_list is not None and not isinstance(audio_list, (list, tuple)):
            audio_list = [audio_list]
        if music_list is not None and not isinstance(music_list, (list, tuple)):
            music_list = [music_list]

        # Merge: prefer audio, fallback to music per sample
        n = max(len(audio_list or []), len(music_list or []))
        if n == 0:
            return None

        merged: List[Optional[torch.Tensor]] = []
        for i in range(n):
            a = audio_list[i] if audio_list and i < len(audio_list) else None
            m = music_list[i] if music_list and i < len(music_list) else None
            merged.append(a if a is not None else m)

        # Filter valid waveforms
        valid = [w for w in merged if w is not None]
        if not valid:
            return None

        # Pad to same length and stack
        max_len = max(w.shape[-1] for w in valid)
        padded = []
        for w in valid:
            if w.ndim == 1:
                w = w.unsqueeze(0)  # (T_samples,) → (1, T_samples)
            if w.shape[-1] < max_len:
                w = F.pad(w, (0, max_len - w.shape[-1]))
            padded.append(w)
        waveform = torch.cat(padded, dim=0)  # (B_valid, T_samples)

        return self.bundle.encode_audio(waveform)

    def _apply_audio_dropout(
        self,
        audio_features: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Randomly zero-out audio features for classifier-free guidance."""
        if audio_features is None:
            return None
        if self.audio_drop_rate > 0 and self.training:
            drop_mask = (
                torch.rand(batch_size, 1, 1, device=device) < self.audio_drop_rate
            )
            audio_features = audio_features.masked_fill(drop_mask, 0.0)
        return audio_features

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        motion = batch['motion']
        # motion may be a list of tensors when pad_mode='none' produces
        # variable-length clips.  Stack with right-padding when needed.
        if isinstance(motion, (list, tuple)):
            motion = self._pad_and_stack(motion)
        batch_size = motion.shape[0]
        captions = batch.get('caption', [''] * batch_size)
        # Replace None captions with empty strings (from set_dummy_value=True)
        if isinstance(captions, (list, tuple)):
            captions = [c if c is not None else '' for c in captions]
        num_frames = batch.get('num_frames')

        # 1. Encode motion
        latents = self.bundle.encode_motion(motion)
        batch_size, _, latent_frames, latent_joints = latents.shape

        # 2. Create masks
        padding_mask = self.bundle.create_padding_mask(
            num_frames=num_frames,
            batch_size=batch_size,
            latent_frames=latent_frames,
            latent_joints=latent_joints,
            device=latents.device,
        )

        # 3. Encode text
        text_states = self.bundle.encode_prompt(
            captions,
            max_sequence_length=self.max_text_length,
            prompt_drop_rate=self.prompt_drop_rate,
            dtype=next(self.bundle.control_transformer.parameters()).dtype,
        )

        # 4. Condition mask
        condition_frame_mask_vae = self.bundle.create_condition_mask(
            latents,
            frame_condition_rate=self.frame_condition_rate,
            condition_num_frames=self.condition_num_frames,
        )

        # 5. Sample timesteps and add noise
        step_indices = torch.randint(
            0,
            len(self.bundle.scheduler.timesteps),
            (batch_size,),
            device=latents.device,
        )
        scheduler_timesteps = self.bundle.scheduler.timesteps.to(device=latents.device)
        timesteps = scheduler_timesteps[step_indices]

        noisy_latents, targets = self.bundle.add_flow_noise(latents, timesteps)
        noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
        timesteps = self.bundle.create_sequence_ts(
            timesteps,
            condition_frame_mask_vae,
            self.bundle.transformer.config.patch_size,
        )

        # 6. Audio features
        audio_features = self._get_audio_features(batch)
        audio_features = self._apply_audio_dropout(
            audio_features, batch_size, latents.device,
        )

        # 7. Cast to control transformer dtype
        ctrl_dtype = next(self.bundle.control_transformer.parameters()).dtype
        noisy_latents = noisy_latents.to(dtype=ctrl_dtype)

        # 8. Forward with control
        model_pred = self.bundle.predict_with_control(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            text_states=text_states,
            audio_features=audio_features,
            hidden_states_mask=padding_mask if num_frames is not None else None,
            encoder_hidden_states_mask=None,
        ).float()

        # 9. Loss
        # Safety net: replace NaN/Inf that may arise from numerical overflow
        # in the frozen main branch (e.g. fp16 on V100).  With bf16 this is
        # effectively a no-op but guards against edge cases.
        model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        model_pred = model_pred.clamp(-1e4, 1e4)
        mse = F.mse_loss(model_pred, targets.float(), reduction='none')
        condition_mask = condition_frame_mask_vae.expand_as(mse).float()
        padding_mask_expanded = padding_mask.unsqueeze(1).expand_as(mse).float()
        full_mask = condition_mask * padding_mask_expanded
        loss = (mse * full_mask).sum() / (full_mask.sum() + 1e-6)

        return {'loss': loss, 'loss_flow': loss.detach()}

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        from hftrainer.pipelines.motion.prism_mcm_pipeline import PrismMCMPipeline

        pipeline = PrismMCMPipeline(self.bundle)
        preds = pipeline(
            prompts=self.val_prompts[0],
            audio=None,
            num_frames_per_segment=33,
            num_inference_steps=self.num_val_inference_steps,
            guidance_scale=self.guidance_scale,
        )
        return {'preds': preds, 'prompts': self.val_prompts}
