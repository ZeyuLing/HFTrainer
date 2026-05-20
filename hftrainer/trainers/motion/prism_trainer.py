"""PRISM trainer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from hftrainer.registry import TRAINERS
from hftrainer.trainers.base_trainer import BaseTrainer


@TRAINERS.register_module()
class PrismTrainer(BaseTrainer):
    """Trainer for PRISM flow-matching motion generation."""

    def __init__(
        self,
        bundle,
        condition_num_frames: Union[int, List[int]] = 1,
        frame_condition_rate: float = 0.1,
        prompt_drop_rate: float = 0.1,
        max_text_length: int = 128,
        val_prompts: Optional[List[str]] = None,
        num_val_inference_steps: int = 10,
        guidance_scale: float = 5.0,
        translation_loss_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(bundle)
        self.condition_num_frames = condition_num_frames
        self.frame_condition_rate = frame_condition_rate
        self.prompt_drop_rate = prompt_drop_rate
        self.max_text_length = max_text_length
        self.val_prompts = val_prompts or ['a person walking forward']
        self.num_val_inference_steps = num_val_inference_steps
        self.guidance_scale = guidance_scale
        self.translation_loss_weight = translation_loss_weight

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        motion = batch['motion']
        captions = batch['caption']
        num_frames = batch.get('num_frames')

        latents = self.bundle.encode_motion(motion)
        batch_size, _, latent_frames, latent_joints = latents.shape

        padding_mask = self.bundle.create_padding_mask(
            num_frames=num_frames,
            batch_size=batch_size,
            latent_frames=latent_frames,
            latent_joints=latent_joints,
            device=latents.device,
        )
        text_states, text_mask = self.bundle.encode_prompt_with_mask(
            captions,
            max_sequence_length=self.max_text_length,
            prompt_drop_rate=self.prompt_drop_rate,
            dtype=next(self.bundle.transformer.parameters()).dtype,
        )
        condition_frame_mask_vae = self.bundle.create_condition_mask(
            latents,
            frame_condition_rate=self.frame_condition_rate,
            condition_num_frames=self.condition_num_frames,
        )

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
        transformer_dtype = next(self.bundle.transformer.parameters()).dtype
        noisy_latents = noisy_latents.to(dtype=transformer_dtype)

        model_pred = self.bundle.transformer(
            hidden_states=noisy_latents,
            encoder_hidden_states=text_states,
            timestep=timesteps,
            hidden_states_mask=padding_mask if num_frames is not None else None,
            encoder_hidden_states_mask=text_mask,
        ).float()

        mse = F.mse_loss(model_pred, targets.float(), reduction='none')
        # mse shape: [B, C, T', J] where J=23 (token 0=translation, 1-22=rotation)
        condition_mask = condition_frame_mask_vae.expand_as(mse).float()
        padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
        full_mask = condition_mask * padding_mask

        # Separate translation (J=0) and rotation (J=1:) to prevent
        # translation loss dilution (1/23 ≈ 4.3% vs 22/23 ≈ 95.7%).
        mse_transl = mse[:, :, :, :1]           # [B, C, T', 1]
        mask_transl = full_mask[:, :, :, :1]
        loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)

        mse_rot = mse[:, :, :, 1:]              # [B, C, T', 22]
        mask_rot = full_mask[:, :, :, 1:]
        loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)

        w_t = self.translation_loss_weight
        loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
        return {
            'loss': loss,
            'loss_flow': loss.detach(),
            'loss_transl': loss_transl.detach(),
            'loss_rot': loss_rot.detach(),
        }

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

        pipeline = PrismPipeline(self.bundle)
        preds = pipeline(
            prompts=self.val_prompts[0],
            num_frames_per_segment=33,
            num_inference_steps=self.num_val_inference_steps,
            guidance_scale=self.guidance_scale,
        )
        return {'preds': preds, 'prompts': self.val_prompts}
