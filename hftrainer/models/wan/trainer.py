"""WAN text-to-video trainer."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List

from hftrainer.trainers.base_trainer import BaseTrainer
from hftrainer.registry import TRAINERS


@TRAINERS.register_module()
class WanTrainer(BaseTrainer):
    """
    Trainer for WAN text-to-video using flow matching.

    WAN uses FlowMatchEulerDiscreteScheduler with flow-matching objective:
      - Adds noise using linear interpolation (not DDPM)
      - Loss = MSE between predicted velocity and actual velocity

    train_step:
      1. Encode text → text embeddings
      2. Encode video → latents
      3. Sample noise and timesteps → noisy latents (flow matching)
      4. Predict noise with transformer
      5. Compute MSE loss

    val_step:
      1. Generate sample video (short, for smoke test)
    """

    def __init__(
        self,
        bundle,
        prediction_type: str = 'flow_matching',
        snr_gamma: Optional[float] = None,
        num_val_inference_steps: int = 5,
        val_prompts: Optional[List[str]] = None,
        val_num_frames: int = 8,
        val_height: int = 64,
        val_width: int = 64,
        **kwargs,
    ):
        super().__init__(bundle)
        self.prediction_type = prediction_type
        self.snr_gamma = snr_gamma
        self.num_val_inference_steps = num_val_inference_steps
        self.val_prompts = val_prompts or ['a cat walking', 'ocean waves']
        self.val_num_frames = val_num_frames
        self.val_height = val_height
        self.val_width = val_width

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch: dict with:
              - 'video': Tensor[B, T, C, H, W] or [B, C, T, H, W] in [-1, 1]
              - 'text': list[str]

        Returns:
            {'loss': Tensor, 'loss_flow': Tensor}
        """
        videos = batch['video']   # [B, T, C, H, W] or [B, C, T, H, W]
        texts = batch['text']
        bsz = videos.shape[0]

        # Ensure BCTHW format
        if videos.ndim == 5:
            if videos.shape[2] == 3:  # BTCHW
                videos = videos.permute(0, 2, 1, 3, 4)  # -> BCTHW

        device = videos.device

        # Encode text
        text_embeds = self.bundle.encode_text(texts)

        # Encode video to latents
        with torch.no_grad():
            latents = self.bundle.encode_video(videos)

        # Flow matching: add noise via linear interpolation
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.bundle.scheduler.config.num_train_timesteps,
            (bsz,),
            device=device,
        ).float()

        # Flow matching: noisy_latents = t * noise + (1-t) * latents
        t = timesteps / self.bundle.scheduler.config.num_train_timesteps
        # Reshape t for broadcasting: [B, 1, 1, 1, 1]
        t_broadcast = t.view(bsz, *([1] * (latents.ndim - 1)))
        noisy_latents = t_broadcast * noise + (1 - t_broadcast) * latents

        # Target velocity = noise - latents
        target = noise - latents

        # Predict velocity
        noise_pred = self.bundle.predict_noise(
            noisy_latents,
            timesteps.long(),
            text_embeds,
        )

        # MSE loss
        loss = F.mse_loss(noise_pred.float(), target.float())

        return {'loss': loss, 'loss_flow': loss.detach()}

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a short sample video.

        Returns:
            {
                'preds': Tensor[B, T, C, H, W] in [0, 1]
                'prompts': list[str]
            }
        """
        import copy

        prompts = self.val_prompts[:min(2, len(self.val_prompts))]

        scheduler = copy.deepcopy(self.bundle.scheduler)
        scheduler.set_timesteps(self.num_val_inference_steps)

        device = next(self.bundle.transformer.parameters()).device
        dtype = next(self.bundle.transformer.parameters()).dtype

        # Encode text
        text_embeds = self.bundle.encode_text(prompts)

        # Initialize random latents
        # WAN VAE compresses: T/4 frames, H/8 spatial
        lat_t = max(1, self.val_num_frames // 4)
        lat_h = self.val_height // 8
        lat_w = self.val_width // 8
        in_channels = self.bundle.transformer.config.in_channels

        latents = torch.randn(
            len(prompts), in_channels, lat_t, lat_h, lat_w,
            device=device, dtype=dtype,
        )

        # Denoising loop (simplified for flow matching)
        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.bundle.predict_noise(
                    latents,
                    t.expand(len(prompts)).to(device),
                    text_embeds,
                )
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        videos = self.bundle.decode_latent(latents)  # [B, C, T, H, W]

        # Convert to [0, 1] and rearrange to [B, T, C, H, W]
        videos = (videos / 2 + 0.5).clamp(0, 1)
        videos = videos.permute(0, 2, 1, 3, 4)  # BCTHW -> BTCHW

        return {'preds': videos.cpu(), 'prompts': prompts}
