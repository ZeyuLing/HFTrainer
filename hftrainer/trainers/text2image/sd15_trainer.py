"""SD1.5 text-to-image trainer."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from hftrainer.trainers.base_trainer import BaseTrainer
from hftrainer.registry import TRAINERS


@TRAINERS.register_module()
class SD15Trainer(BaseTrainer):
    """
    Trainer for Stable Diffusion 1.5 text-to-image.

    train_step:
      1. Encode text prompts → encoder_hidden_states
      2. Encode images to latents → encode_image
      3. Sample noise → add noise (forward diffusion)
      4. Predict noise with UNet
      5. Compute MSE loss between predicted and actual noise

    val_step:
      1. Generate sample images via denoising loop
      2. Return {'preds', 'prompts'}
    """

    def __init__(
        self,
        bundle,
        prediction_type: str = 'epsilon',  # 'epsilon' or 'v_prediction'
        snr_gamma: Optional[float] = None,
        num_val_inference_steps: int = 20,
        val_prompts: Optional[list] = None,
        guidance_scale: float = 7.5,
        **kwargs,
    ):
        super().__init__(bundle)
        self.prediction_type = prediction_type
        self.snr_gamma = snr_gamma
        self.num_val_inference_steps = num_val_inference_steps
        self.val_prompts = val_prompts or ['a photo of a cat', 'a beautiful sunset']
        self.guidance_scale = guidance_scale

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch: dict with 'pixel_values' (Tensor[B,3,H,W] in [-1,1]) and 'text' (list[str])

        Returns:
            {'loss': Tensor, 'loss_mse': Tensor}
        """
        pixel_values = batch['pixel_values']
        texts = batch['text']
        bsz = pixel_values.shape[0]

        # Encode text
        encoder_hidden_states = self.bundle.encode_text(texts)

        # Encode image to latents
        with torch.no_grad():
            latents = self.bundle.encode_image(pixel_values)

        # Sample noise
        noise = torch.randn_like(latents)

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.bundle.scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        ).long()

        # Add noise to latents (forward diffusion)
        noisy_latents = self.bundle.add_noise(latents, noise, timesteps)

        # Predict noise (or v-prediction)
        noise_pred = self.bundle.predict_noise(noisy_latents, timesteps, encoder_hidden_states)

        # Compute target
        if self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'v_prediction':
            target = self.bundle.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # MSE loss
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction='mean')

        return {'loss': loss, 'loss_mse': loss.detach()}

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate sample images for visualization.

        Returns:
            {
                'preds': Tensor[B, 3, H, W] in [0, 1]  (generated images)
                'prompts': list[str]
            }
        """
        from diffusers import DDIMScheduler
        import copy

        prompts = self.val_prompts[:min(4, len(self.val_prompts))]

        # Use DDIM scheduler for fast inference
        scheduler = copy.deepcopy(self.bundle.scheduler)
        if not hasattr(scheduler, 'set_timesteps'):
            # Fallback: use the bundle scheduler
            pass
        scheduler.set_timesteps(self.num_val_inference_steps)

        device = next(self.bundle.unet.parameters()).device
        dtype = next(self.bundle.unet.parameters()).dtype

        # Encode prompts
        encoder_hidden_states = self.bundle.encode_text(prompts)

        # Start with random latents
        latents = torch.randn(
            len(prompts), 4, 64, 64,
            device=device, dtype=dtype,
        )
        latents = latents * scheduler.init_noise_sigma

        # Denoising loop
        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.bundle.predict_noise(
                    latents, t.expand(len(prompts)).to(device), encoder_hidden_states
                )
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to images
        images = self.bundle.decode_latent(latents)

        # Convert from [-1, 1] to [0, 1]
        images = (images / 2 + 0.5).clamp(0, 1)

        return {'preds': images.cpu(), 'prompts': prompts}
