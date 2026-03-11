"""Distribution Matching Distillation bundle."""

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class DMDBundle(ModelBundle):
    """
    Bundle for DMD-style one-step diffusion distillation.

    Modules:
      - text_encoder: frozen text encoder
      - vae: frozen VAE
      - real_score_unet: frozen teacher score network
      - fake_score_unet: trainable fake score network
      - generator_unet: trainable one-step generator UNet
      - scheduler: diffusion scheduler
    """

    def __init__(
        self,
        text_encoder: dict,
        vae: dict,
        real_score_unet: dict,
        fake_score_unet: dict,
        generator_unet: dict,
        scheduler: dict,
        tokenizer_path: Optional[str] = None,
        max_token_length: int = 77,
        image_size: int = 512,
        conditioning_timestep: int = 999,
        dm_min_timestep_percent: float = 0.02,
        dm_max_timestep_percent: float = 0.98,
        generator_guidance_scale: float = 1.0,
        real_score_guidance_scale: float = 7.5,
        fake_score_guidance_scale: float = 1.0,
        regression_guidance_scale: float = 7.5,
    ):
        super().__init__()
        self.max_token_length = max_token_length
        self.image_size = image_size
        self.conditioning_timestep = conditioning_timestep
        self.dm_min_timestep_percent = dm_min_timestep_percent
        self.dm_max_timestep_percent = dm_max_timestep_percent
        self.generator_guidance_scale = generator_guidance_scale
        self.real_score_guidance_scale = real_score_guidance_scale
        self.fake_score_guidance_scale = fake_score_guidance_scale
        self.regression_guidance_scale = regression_guidance_scale

        self._build_modules({
            'text_encoder': text_encoder,
            'vae': vae,
            'real_score_unet': real_score_unet,
            'fake_score_unet': fake_score_unet,
            'generator_unet': generator_unet,
            'scheduler': scheduler,
        })

        pretrained_path = tokenizer_path
        if pretrained_path is None and isinstance(text_encoder, dict):
            fp = text_encoder.get('from_pretrained', {})
            pretrained_path = fp.get('pretrained_model_name_or_path') if fp else None

        if pretrained_path is not None:
            from transformers import CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(
                pretrained_path, subfolder='tokenizer'
            )
        else:
            self.tokenizer = None

        self.latent_channels = getattr(
            self.generator_unet.config, 'in_channels', 4
        )
        self.latent_size = image_size // 8

    def _module_device(self, module) -> torch.device:
        return next(module.parameters()).device

    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        assert self.tokenizer is not None, "Tokenizer not initialized"
        tokens = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.max_token_length,
            truncation=True,
            return_tensors='pt',
        )
        input_ids = tokens.input_ids.to(self._module_device(self.text_encoder))
        with torch.set_grad_enabled(self.text_encoder.training):
            outputs = self.text_encoder(input_ids=input_ids)
        return outputs.last_hidden_state

    def get_unconditional_text_embeddings(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        embeddings = self.encode_text([''] * batch_size)
        if device is not None:
            embeddings = embeddings.to(device)
        return embeddings

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.vae.training):
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.vae.config.scaling_factor
        with torch.set_grad_enabled(self.vae.training):
            images = self.vae.decode(latents).sample
        return images

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        return noisy_latents.to(dtype=latents.dtype)

    def sample_latent_noise(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = device or self._module_device(self.generator_unet)
        dtype = dtype or next(self.generator_unet.parameters()).dtype
        return torch.randn(
            batch_size,
            self.latent_channels,
            self.latent_size,
            self.latent_size,
            device=device,
            dtype=dtype,
        )

    def _predict_noise(
        self,
        unet,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        model_dtype = next(unet.parameters()).dtype
        noisy_latents = noisy_latents.to(dtype=model_dtype)
        cond_embeddings = cond_embeddings.to(dtype=model_dtype)
        if uncond_embeddings is not None:
            uncond_embeddings = uncond_embeddings.to(dtype=model_dtype)

        if guidance_scale == 1.0 or uncond_embeddings is None:
            return unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=cond_embeddings,
            ).sample

        model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
        model_timesteps = torch.cat([timesteps, timesteps], dim=0)
        hidden_states = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
        noise_pred = unet(
            model_input,
            model_timesteps,
            encoder_hidden_states=hidden_states,
        ).sample
        noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
        return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    def _predict_x0(
        self,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alphas = self.scheduler.alphas_cumprod.to(noisy_latents.device)[timesteps]
        alphas = alphas.view(-1, 1, 1, 1).to(noisy_latents.dtype)
        return (
            noisy_latents - torch.sqrt(1 - alphas) * noise_pred
        ) / torch.sqrt(alphas)

    def generate_latents(
        self,
        noise_latents: torch.Tensor,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        timestep: Optional[int] = None,
        return_noise: bool = False,
    ):
        guidance_scale = (
            self.generator_guidance_scale if guidance_scale is None else guidance_scale
        )
        timestep = (
            self.conditioning_timestep if timestep is None else timestep
        )
        timesteps = torch.full(
            (noise_latents.shape[0],),
            int(timestep),
            device=noise_latents.device,
            dtype=torch.long,
        )
        noise_pred = self._predict_noise(
            self.generator_unet,
            noise_latents,
            timesteps,
            cond_embeddings,
            uncond_embeddings=uncond_embeddings,
            guidance_scale=guidance_scale,
        )
        latents = self._predict_x0(noise_latents, noise_pred, timesteps)
        if return_noise:
            return latents, noise_pred
        return latents

    def _sample_dm_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        total_steps = int(self.scheduler.config.num_train_timesteps)
        min_step = int(total_steps * self.dm_min_timestep_percent)
        max_step = int(total_steps * self.dm_max_timestep_percent)
        max_step = max(min_step + 1, min(total_steps, max_step + 1))
        return torch.randint(min_step, max_step, (batch_size,), device=device).long()

    def compute_distribution_matching_loss(
        self,
        fake_latents: torch.Tensor,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        with torch.no_grad():
            latents = fake_latents.detach()
            noise = torch.randn_like(latents)
            timesteps = self._sample_dm_timesteps(latents.shape[0], latents.device)
            noisy_latents = self.add_noise(latents, noise, timesteps)

            pred_fake_noise = self._predict_noise(
                self.fake_score_unet,
                noisy_latents,
                timesteps,
                cond_embeddings,
                guidance_scale=self.fake_score_guidance_scale,
            )
            pred_fake_x0 = self._predict_x0(noisy_latents, pred_fake_noise, timesteps)

            pred_real_noise = self._predict_noise(
                self.real_score_unet,
                noisy_latents,
                timesteps,
                cond_embeddings,
                uncond_embeddings=uncond_embeddings,
                guidance_scale=self.real_score_guidance_scale,
            )
            pred_real_x0 = self._predict_x0(noisy_latents, pred_real_noise, timesteps)

            p_real = latents - pred_real_x0
            p_fake = latents - pred_fake_x0
            denom = p_real.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
            grad = torch.nan_to_num((p_real - p_fake) / denom)

        loss = 0.5 * F.mse_loss(
            fake_latents.float(),
            (fake_latents - grad).detach().float(),
        )
        log_dict = {
            'dm_timesteps': timesteps.detach(),
            'dm_grad_norm': grad.norm().detach(),
            'pred_real_x0': pred_real_x0.detach(),
            'pred_fake_x0': pred_fake_x0.detach(),
        }
        return loss, log_dict

    def compute_fake_score_loss(
        self,
        fake_latents: torch.Tensor,
        cond_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        latents = fake_latents.detach()
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()
        noisy_latents = self.add_noise(latents, noise, timesteps)
        pred_noise = self._predict_noise(
            self.fake_score_unet,
            noisy_latents,
            timesteps,
            cond_embeddings,
            guidance_scale=self.fake_score_guidance_scale,
        )
        loss = F.mse_loss(pred_noise.float(), noise.float())
        return loss, {
            'fake_score_timesteps': timesteps.detach(),
            'fake_score_pred': pred_noise.detach(),
        }

    def sample_teacher_deterministic(
        self,
        noise_latents: torch.Tensor,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        guidance_scale = (
            self.regression_guidance_scale
            if guidance_scale is None else guidance_scale
        )
        scheduler = copy.deepcopy(self.scheduler)
        try:
            scheduler.set_timesteps(num_inference_steps, device=noise_latents.device)
        except TypeError:
            scheduler.set_timesteps(num_inference_steps)
        sample = noise_latents * getattr(scheduler, 'init_noise_sigma', 1.0)

        with torch.no_grad():
            for t in scheduler.timesteps:
                timestep_value = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                batch_timesteps = torch.full(
                    (sample.shape[0],),
                    timestep_value,
                    device=sample.device,
                    dtype=torch.long,
                )
                noise_pred = self._predict_noise(
                    self.real_score_unet,
                    sample,
                    batch_timesteps,
                    cond_embeddings,
                    uncond_embeddings=uncond_embeddings,
                    guidance_scale=guidance_scale,
                )
                sample = scheduler.step(noise_pred, t, sample).prev_sample
        return sample
