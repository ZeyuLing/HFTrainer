"""SD1.5 Text-to-Image ModelBundle."""

import os
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class SD15Bundle(ModelBundle):
    """
    ModelBundle for Stable Diffusion 1.5 text-to-image.

    Sub-modules:
      - text_encoder: CLIPTextModel (frozen by default)
      - vae: AutoencoderKL (frozen by default)
      - unet: UNet2DConditionModel (trainable)
      - scheduler: DDPMScheduler (no params)
      - tokenizer: CLIPTokenizer (not nn.Module)

    Atomic forward functions shared by Trainer and Pipeline:
      - encode_text(prompts) -> encoder_hidden_states
      - encode_image(images) -> latents
      - decode_latent(latents) -> images
      - predict_noise(noisy_latents, timesteps, encoder_hidden_states) -> noise_pred
    """

    def __init__(
        self,
        text_encoder: dict,
        vae: dict,
        unet: dict,
        scheduler: dict,
        tokenizer_path: Optional[str] = None,
        max_token_length: int = 77,
    ):
        super().__init__()
        self.max_token_length = max_token_length

        # Build all sub-modules
        self._build_modules({
            'text_encoder': text_encoder,
            'vae': vae,
            'unet': unet,
            'scheduler': scheduler,
        })

        # Load tokenizer (not an nn.Module — stored as plain attribute)
        pretrained_path = tokenizer_path
        if pretrained_path is None:
            # Try to get from text_encoder config
            te_cfg = text_encoder
            if isinstance(te_cfg, dict):
                fp = te_cfg.get('from_pretrained', {})
                pretrained_path = fp.get('pretrained_model_name_or_path') if fp else None

        if pretrained_path is not None:
            from transformers import CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(
                pretrained_path, subfolder='tokenizer'
            )
        else:
            self.tokenizer = None

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        text_encoder_type: str = 'CLIPTextModel',
        vae_type: str = 'AutoencoderKL',
        unet_type: str = 'UNet2DConditionModel',
        scheduler_type: str = 'DDPMScheduler',
        text_encoder_overrides: Optional[Dict[str, Any]] = None,
        vae_overrides: Optional[Dict[str, Any]] = None,
        unet_overrides: Optional[Dict[str, Any]] = None,
        scheduler_overrides: Optional[Dict[str, Any]] = None,
        tokenizer_path: Optional[str] = None,
        max_token_length: int = 77,
        shared_pretrained_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        shared_pretrained_kwargs = shared_pretrained_kwargs or {}

        def build_component(component_type: str, subfolder: str, overrides: Optional[Dict[str, Any]]):
            component_cfg = {
                'type': component_type,
                'from_pretrained': {
                    'pretrained_model_name_or_path': pretrained_model_name_or_path,
                    'subfolder': subfolder,
                },
            }
            cls._merge_nested_dict(component_cfg['from_pretrained'], shared_pretrained_kwargs)
            cls._merge_nested_dict(component_cfg, overrides)
            return component_cfg

        return {
            'text_encoder': build_component(text_encoder_type, 'text_encoder', text_encoder_overrides),
            'vae': build_component(vae_type, 'vae', vae_overrides),
            'unet': build_component(unet_type, 'unet', unet_overrides),
            'scheduler': build_component(scheduler_type, 'scheduler', scheduler_overrides),
            'tokenizer_path': tokenizer_path or pretrained_model_name_or_path,
            'max_token_length': max_token_length,
        }

    def save_pretrained(
        self,
        save_directory: str,
        merge_lora: bool = True,
        safe_serialization: bool = True,
        **kwargs,
    ):
        from diffusers import StableDiffusionPipeline
        from hftrainer.utils.hf_export import safe_hf_export

        os.makedirs(save_directory, exist_ok=True)
        if merge_lora:
            self.merge_lora_weights([name for name in ('text_encoder', 'unet') if self.is_lora_module(name)])

        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        with safe_hf_export():
            pipeline.save_pretrained(
                save_directory,
                safe_serialization=safe_serialization,
                **kwargs,
            )

    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts to CLIP embeddings.

        Args:
            prompts: list of text strings

        Returns:
            encoder_hidden_states: Tensor[B, seq_len, hidden_dim]
        """
        assert self.tokenizer is not None, "Tokenizer not initialized"

        tokens = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.max_token_length,
            truncation=True,
            return_tensors='pt',
        )
        input_ids = tokens.input_ids.to(self.text_encoder.device)

        with torch.set_grad_enabled(self.text_encoder.training):
            outputs = self.text_encoder(input_ids=input_ids)
        return outputs.last_hidden_state

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to VAE latents.

        Args:
            images: Tensor[B, 3, H, W] in [-1, 1]

        Returns:
            latents: Tensor[B, 4, H/8, W/8]
        """
        vae_dtype = next(self.vae.parameters()).dtype
        images = images.to(dtype=vae_dtype)
        with torch.set_grad_enabled(self.vae.training):
            latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            latents: Tensor[B, 4, H/8, W/8]

        Returns:
            images: Tensor[B, 3, H, W] in [-1, 1]
        """
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)
        latents = latents / self.vae.config.scaling_factor
        with torch.set_grad_enabled(self.vae.training):
            images = self.vae.decode(latents).sample
        return images

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise with the UNet.

        Args:
            noisy_latents: Tensor[B, 4, H/8, W/8]
            timesteps: Tensor[B]
            encoder_hidden_states: Tensor[B, seq_len, hidden_dim]

        Returns:
            noise_pred: Tensor[B, 4, H/8, W/8]
        """
        unet_dtype = next(self.unet.parameters()).dtype
        noisy_latents = noisy_latents.to(dtype=unet_dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=unet_dtype)
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        return noise_pred

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to latents using the scheduler."""
        return self.scheduler.add_noise(latents, noise, timesteps)
