"""WAN Text-to-Video ModelBundle."""

import os
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class WanBundle(ModelBundle):
    """
    ModelBundle for WAN (Wan2.1) text-to-video generation.

    Sub-modules:
      - text_encoder: UMT5EncoderModel (frozen by default)
      - vae: AutoencoderKLWan (frozen by default)
      - transformer: WanTransformer3DModel (trainable)
      - scheduler: FlowMatchEulerDiscreteScheduler (no params)
      - tokenizer: AutoTokenizer (not nn.Module)

    Atomic forward functions:
      - encode_text(prompts) -> encoder_hidden_states
      - encode_video(videos) -> latents
      - decode_latent(latents) -> videos
      - predict_noise(noisy_latents, t, encoder_hidden_states) -> noise_pred
    """

    def __init__(
        self,
        text_encoder: dict,
        vae: dict,
        transformer: dict,
        scheduler: dict,
        tokenizer_path: Optional[str] = None,
        max_token_length: int = 512,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.max_token_length = max_token_length

        transformer_cfg = dict(transformer)
        if gradient_checkpointing and 'gradient_checkpointing' not in transformer_cfg:
            transformer_cfg['gradient_checkpointing'] = True

        # Build sub-modules
        self._build_modules({
            'text_encoder': text_encoder,
            'vae': vae,
            'transformer': transformer_cfg,
            'scheduler': scheduler,
        })

        # Load tokenizer
        pretrained_path = tokenizer_path
        if pretrained_path is None:
            te_cfg = text_encoder if isinstance(text_encoder, dict) else {}
            fp = te_cfg.get('from_pretrained', {})
            pretrained_path = fp.get('pretrained_model_name_or_path') if fp else None

        if pretrained_path is not None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        else:
            self.tokenizer = None

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        text_encoder_type: str = 'UMT5EncoderModel',
        vae_type: str = 'AutoencoderKLWan',
        transformer_type: str = 'WanTransformer3DModel',
        scheduler_type: str = 'FlowMatchEulerDiscreteScheduler',
        text_encoder_overrides: Optional[Dict[str, Any]] = None,
        vae_overrides: Optional[Dict[str, Any]] = None,
        transformer_overrides: Optional[Dict[str, Any]] = None,
        scheduler_overrides: Optional[Dict[str, Any]] = None,
        tokenizer_path: Optional[str] = None,
        max_token_length: int = 512,
        gradient_checkpointing: bool = False,
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
            'transformer': build_component(transformer_type, 'transformer', transformer_overrides),
            'scheduler': build_component(scheduler_type, 'scheduler', scheduler_overrides),
            'tokenizer_path': tokenizer_path or pretrained_model_name_or_path,
            'max_token_length': max_token_length,
            'gradient_checkpointing': gradient_checkpointing,
        }

    def save_pretrained(
        self,
        save_directory: str,
        merge_lora: bool = True,
        safe_serialization: bool = True,
        **kwargs,
    ):
        from diffusers import WanPipeline
        from hftrainer.utils.hf_export import safe_hf_export

        os.makedirs(save_directory, exist_ok=True)
        if merge_lora and self.is_lora_module('transformer'):
            self.merge_lora_weights(['transformer'])

        pipeline = WanPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.scheduler,
            transformer=self.transformer,
        )
        with safe_hf_export():
            pipeline.save_pretrained(
                save_directory,
                safe_serialization=safe_serialization,
                **kwargs,
            )

    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts to T5 embeddings.

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
        attention_mask = tokens.attention_mask.to(self.text_encoder.device)

        with torch.set_grad_enabled(self.text_encoder.training):
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.last_hidden_state

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize latents using VAE's latents_mean/std (WAN-specific)."""
        if hasattr(self.vae.config, 'latents_mean') and self.vae.config.latents_mean is not None:
            mean = torch.tensor(self.vae.config.latents_mean, dtype=latents.dtype, device=latents.device)
            std = torch.tensor(self.vae.config.latents_std, dtype=latents.dtype, device=latents.device)
            # Reshape for broadcasting: [C] -> [1, C, 1, 1, 1]
            shape = [1, -1] + [1] * (latents.ndim - 2)
            mean = mean.view(*shape)
            std = std.view(*shape)
            return (latents - mean) / std
        elif hasattr(self.vae.config, 'scaling_factor'):
            return latents * self.vae.config.scaling_factor
        return latents

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Denormalize latents using VAE's latents_mean/std (WAN-specific)."""
        if hasattr(self.vae.config, 'latents_mean') and self.vae.config.latents_mean is not None:
            mean = torch.tensor(self.vae.config.latents_mean, dtype=latents.dtype, device=latents.device)
            std = torch.tensor(self.vae.config.latents_std, dtype=latents.dtype, device=latents.device)
            shape = [1, -1] + [1] * (latents.ndim - 2)
            mean = mean.view(*shape)
            std = std.view(*shape)
            return latents * std + mean
        elif hasattr(self.vae.config, 'scaling_factor'):
            return latents / self.vae.config.scaling_factor
        return latents

    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames to VAE latents.

        Args:
            videos: Tensor[B, C, T, H, W] in [-1, 1]
                    or Tensor[B, T, C, H, W] — will be permuted to BCTHW

        Returns:
            latents: Tensor[B, C', T', H', W'] normalized
        """
        if videos.ndim == 5 and videos.shape[1] != self.vae.config.in_channels:
            # BTCHW -> BCTHW
            videos = videos.permute(0, 2, 1, 3, 4)

        # Cast to VAE dtype (in case input comes in fp32 but model is bf16)
        vae_dtype = next(self.vae.parameters()).dtype
        videos = videos.to(dtype=vae_dtype)

        with torch.set_grad_enabled(self.vae.training):
            latents = self.vae.encode(videos).latent_dist.sample()
        return self._normalize_latents(latents)

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to video frames.

        Args:
            latents: Tensor[B, C', T', H', W']

        Returns:
            videos: Tensor[B, C, T, H, W] in [-1, 1]
        """
        latents = self._denormalize_latents(latents)
        with torch.set_grad_enabled(self.vae.training):
            videos = self.vae.decode(latents).sample
        return videos

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise/velocity with the WAN transformer.

        Args:
            noisy_latents: Tensor[B, C', T', H', W']
            timesteps: Tensor[B] — current timestep
            encoder_hidden_states: Tensor[B, seq_len, hidden_dim]

        Returns:
            noise_pred: Tensor[B, C', T', H', W']
        """
        output = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        return output[0]
