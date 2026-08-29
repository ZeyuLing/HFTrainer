"""DMD one-step text-to-image inference pipeline."""

from typing import List, Union

import torch

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


@PIPELINES.register_module()
class DMDPipeline(BasePipeline):
    """Generate images with a DMD bundle."""

    def __init__(self, bundle, guidance_scale: float = None):
        super().__init__(bundle)
        self.guidance_scale = guidance_scale

    def __call__(self, prompts: Union[str, List[str]]):
        if isinstance(prompts, str):
            prompts = [prompts]

        device = next(self.bundle.generator_unet.parameters()).device
        with torch.no_grad():
            cond_embeddings = self.bundle.encode_text(prompts).to(device)
            uncond_embeddings = self.bundle.get_unconditional_text_embeddings(
                len(prompts), device=device
            )
            noise = self.bundle.sample_latent_noise(len(prompts), device=device)
            latents = self.bundle.generate_latents(
                noise,
                cond_embeddings,
                uncond_embeddings=uncond_embeddings,
                guidance_scale=self.guidance_scale,
            )
            images = self.bundle.decode_latent(latents)
        return (images / 2 + 0.5).clamp(0, 1)
