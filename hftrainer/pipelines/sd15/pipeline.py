"""SD1.5 inference pipeline."""

import torch
from typing import List, Optional, Union

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


@PIPELINES.register_module()
class SD15Pipeline(BasePipeline):
    """
    Inference pipeline for Stable Diffusion 1.5.

    Usage:
        pipeline = SD15Pipeline(bundle=sd15_bundle)
        images = pipeline("a photo of a cat")
    """

    def __init__(
        self,
        bundle,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
    ):
        super().__init__(bundle)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> torch.Tensor:
        """
        Generate images from text prompts.

        Args:
            prompt: text prompt(s)
            num_inference_steps: number of denoising steps
            guidance_scale: classifier-free guidance scale
            height, width: image dimensions (must be multiples of 8)
            negative_prompt: negative prompt(s) for CFG

        Returns:
            images: Tensor[B, 3, H, W] in [0, 1]
        """
        import copy

        num_inference_steps = num_inference_steps or self.num_inference_steps
        guidance_scale = guidance_scale or self.guidance_scale
        height = height or self.height
        width = width or self.width

        if isinstance(prompt, str):
            prompt = [prompt]
        bsz = len(prompt)

        scheduler = copy.deepcopy(self.bundle.scheduler)
        scheduler.set_timesteps(num_inference_steps)

        device = next(self.bundle.unet.parameters()).device
        dtype = next(self.bundle.unet.parameters()).dtype

        # Encode text
        encoder_hidden_states = self.bundle.encode_text(prompt)

        # Encode negative prompt for CFG
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            neg_prompts = negative_prompt if negative_prompt else [''] * bsz
            if isinstance(neg_prompts, str):
                neg_prompts = [neg_prompts] * bsz
            uncond_hidden_states = self.bundle.encode_text(neg_prompts)
            # Concatenate for batch processing: [uncond, cond]
            encoder_hidden_states_cfg = torch.cat([uncond_hidden_states, encoder_hidden_states])

        # Initialize random latents
        latent_h, latent_w = height // 8, width // 8
        latents = torch.randn(bsz, 4, latent_h, latent_w, device=device, dtype=dtype)
        latents = latents * scheduler.init_noise_sigma

        # Denoising loop
        for t in scheduler.timesteps:
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.bundle.predict_noise(
                    latent_model_input,
                    t.expand(bsz * 2).to(device),
                    encoder_hidden_states_cfg,
                )
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                latent_model_input = scheduler.scale_model_input(latents, t)
                noise_pred = self.bundle.predict_noise(
                    latent_model_input,
                    t.expand(bsz).to(device),
                    encoder_hidden_states,
                )
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        images = self.bundle.decode_latent(latents)
        images = (images / 2 + 0.5).clamp(0, 1)
        return images.cpu()
