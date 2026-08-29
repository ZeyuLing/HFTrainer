"""WAN text-to-video inference pipeline."""

import torch
from typing import List, Optional, Union

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


@PIPELINES.register_module()
class WanPipeline(BasePipeline):
    """
    Inference pipeline for WAN text-to-video generation.

    Usage:
        pipeline = WanPipeline(bundle=wan_bundle)
        videos = pipeline("a cat running in the park")
    """

    def __init__(
        self,
        bundle,
        num_inference_steps: int = 50,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
    ):
        super().__init__(bundle)
        self.num_inference_steps = num_inference_steps
        self.num_frames = num_frames
        self.height = height
        self.width = width

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: Optional[int] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> torch.Tensor:
        """
        Generate videos from text prompts.

        Args:
            prompt: text prompt(s)
            num_inference_steps: denoising steps
            num_frames: number of output frames
            height, width: frame dimensions (must be multiples of 8)
            negative_prompt: negative prompt(s)

        Returns:
            videos: Tensor[B, T, C, H, W] in [0, 1]
        """
        import copy

        num_inference_steps = num_inference_steps or self.num_inference_steps
        num_frames = num_frames or self.num_frames
        height = height or self.height
        width = width or self.width

        if isinstance(prompt, str):
            prompt = [prompt]
        bsz = len(prompt)

        scheduler = copy.deepcopy(self.bundle.scheduler)
        scheduler.set_timesteps(num_inference_steps)

        device = next(self.bundle.transformer.parameters()).device
        dtype = next(self.bundle.transformer.parameters()).dtype

        # Encode text
        text_embeds = self.bundle.encode_text(prompt)

        # Initialize random latents
        lat_t = (num_frames - 1) // 4 + 1
        lat_h = height // 8
        lat_w = width // 8
        in_channels = self.bundle.transformer.config.in_channels

        latents = torch.randn(bsz, in_channels, lat_t, lat_h, lat_w, device=device, dtype=dtype)

        # Denoising loop
        for t in scheduler.timesteps:
            noise_pred = self.bundle.predict_noise(
                latents,
                t.expand(bsz).to(device),
                text_embeds,
            )
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode and rearrange
        videos = self.bundle.decode_latent(latents)  # BCTHW
        videos = (videos / 2 + 0.5).clamp(0, 1)
        videos = videos.permute(0, 2, 1, 3, 4)  # BCTHW -> BTCHW

        return videos.cpu()
