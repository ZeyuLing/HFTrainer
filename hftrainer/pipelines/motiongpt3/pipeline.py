"""MotionGPT3 text-to-motion pipeline."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


@PIPELINES.register_module()
class MotionGPT3Pipeline(BasePipeline):
    """Inference pipeline for the MotionGPT3 bundle."""

    BUNDLE_CLS = "hftrainer.models.motion.motiongpt3.MotionGPT3Bundle"

    def __init__(self, bundle, device=None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.bundle.to_device(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @staticmethod
    def clamp_length(n_frames: int, min_length: int = 40, max_length: int = 196) -> int:
        length = (int(n_frames) // 4) * 4
        return max(min_length, min(max_length, length))

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        stage: str = "test",
        temperature: float = 1.0,
    ) -> List[np.ndarray]:
        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        model = self.bundle.model
        device = self.device
        lengths = [self.clamp_length(x) for x in lengths]

        outputs = model.lm.generate_conditional(
            list(captions),
            lengths=lengths,
            stage=stage,
            tasks=None,
        )
        sampled_token_latents, motion_mask = model.lm.sample_tokens(
            outputs,
            model.lm.device,
            temperature=temperature,
            cfg=model.guidance_scale,
            vae_mean_std_inv=model.vae.mean_std_inv,
        )
        z = sampled_token_latents.reshape(len(lengths), model.vae.latent_size, -1).permute(1, 0, 2)
        feats = model.vae.decode(z, lengths=lengths)
        if motion_mask is not None:
            feats = feats.clone()
            mask = motion_mask.to(device=feats.device, dtype=torch.bool)
            while mask.ndim < feats.ndim:
                mask = mask.unsqueeze(-1)
            feats = torch.where(mask, torch.zeros_like(feats), feats)
        feats = self.bundle.denormalize(feats).detach().cpu().numpy().astype(np.float32)
        return [feats[i, : lengths[i]] for i in range(len(lengths))]

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
