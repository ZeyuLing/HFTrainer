"""MotionStreamer text-to-motion pipeline.

Drives the hftrainer-native MotionStreamer implementation
(``hftrainer.models.motion.motionstreamer.network``): SentenceT5-XXL text features ->
LLaMA autoregressive transformer with classifier-free guidance and per-token
diffusion sampling -> latent tokens -> causal TAE decoder -> 272-dim motion.

Matches the upstream eval generation path
(``utils.eval_trans.evaluation_transformer_272_single`` ->
``LLaMAHF.sample_for_eval_CFG`` + ``Causal_HumanTAE.forward_decoder``) so the
reproduced metrics align with the released checkpoints.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES

# MotionStreamer token unit length (TAE temporal downsample = stride_t**down_t = 4).
MS_UNIT_LENGTH = 4
# Block size 78 -> at most 77 latent tokens after the text-condition slot.
MS_MAX_TOKENS = 77


@PIPELINES.register_module()
class MotionStreamerPipeline(BasePipeline):
    """Inference pipeline for the MotionStreamer bundle."""

    BUNDLE_CLS = "hftrainer.models.motion.motionstreamer.MotionStreamerBundle"

    def __init__(self, bundle, device: Optional[str] = None, **kwargs):
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
    def clamp_length(n_frames: int) -> int:
        """Clamp a target frame count to a valid (token-aligned) motion length."""
        n_tokens = int(n_frames) // MS_UNIT_LENGTH
        n_tokens = max(1, min(MS_MAX_TOKENS, n_tokens))
        return n_tokens * MS_UNIT_LENGTH

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        guidance_param: Optional[float] = None,
        progress: bool = False,
    ) -> List[np.ndarray]:
        """Generate MotionStreamer-272 motions (physical scale) from text.

        Args:
            captions: list of B text prompts.
            lengths: list of B target lengths in frames (30 fps native). Each is
                clamped to a token-aligned length.
            guidance_param: classifier-free guidance scale; defaults to the
                bundle's configured value (4.0).
            progress: optional progress print of per-sample generation.

        Returns:
            List of B arrays, each ``(length_i, 272)`` un-standardized.
        """
        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        bundle = self.bundle
        if bundle.text_model is None:
            raise RuntimeError(
                "MotionStreamerBundle was built with load_text_model=False; "
                "the SentenceT5 text encoder is required for generation."
            )
        device = self.device
        scale = bundle.guidance_param if guidance_param is None else float(guidance_param)

        outputs: List[np.ndarray] = []
        for i, (cap, raw_len) in enumerate(zip(captions, lengths)):
            length = self.clamp_length(raw_len)
            # Upstream calls sample_for_eval_CFG with a single-caption list.
            latent = bundle.ar.sample_for_eval_CFG(
                [cap],
                length=length,
                tokenize_model=bundle.text_model,
                device=device,
                unit_length=MS_UNIT_LENGTH,
                cfg=scale,
            )  # (1, length//unit, latent_dim)
            motion = bundle.tae.forward_decoder(latent)  # (1, T, 272)
            motion = bundle.denormalize(motion[0])  # (T, 272)
            motion = motion[:length].cpu().numpy().astype(np.float32)
            outputs.append(motion)
            if progress:
                print(f"[ms] {i + 1}/{len(captions)} len={length} -> {motion.shape}", flush=True)

        return outputs

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
