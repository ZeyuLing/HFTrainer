# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Latent upsampler model components."""

from hftrainer.models.ltx_video.network.model.upsampler.model import LatentUpsampler, upsample_video
from hftrainer.models.ltx_video.network.model.upsampler.model_configurator import LatentUpsamplerConfigurator

__all__ = [
    "LatentUpsampler",
    "LatentUpsamplerConfigurator",
    "upsample_video",
]
