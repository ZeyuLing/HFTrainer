# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Multi-GPU utilities for the Gemma text encoder."""

from hftrainer.models.ltx_video.network.multigpu.gemma.accelerate_wrapper import AccelerateGemmaWrapper
from hftrainer.models.ltx_video.network.multigpu.gemma.loader import load_gemma_with_device_map

__all__ = ["AccelerateGemmaWrapper", "load_gemma_with_device_map"]
