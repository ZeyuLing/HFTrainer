# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Multi-GPU utilities for VAE decoding."""

from hftrainer.models.ltx_video.network.multigpu.vae.distributed_decoder import DistributedVideoDecoder

__all__ = ["DistributedVideoDecoder"]
