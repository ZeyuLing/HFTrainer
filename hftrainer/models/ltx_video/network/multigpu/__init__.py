# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""
Multi-GPU utilities for LTX models.
This package provides utilities for running LTX models across multiple GPUs
using tiled data-parallel techniques and sharded state-dict utilities.
"""

from hftrainer.models.ltx_video.network.multigpu import transformer, vae
from hftrainer.models.ltx_video.network.multigpu.sharded_sd import ShardedSD
from hftrainer.models.ltx_video.network.tiling import DimensionTilingConfig, TileCountConfig

__all__ = ["DimensionTilingConfig", "ShardedSD", "TileCountConfig", "transformer", "vae"]
