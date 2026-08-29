# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""
Multi-GPU transformer utilities for LTX models.
This module provides utilities for running LTX transformer models across multiple GPUs
using tiled data parallelism.
"""

from hftrainer.models.ltx_video.network.multigpu.transformer.tiled_data_parallel import (
    TiledDataParallelModelWrapper,
)

__all__ = [
    "TiledDataParallelModelWrapper",
]
