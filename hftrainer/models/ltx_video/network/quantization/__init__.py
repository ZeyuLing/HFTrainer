# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
from hftrainer.models.ltx_video.network.quantization.fp8_cast import (
    TRANSFORMER_LINEAR_DOWNCAST_MAP,
    UPCAST_DURING_INFERENCE,
    UpcastWithStochasticRounding,
    fp8_cast_fuse_rule,
)
from hftrainer.models.ltx_video.network.quantization.fp8_scaled_mm import fp8_scaled_mm_fuse_rule
from hftrainer.models.ltx_video.network.quantization.policy import QuantizationPolicy

__all__ = [
    "TRANSFORMER_LINEAR_DOWNCAST_MAP",
    "UPCAST_DURING_INFERENCE",
    "QuantizationPolicy",
    "UpcastWithStochasticRounding",
    "fp8_cast_fuse_rule",
    "fp8_scaled_mm_fuse_rule",
]
