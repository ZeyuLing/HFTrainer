# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""User-facing dispatch for HFTrainer's local LTX quantization policies.

Each repository-owned backend exposes a ``build_policy`` factory. This module
keeps CLI/pipeline selection in one place so adding or removing a local backend
does not affect the model graph.
"""

from enum import Enum

from typing_extensions import assert_never

from hftrainer.models.ltx_video.network.quantization import QuantizationPolicy
from hftrainer.models.ltx_video.network.quantization.fp8_cast import build_policy as _build_fp8_cast_policy
from hftrainer.models.ltx_video.network.quantization.fp8_scaled_mm import build_policy as _build_fp8_scaled_mm_policy
class QuantizationKind(str, Enum):
    FP8_CAST = "fp8-cast"
    FP8_SCALED_MM = "fp8-scaled-mm"

    def to_policy(self, checkpoint_path: str | None = None) -> QuantizationPolicy:
        """Build the :class:`QuantizationPolicy` for this kind.
        ``checkpoint_path`` is required for both repository-owned FP8 modes.
        """
        match self:
            case QuantizationKind.FP8_CAST:
                if checkpoint_path is None:
                    raise ValueError(f"{self.value} quantization requires checkpoint_path.")
                return _build_fp8_cast_policy(checkpoint_path)
            case QuantizationKind.FP8_SCALED_MM:
                if checkpoint_path is None:
                    raise ValueError(f"{self.value} quantization requires checkpoint_path.")
                return _build_fp8_scaled_mm_policy(checkpoint_path)
            case _:
                assert_never(self)
