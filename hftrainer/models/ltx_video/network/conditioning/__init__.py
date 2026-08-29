# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Conditioning utilities: latent state, tools, and conditioning types."""

from hftrainer.models.ltx_video.network.conditioning.exceptions import ConditioningError
from hftrainer.models.ltx_video.network.conditioning.item import ConditioningItem
from hftrainer.models.ltx_video.network.conditioning.types import (
    AudioConditionByReferenceLatent,
    ConditioningItemAttentionStrengthWrapper,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
    VideoConditionByMask,
    VideoConditionByReferenceLatent,
    VideoGeneratedKeyframeSlots,
)

__all__ = [
    "AudioConditionByReferenceLatent",
    "ConditioningError",
    "ConditioningItem",
    "ConditioningItemAttentionStrengthWrapper",
    "VideoConditionByKeyframeIndex",
    "VideoConditionByLatentIndex",
    "VideoConditionByMask",
    "VideoConditionByReferenceLatent",
    "VideoGeneratedKeyframeSlots",
]
