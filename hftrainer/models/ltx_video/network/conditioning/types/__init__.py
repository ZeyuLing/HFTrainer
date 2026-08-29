# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Conditioning type implementations."""

from hftrainer.models.ltx_video.network.conditioning.types.attention_strength_wrapper import ConditioningItemAttentionStrengthWrapper
from hftrainer.models.ltx_video.network.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from hftrainer.models.ltx_video.network.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
from hftrainer.models.ltx_video.network.conditioning.types.latent_cond import VideoConditionByLatentIndex
from hftrainer.models.ltx_video.network.conditioning.types.mask_cond import VideoConditionByMask
from hftrainer.models.ltx_video.network.conditioning.types.reference_audio_cond import AudioConditionByReferenceLatent
from hftrainer.models.ltx_video.network.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent

__all__ = [
    "AudioConditionByReferenceLatent",
    "ConditioningItemAttentionStrengthWrapper",
    "VideoConditionByKeyframeIndex",
    "VideoConditionByLatentIndex",
    "VideoConditionByMask",
    "VideoConditionByReferenceLatent",
    "VideoGeneratedKeyframeSlots",
]
