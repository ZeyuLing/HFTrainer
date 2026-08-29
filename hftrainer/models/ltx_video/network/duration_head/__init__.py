# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""DurationHead: predicts shot duration from Connector token outputs."""

from hftrainer.models.ltx_video.network.duration_head.duration_head import AttentionPooler, DurationHead
from hftrainer.models.ltx_video.network.duration_head.model_configurator import DURATION_HEAD_KEY_OPS, DurationHeadConfigurator

__all__ = [
    "DURATION_HEAD_KEY_OPS",
    "AttentionPooler",
    "DurationHead",
    "DurationHeadConfigurator",
]
