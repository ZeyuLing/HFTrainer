# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
from enum import Enum


class NormLayerType(Enum):
    GROUP_NORM = "group_norm"
    PIXEL_NORM = "pixel_norm"


class LogVarianceType(Enum):
    PER_CHANNEL = "per_channel"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    NONE = "none"


class PaddingModeType(Enum):
    ZEROS = "zeros"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"
