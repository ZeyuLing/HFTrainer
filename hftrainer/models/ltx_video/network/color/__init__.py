# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Colour transforms, primaries, YUV packing, and HLG encode.
VAE working-space helpers (``HDRTransfer``, ``TransferEncoding``) live in
:mod:`hftrainer.models.ltx_video.network.hdr` — import them from there. This package does not re-export
hdr symbols (that would cycle: ``hdr`` imports ``color.primaries``).
"""

from hftrainer.models.ltx_video.network.color.hlg import (
    HlgGpuConverter,
    HlgPyAVEncoder,
    encode_linear_hdr_frames_to_hlg_mp4,
    hlg_inverse_oetf,
)
from hftrainer.models.ltx_video.network.color.primaries import (
    ACESCG_TO_2020,
    ACESCG_TO_SRGB,
    EXR_CHROMATICITIES,
    PRIMARY_MATRIX_TO_2020,
    REC709_TO_2020,
    SRGB_TO_ACESCG,
    Primaries,
    apply_acescg_to_srgb,
    apply_srgb_to_acescg,
)
from hftrainer.models.ltx_video.network.color.yuv import (
    ColorRange,
    ColorSpace,
    FrameConverter,
    PixelFormat,
    rgb_to_yuv,
    rgb_to_yuv420,
    rgb_to_yuv420p10,
    rgb_uint8_converter_,
    yuv420p_bt709_converter_,
)

__all__ = [
    "ACESCG_TO_2020",
    "ACESCG_TO_SRGB",
    "EXR_CHROMATICITIES",
    "PRIMARY_MATRIX_TO_2020",
    "REC709_TO_2020",
    "SRGB_TO_ACESCG",
    "ColorRange",
    "ColorSpace",
    "FrameConverter",
    "HlgGpuConverter",
    "HlgPyAVEncoder",
    "PixelFormat",
    "Primaries",
    "apply_acescg_to_srgb",
    "apply_srgb_to_acescg",
    "encode_linear_hdr_frames_to_hlg_mp4",
    "hlg_inverse_oetf",
    "rgb_to_yuv",
    "rgb_to_yuv420",
    "rgb_to_yuv420p10",
    "rgb_uint8_converter_",
    "yuv420p_bt709_converter_",
]
