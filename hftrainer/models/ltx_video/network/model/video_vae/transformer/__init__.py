# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Neighborhood-Attention transformer building blocks for the diffusion video VAE."""

from hftrainer.models.ltx_video.network.model.video_vae.transformer.apply import (
    apply_diffvae_config,
    apply_diffvae_mode,
    build_compile_diffusion_decoder_op,
    build_cutlass_fna_diffusion_decoder_op,
    build_diffvae_mode_op,
    resolve_attention_for_host,
)
from hftrainer.models.ltx_video.network.model.video_vae.transformer.attention import (
    NAAttentionCallable,
    NattenAttention,
    NeighborhoodAttention3D,
    configure_w_chunks,
    natten_available,
)
from hftrainer.models.ltx_video.network.model.video_vae.transformer.blocks import (
    DiffusionNABlock,
    NABlock,
)
from hftrainer.models.ltx_video.network.model.video_vae.transformer.chunked.block import ChunkedDiffusionNABlock
from hftrainer.models.ltx_video.network.model.video_vae.transformer.combined.block import CombinedDiffusionNABlock
from hftrainer.models.ltx_video.network.model.video_vae.transformer.compiling import (
    compile_diffusion_decoder,
    configure_cutlass_fna_diffusion_decoder,
    configure_natten_backend,
)
from hftrainer.models.ltx_video.network.model.video_vae.transformer.config import (
    DiffVAEBlockKind,
    DiffVAEConfig,
    DiffVAEMode,
    NAttentionKind,
)
from hftrainer.models.ltx_video.network.model.video_vae.transformer.fallback_na import (
    EagerSdpaAttention,
    TritonNaAttention,
    triton_na_available,
)
from hftrainer.models.ltx_video.network.model.video_vae.transformer.layers import (
    AdaLNZero,
    ChannelLinear,
    LinearPixelShuffleUpsample,
    modulate,
)
from hftrainer.models.ltx_video.network.model.video_vae.transformer.qkv import QKVProjections
from hftrainer.models.ltx_video.network.model.video_vae.transformer.rope import (
    apply_abs_rope,
    apply_abs_rope_slab,
    configure_abs_rope,
    default_rope_dim_split,
    rope_inv_freqs,
)
from hftrainer.models.ltx_video.network.model.video_vae.transformer.swiglu import SwiGLU, SwiGLUTileSpec, configure_swiglu_tile

__all__ = [
    "AdaLNZero",
    "ChannelLinear",
    "ChunkedDiffusionNABlock",
    "CombinedDiffusionNABlock",
    "DiffVAEBlockKind",
    "DiffVAEConfig",
    "DiffVAEMode",
    "DiffusionNABlock",
    "EagerSdpaAttention",
    "LinearPixelShuffleUpsample",
    "NAAttentionCallable",
    "NABlock",
    "NAttentionKind",
    "NattenAttention",
    "NeighborhoodAttention3D",
    "QKVProjections",
    "SwiGLU",
    "SwiGLUTileSpec",
    "TritonNaAttention",
    "apply_abs_rope",
    "apply_abs_rope_slab",
    "apply_diffvae_config",
    "apply_diffvae_mode",
    "build_compile_diffusion_decoder_op",
    "build_cutlass_fna_diffusion_decoder_op",
    "build_diffvae_mode_op",
    "compile_diffusion_decoder",
    "configure_abs_rope",
    "configure_cutlass_fna_diffusion_decoder",
    "configure_natten_backend",
    "configure_swiglu_tile",
    "configure_w_chunks",
    "default_rope_dim_split",
    "modulate",
    "natten_available",
    "resolve_attention_for_host",
    "rope_inv_freqs",
    "triton_na_available",
]
