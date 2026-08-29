"""Compact, repository-local Wan-style causal video KL autoencoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .common import (
    AutoencoderKLOutput,
    DecoderOutput,
    DiagonalGaussianDistribution,
    LocalWanModelMixin,
    WanConfig,
    pick_group_count,
)


class WanCausalConv3d(nn.Conv3d):
    """Conv3d with left-only temporal padding and symmetric spatial padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        bias: bool = True,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        super().__init__(
            in_channels,
            out_channels,
            tuple(kernel_size),
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kt, kh, kw = self.kernel_size
        padding = (kw // 2, kw // 2, kh // 2, kh // 2, kt - 1, 0)
        hidden_states = F.pad(hidden_states, padding)
        return super().forward(hidden_states)


class WanVaeResnetBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dropout: float, norm_num_groups: int
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            pick_group_count(in_channels, norm_num_groups), in_channels, eps=1e-6
        )
        self.conv1 = WanCausalConv3d(in_channels, out_channels, 3)
        self.norm2 = nn.GroupNorm(
            pick_group_count(out_channels, norm_num_groups), out_channels, eps=1e-6
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_channels, out_channels, 3)
        self.nin_shortcut = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.nin_shortcut(hidden_states)
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(self.dropout(F.silu(self.norm2(hidden_states))))
        return hidden_states + residual


class WanVaeAttentionBlock(nn.Module):
    def __init__(self, channels: int, norm_num_groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(
            pick_group_count(channels, norm_num_groups), channels, eps=1e-6
        )
        self.q = nn.Conv3d(channels, channels, 1)
        self.k = nn.Conv3d(channels, channels, 1)
        self.v = nn.Conv3d(channels, channels, 1)
        self.proj_out = nn.Conv3d(channels, channels, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        batch, channels, frames, height, width = hidden_states.shape
        query = self.q(hidden_states).reshape(batch, channels, -1).transpose(1, 2)
        key = self.k(hidden_states).reshape(batch, channels, -1)
        value = self.v(hidden_states).reshape(batch, channels, -1).transpose(1, 2)
        scores = torch.matmul(query.float(), key.float()) * channels**-0.5
        weights = torch.softmax(scores, dim=-1).to(value.dtype)
        hidden_states = torch.matmul(weights, value).transpose(1, 2)
        hidden_states = hidden_states.reshape(batch, channels, frames, height, width)
        return residual + self.proj_out(hidden_states)


class WanVaeMidBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, norm_num_groups: int):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                WanVaeResnetBlock(channels, channels, dropout, norm_num_groups),
                WanVaeResnetBlock(channels, channels, dropout, norm_num_groups),
            ]
        )
        self.attentions = nn.ModuleList(
            [WanVaeAttentionBlock(channels, norm_num_groups)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = self.attentions[0](hidden_states)
        return self.resnets[1](hidden_states)


class WanEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_dim: int,
        z_dim: int,
        num_res_blocks: int,
        dropout: float,
        norm_num_groups: int,
    ):
        super().__init__()
        self.conv_in = WanCausalConv3d(in_channels, base_dim, 3)
        self.resnets = nn.ModuleList(
            [
                WanVaeResnetBlock(base_dim, base_dim, dropout, norm_num_groups)
                for _ in range(max(1, num_res_blocks))
            ]
        )
        self.mid_block = WanVaeMidBlock(base_dim, dropout, norm_num_groups)
        self.norm_out = nn.GroupNorm(
            pick_group_count(base_dim, norm_num_groups), base_dim, eps=1e-6
        )
        self.conv_out = WanCausalConv3d(base_dim, 2 * z_dim, 3)

    def forward(
        self, sample: torch.Tensor, output_size: tuple[int, int, int]
    ) -> torch.Tensor:
        hidden_states = self.conv_in(sample)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if tuple(hidden_states.shape[-3:]) != output_size:
            hidden_states = F.interpolate(
                hidden_states,
                size=output_size,
                mode="trilinear",
                align_corners=False,
            )
        hidden_states = self.mid_block(hidden_states)
        hidden_states = self.conv_out(F.silu(self.norm_out(hidden_states)))
        return hidden_states


class WanDecoder3d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        base_dim: int,
        z_dim: int,
        num_res_blocks: int,
        dropout: float,
        norm_num_groups: int,
    ):
        super().__init__()
        self.conv_in = WanCausalConv3d(z_dim, base_dim, 3)
        self.mid_block = WanVaeMidBlock(base_dim, dropout, norm_num_groups)
        self.resnets = nn.ModuleList(
            [
                WanVaeResnetBlock(base_dim, base_dim, dropout, norm_num_groups)
                for _ in range(max(1, num_res_blocks))
            ]
        )
        self.norm_out = nn.GroupNorm(
            pick_group_count(base_dim, norm_num_groups), base_dim, eps=1e-6
        )
        self.conv_out = WanCausalConv3d(base_dim, out_channels, 3)

    def forward(
        self, sample: torch.Tensor, output_size: tuple[int, int, int]
    ) -> torch.Tensor:
        hidden_states = self.conv_in(sample)
        hidden_states = self.mid_block(hidden_states)
        if tuple(hidden_states.shape[-3:]) != output_size:
            hidden_states = F.interpolate(
                hidden_states,
                size=output_size,
                mode="trilinear",
                align_corners=False,
            )
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        return torch.tanh(self.conv_out(F.silu(self.norm_out(hidden_states))))


class AutoencoderKLWan(LocalWanModelMixin, nn.Module):
    """Wan video VAE API implemented entirely with local PyTorch modules.

    The temporal shape convention is ``latent_t = (frames - 1) // 4 + 1`` and
    decoding yields ``4 * (latent_t - 1) + 1`` frames, matching Wan pipelines.
    """

    component_name = "vae"

    def __init__(
        self,
        base_dim: int = 96,
        decoder_base_dim: int | None = None,
        z_dim: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence[float] = (),
        temperal_downsample: Sequence[bool] = (False, True, True),
        dropout: float = 0.0,
        latents_mean: Sequence[float] | None = None,
        latents_std: Sequence[float] | None = None,
        scaling_factor: float = 1.0,
        scale_factor_temporal: int | None = 4,
        scale_factor_spatial: int | None = 8,
        patch_size: int | Sequence[int] | None = None,
        norm_num_groups: int = 32,
        is_residual: bool = False,
        **kwargs,
    ):
        super().__init__()
        kwargs.pop("latent_channels", None)
        decoder_base_dim = int(decoder_base_dim or base_dim)
        scale_factor_temporal = int(
            scale_factor_temporal
            if scale_factor_temporal is not None
            else 2 ** sum(bool(value) for value in temperal_downsample)
        )
        scale_factor_spatial = int(
            scale_factor_spatial
            if scale_factor_spatial is not None
            else 2 ** max(len(dim_mult) - 1, 0)
        )
        if z_dim <= 0 or base_dim <= 0:
            raise ValueError("VAE channel dimensions must be positive")
        if latents_mean is None:
            latents_mean = [0.0] * z_dim
        if latents_std is None:
            latents_std = [1.0] * z_dim
        if len(latents_mean) != z_dim or len(latents_std) != z_dim:
            raise ValueError(
                "latents_mean and latents_std must each have z_dim entries"
            )
        if any(float(value) == 0.0 for value in latents_std):
            raise ValueError("latents_std entries must be non-zero")

        self.config = WanConfig(
            base_dim=int(base_dim),
            decoder_base_dim=decoder_base_dim,
            z_dim=int(z_dim),
            latent_channels=int(z_dim),
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            dim_mult=list(dim_mult),
            num_res_blocks=int(num_res_blocks),
            attn_scales=list(attn_scales),
            temperal_downsample=list(temperal_downsample),
            dropout=float(dropout),
            latents_mean=list(latents_mean),
            latents_std=list(latents_std),
            scaling_factor=float(scaling_factor),
            scale_factor_temporal=int(scale_factor_temporal),
            scale_factor_spatial=int(scale_factor_spatial),
            patch_size=(
                list(patch_size) if isinstance(patch_size, Sequence) else patch_size
            ),
            norm_num_groups=int(norm_num_groups),
            is_residual=bool(is_residual),
            **kwargs,
        )
        self.encoder = WanEncoder3d(
            in_channels,
            base_dim,
            z_dim,
            num_res_blocks,
            dropout,
            norm_num_groups,
        )
        self.quant_conv = nn.Conv3d(2 * z_dim, 2 * z_dim, 1)
        self.post_quant_conv = nn.Conv3d(z_dim, z_dim, 1)
        self.decoder = WanDecoder3d(
            out_channels,
            decoder_base_dim,
            z_dim,
            num_res_blocks,
            dropout,
            norm_num_groups,
        )
        self.gradient_checkpointing = False
        self.use_slicing = False
        self.use_tiling = False

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def enable_tiling(self, **kwargs) -> None:
        self.use_tiling = True

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def _latent_size(self, sample: torch.Tensor) -> tuple[int, int, int]:
        frames, height, width = sample.shape[-3:]
        temporal = (frames - 1) // self.config.scale_factor_temporal + 1
        spatial = self.config.scale_factor_spatial
        return temporal, max(1, height // spatial), max(1, width // spatial)

    def _decoded_size(self, latents: torch.Tensor) -> tuple[int, int, int]:
        frames, height, width = latents.shape[-3:]
        temporal = (frames - 1) * self.config.scale_factor_temporal + 1
        spatial = self.config.scale_factor_spatial
        return temporal, height * spatial, width * spatial

    def encode(self, sample: torch.Tensor, return_dict: bool = True):
        if sample.ndim != 5:
            raise ValueError(
                f"Expected BCTHW video input, got shape {tuple(sample.shape)}"
            )
        output_size = self._latent_size(sample)
        if self.gradient_checkpointing and self.training and sample.requires_grad:
            moments = checkpoint(
                lambda value: self.encoder(value, output_size),
                sample,
                use_reentrant=False,
            )
        else:
            moments = self.encoder(sample, output_size)
        moments = self.quant_conv(moments)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ):
        del generator
        if latents.ndim != 5:
            raise ValueError(
                f"Expected BCTHW latent input, got shape {tuple(latents.shape)}"
            )
        output_size = self._decoded_size(latents)
        latents = self.post_quant_conv(latents)
        if self.gradient_checkpointing and self.training and latents.requires_grad:
            sample = checkpoint(
                lambda value: self.decoder(value, output_size),
                latents,
                use_reentrant=False,
            )
        else:
            sample = self.decoder(latents, output_size)
        if not return_dict:
            return (sample,)
        return DecoderOutput(sample=sample)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ):
        posterior = self.encode(sample).latent_dist
        latents = (
            posterior.sample(generator=generator)
            if sample_posterior
            else posterior.mode()
        )
        return self.decode(latents, return_dict=return_dict)
