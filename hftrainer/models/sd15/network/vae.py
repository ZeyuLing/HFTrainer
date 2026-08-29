"""Local variational autoencoder used by Stable Diffusion 1.5."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hftrainer.registry import MODEL_COMPONENTS

from ..checkpoint import LocalComponentMixin
from .configuration import ConfigDict
from .outputs import AutoencoderKLOutput, DecoderOutput


def _groups(channels: int, requested: int) -> int:
    value = min(int(requested), int(channels))
    while channels % value:
        value -= 1
    return max(1, value)


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other: DiagonalGaussianDistribution | None = None) -> torch.Tensor:
        dimensions = tuple(range(1, self.mean.ndim))
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device)
        if other is None:
            return 0.5 * torch.sum(
                self.mean.square() + self.var - 1.0 - self.logvar,
                dim=dimensions,
            )
        return 0.5 * torch.sum(
            (self.mean - other.mean).square() / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=dimensions,
        )


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        groups: int = 32,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels, groups), in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels, groups), out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(self.nonlinearity(self.norm1(hidden_states)))
        hidden_states = self.conv2(self.dropout(self.nonlinearity(self.norm2(hidden_states))))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(_groups(channels, groups), channels, eps=eps, affine=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Dropout(0.0)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        batch, channels, height, width = hidden_states.shape
        hidden_states = self.group_norm(hidden_states).view(batch, channels, height * width).transpose(1, 2)
        query = self.to_q(hidden_states)[:, None]
        key = self.to_k(hidden_states)[:, None]
        value = self.to_v(hidden_states)[:, None]
        hidden_states = F.scaled_dot_product_attention(query, key, value)[:, 0]
        hidden_states = self.to_out[1](self.to_out[0](hidden_states))
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, channels, height, width)
        return residual + hidden_states


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(hidden_states, (0, 1, 0, 1)))


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode='nearest')
        return self.conv(hidden_states)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, add_downsample, groups):
        super().__init__()
        self.resnets = nn.ModuleList()
        for index in range(num_layers):
            self.resnets.append(
                ResnetBlock2D(in_channels if index == 0 else out_channels, out_channels, groups=groups)
            )
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels)]) if add_downsample else None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, add_upsample, groups):
        super().__init__()
        self.resnets = nn.ModuleList()
        for index in range(num_layers):
            self.resnets.append(
                ResnetBlock2D(in_channels if index == 0 else out_channels, out_channels, groups=groups)
            )
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class UNetMidBlock2D(nn.Module):
    def __init__(self, channels: int, groups: int, add_attention: bool):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels, groups=groups),
            ResnetBlock2D(channels, channels, groups=groups),
        ])
        self.attentions = nn.ModuleList([AttentionBlock(channels, groups=groups)]) if add_attention else None

    def forward(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        if self.attentions is not None:
            hidden_states = self.attentions[0](hidden_states)
        return self.resnets[1](hidden_states)


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_channels, block_out_channels, layers_per_block, groups, mid_attention):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        input_channel = block_out_channels[0]
        for index, output_channel in enumerate(block_out_channels):
            self.down_blocks.append(
                DownEncoderBlock2D(
                    input_channel,
                    output_channel,
                    layers_per_block,
                    add_downsample=index < len(block_out_channels) - 1,
                    groups=groups,
                )
            )
            input_channel = output_channel
        self.mid_block = UNetMidBlock2D(input_channel, groups, mid_attention)
        self.conv_norm_out = nn.GroupNorm(_groups(input_channel, groups), input_channel, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(input_channel, 2 * latent_channels, 3, padding=1)

    def forward(self, sample):
        sample = self.conv_in(sample)
        for block in self.down_blocks:
            sample = block(sample)
        sample = self.mid_block(sample)
        return self.conv_out(self.conv_act(self.conv_norm_out(sample)))


class Decoder(nn.Module):
    def __init__(self, out_channels, latent_channels, block_out_channels, layers_per_block, groups, mid_attention):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))
        self.conv_in = nn.Conv2d(latent_channels, reversed_channels[0], 3, padding=1)
        self.mid_block = UNetMidBlock2D(reversed_channels[0], groups, mid_attention)
        self.up_blocks = nn.ModuleList()
        input_channel = reversed_channels[0]
        for index, output_channel in enumerate(reversed_channels):
            self.up_blocks.append(
                UpDecoderBlock2D(
                    input_channel,
                    output_channel,
                    layers_per_block + 1,
                    add_upsample=index < len(reversed_channels) - 1,
                    groups=groups,
                )
            )
            input_channel = output_channel
        self.conv_norm_out = nn.GroupNorm(_groups(input_channel, groups), input_channel, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(input_channel, out_channels, 3, padding=1)

    def forward(self, sample):
        sample = self.conv_in(sample)
        sample = self.mid_block(sample)
        for block in self.up_blocks:
            sample = block(sample)
        return self.conv_out(self.conv_act(self.conv_norm_out(sample)))


@MODEL_COMPONENTS.register_module(name='AutoencoderKL', force=True)
class AutoencoderKL(LocalComponentMixin, nn.Module):
    """Convolutional KL autoencoder with SD1.5-compatible component names."""

    component_kind = 'autoencoder_kl'

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Sequence[str] = ('DownEncoderBlock2D',) * 4,
        up_block_types: Sequence[str] = ('UpDecoderBlock2D',) * 4,
        block_out_channels: Sequence[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = 'silu',
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int | Sequence[int] = 512,
        scaling_factor: float = 0.18215,
        shift_factor: float | None = None,
        latents_mean: Sequence[float] | None = None,
        latents_std: Sequence[float] | None = None,
        force_upcast: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
        **metadata: Any,
    ):
        nn.Module.__init__(self)
        if act_fn not in {'silu', 'swish'}:
            raise ValueError("AutoencoderKL currently supports act_fn='silu'.")
        channels = tuple(int(value) for value in block_out_channels)
        self.config = ConfigDict(
            in_channels=int(in_channels), out_channels=int(out_channels),
            down_block_types=list(down_block_types), up_block_types=list(up_block_types),
            block_out_channels=list(channels), layers_per_block=int(layers_per_block),
            act_fn=act_fn, latent_channels=int(latent_channels),
            norm_num_groups=int(norm_num_groups), sample_size=sample_size,
            scaling_factor=float(scaling_factor), shift_factor=shift_factor,
            latents_mean=latents_mean, latents_std=latents_std,
            force_upcast=bool(force_upcast), use_quant_conv=bool(use_quant_conv),
            use_post_quant_conv=bool(use_post_quant_conv),
            mid_block_add_attention=bool(mid_block_add_attention), **metadata,
        )
        self.encoder = Encoder(
            in_channels, latent_channels, channels, layers_per_block,
            norm_num_groups, mid_block_add_attention,
        )
        self.decoder = Decoder(
            out_channels, latent_channels, channels, layers_per_block,
            norm_num_groups, mid_block_add_attention,
        )
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else nn.Identity()
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else nn.Identity()
        self.use_slicing = False
        self.use_tiling = False

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def enable_tiling(self, **_):
        self.use_tiling = True

    def disable_tiling(self):
        self.use_tiling = False

    def encode(self, sample: torch.Tensor, return_dict: bool = True):
        moments = self.quant_conv(self.encoder(sample))
        posterior = DiagonalGaussianDistribution(moments)
        output = AutoencoderKLOutput(latent_dist=posterior)
        return output if return_dict else (posterior,)

    def decode(self, latents: torch.Tensor, return_dict: bool = True, **_):
        sample = self.decoder(self.post_quant_conv(latents))
        output = DecoderOutput(sample=sample)
        return output if return_dict else (sample,)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        posterior = self.encode(sample).latent_dist
        latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(latents, return_dict=return_dict)
