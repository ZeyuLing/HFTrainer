"""Local conditional U-Net for Stable Diffusion 1.5."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hftrainer.registry import MODEL_COMPONENTS

from ..checkpoint import LocalComponentMixin
from .configuration import ConfigDict
from .outputs import UNet2DConditionOutput


def _groups(channels: int, requested: int) -> int:
    groups = min(int(requested), int(channels))
    while channels % groups:
        groups -= 1
    return max(1, groups)


def _as_tuple(value, length: int):
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f'Expected {length} values, got {len(value)}.')
        return tuple(value)
    return (value,) * length


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    max_period: int = 10000,
) -> torch.Tensor:
    half = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=timesteps.device
    ) / max(half - downscale_freq_shift, 1)
    embedding = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half:], embedding[:, :half]], dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class Timesteps(nn.Module):
    def __init__(self, num_channels, flip_sin_to_cos=True, downscale_freq_shift=0.0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, out_dim=None, act_fn='silu'):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn in {'silu', 'swish'} else nn.GELU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim or time_embed_dim)

    def forward(self, sample):
        return self.linear_2(self.act(self.linear_1(sample)))


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels,
        *,
        groups=32,
        eps=1e-5,
        dropout=0.0,
        time_embedding_norm='default',
        output_scale_factor=1.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels, groups), in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        projection_out = out_channels * (2 if time_embedding_norm == 'scale_shift' else 1)
        self.time_emb_proj = nn.Linear(temb_channels, projection_out)
        self.norm2 = nn.GroupNorm(_groups(out_channels, groups), out_channels, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = float(output_scale_factor)

    def forward(self, input_tensor, temb):
        hidden_states = self.conv1(self.nonlinearity(self.norm1(input_tensor)))
        time_emb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        if self.time_embedding_norm == 'scale_shift':
            scale, shift = torch.chunk(time_emb, 2, dim=1)
            hidden_states = self.norm2(hidden_states) * (1 + scale) + shift
            hidden_states = self.nonlinearity(hidden_states)
        else:
            hidden_states = self.nonlinearity(self.norm2(hidden_states + time_emb))
        hidden_states = self.conv2(self.dropout(hidden_states))
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        return (input_tensor + hidden_states) / self.output_scale_factor


class Attention(nn.Module):
    def __init__(self, query_dim, cross_attention_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        cross_attention_dim = cross_attention_dim or query_dim
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)])

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, query_length, _ = hidden_states.shape
        key_length = context.shape[1]
        query = self.to_q(hidden_states).view(batch, query_length, self.heads, self.dim_head).transpose(1, 2)
        key = self.to_k(context).view(batch, key_length, self.heads, self.dim_head).transpose(1, 2)
        value = self.to_v(context).view(batch, key_length, self.heads, self.dim_head).transpose(1, 2)
        mask = attention_mask
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None]
            if mask.dtype != torch.bool:
                mask = mask.to(query.dtype)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, query_length, -1)
        return self.to_out[1](self.to_out[0](hidden_states))


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.0, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.ModuleList([GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim)])

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, cross_attention_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(
            dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, encoder_attention_mask=None):
        hidden_states = hidden_states + self.attn1(self.norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        return hidden_states + self.ff(self.norm3(hidden_states))


class Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        cross_attention_dim,
        num_layers=1,
        norm_num_groups=32,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.norm = nn.GroupNorm(_groups(in_channels, norm_num_groups), in_channels, eps=1e-6)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, 1)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                cross_attention_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.proj_out = nn.Conv2d(inner_dim, in_channels, 1)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, encoder_attention_mask=None):
        residual = hidden_states
        batch, _, height, width = hidden_states.shape
        hidden_states = self.proj_in(self.norm(hidden_states))
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.view(batch, inner_dim, height * width).transpose(1, 2)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, inner_dim, height, width)
        return residual + self.proj_out(hidden_states)


class Downsample2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, hidden_states):
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states, output_size=None):
        hidden_states = F.interpolate(
            hidden_states,
            size=output_size,
            scale_factor=None if output_size else 2.0,
            mode='nearest',
        )
        return self.conv(hidden_states)


class DownBlock2D(nn.Module):
    has_cross_attention = False

    def __init__(self, in_channels, out_channels, temb_channels, num_layers, add_downsample, groups, **_):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels if index == 0 else out_channels,
                out_channels,
                temb_channels,
                groups=groups,
            )
            for index in range(num_layers)
        ])
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels)]) if add_downsample else None

    def forward(self, hidden_states, temb, **_):
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class CrossAttnDownBlock2D(DownBlock2D):
    has_cross_attention = True

    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels,
        num_layers,
        add_downsample,
        groups,
        cross_attention_dim,
        num_attention_heads,
        transformer_layers=1,
        **_,
    ):
        super().__init__(in_channels, out_channels, temb_channels, num_layers, add_downsample, groups)
        head_dim = out_channels // num_attention_heads
        self.attentions = nn.ModuleList([
            Transformer2DModel(
                out_channels,
                num_attention_heads,
                head_dim,
                cross_attention_dim,
                num_layers=transformer_layers,
                norm_num_groups=groups,
            )
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, temb, encoder_hidden_states=None, attention_mask=None, encoder_attention_mask=None, **_):
        output_states = ()
        for resnet, attention in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class UpBlock2D(nn.Module):
    has_cross_attention = False

    def __init__(
        self,
        in_channels,
        prev_output_channel,
        out_channels,
        temb_channels,
        num_layers,
        add_upsample,
        groups,
        **_,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for index in range(num_layers):
            skip_channels = in_channels if index == num_layers - 1 else out_channels
            current_channels = prev_output_channel if index == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(current_channels + skip_channels, out_channels, temb_channels, groups=groups)
            )
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None

    def forward(self, hidden_states, res_hidden_states_tuple, temb, upsample_size=None, **_):
        for resnet in self.resnets:
            skip = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            if hidden_states.shape[-2:] != skip.shape[-2:]:
                hidden_states = F.interpolate(hidden_states, size=skip.shape[-2:], mode='nearest')
            hidden_states = resnet(torch.cat([hidden_states, skip], dim=1), temb)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states, output_size=upsample_size)
        return hidden_states


class CrossAttnUpBlock2D(UpBlock2D):
    has_cross_attention = True

    def __init__(
        self,
        in_channels,
        prev_output_channel,
        out_channels,
        temb_channels,
        num_layers,
        add_upsample,
        groups,
        cross_attention_dim,
        num_attention_heads,
        transformer_layers=1,
        **_,
    ):
        super().__init__(
            in_channels, prev_output_channel, out_channels, temb_channels,
            num_layers, add_upsample, groups,
        )
        head_dim = out_channels // num_attention_heads
        self.attentions = nn.ModuleList([
            Transformer2DModel(
                out_channels,
                num_attention_heads,
                head_dim,
                cross_attention_dim,
                num_layers=transformer_layers,
                norm_num_groups=groups,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb,
        encoder_hidden_states=None,
        attention_mask=None,
        encoder_attention_mask=None,
        upsample_size=None,
        **_,
    ):
        for resnet, attention in zip(self.resnets, self.attentions):
            skip = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            if hidden_states.shape[-2:] != skip.shape[-2:]:
                hidden_states = F.interpolate(hidden_states, size=skip.shape[-2:], mode='nearest')
            hidden_states = resnet(torch.cat([hidden_states, skip], dim=1), temb)
            hidden_states = attention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states, output_size=upsample_size)
        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    has_cross_attention = True

    def __init__(self, channels, temb_channels, groups, cross_attention_dim, num_attention_heads, transformer_layers=1):
        super().__init__()
        head_dim = channels // num_attention_heads
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels, temb_channels, groups=groups),
            ResnetBlock2D(channels, channels, temb_channels, groups=groups),
        ])
        self.attentions = nn.ModuleList([
            Transformer2DModel(
                channels,
                num_attention_heads,
                head_dim,
                cross_attention_dim,
                num_layers=transformer_layers,
                norm_num_groups=groups,
            )
        ])

    def forward(self, hidden_states, temb, encoder_hidden_states=None, attention_mask=None, encoder_attention_mask=None, **_):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self.resnets[1](hidden_states, temb)


@MODEL_COMPONENTS.register_module(name='UNet2DConditionModel', force=True)
class UNet2DConditionModel(LocalComponentMixin, nn.Module):
    """Conditional latent-space U-Net with the canonical SD1.5 topology."""

    component_kind = 'unet_2d_condition'

    def __init__(
        self,
        sample_size: int | Sequence[int] = 64,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Sequence[str] = (
            'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D',
            'CrossAttnDownBlock2D', 'DownBlock2D',
        ),
        mid_block_type: str = 'UNetMidBlock2DCrossAttn',
        up_block_types: Sequence[str] = (
            'UpBlock2D', 'CrossAttnUpBlock2D',
            'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D',
        ),
        block_out_channels: Sequence[int] = (320, 640, 1280, 1280),
        layers_per_block: int | Sequence[int] = 2,
        cross_attention_dim: int | Sequence[int] = 768,
        attention_head_dim: int | Sequence[int] = 8,
        num_attention_heads: int | Sequence[int] | None = None,
        transformer_layers_per_block: int | Sequence[int] = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        act_fn: str = 'silu',
        time_embedding_dim: int | None = None,
        time_embedding_type: str = 'positional',
        resnet_time_scale_shift: str = 'default',
        **metadata: Any,
    ):
        nn.Module.__init__(self)
        if time_embedding_type != 'positional':
            raise ValueError("Only positional timestep embeddings are supported for SD1.5.")
        if act_fn not in {'silu', 'swish'}:
            raise ValueError("UNet2DConditionModel currently supports act_fn='silu'.")
        channels = tuple(int(value) for value in block_out_channels)
        levels = len(channels)
        if len(down_block_types) != levels or len(up_block_types) != levels:
            raise ValueError('Block type and channel lists must have equal length.')
        layer_counts = tuple(int(value) for value in _as_tuple(layers_per_block, levels))
        cross_dims = tuple(int(value) for value in _as_tuple(cross_attention_dim, levels))
        transformer_depths = tuple(int(value) for value in _as_tuple(transformer_layers_per_block, levels))
        legacy_heads = _as_tuple(attention_head_dim, levels)
        heads = _as_tuple(num_attention_heads, levels) if num_attention_heads is not None else legacy_heads
        heads = tuple(int(value) for value in heads)
        time_embed_dim = int(time_embedding_dim or channels[0] * 4)
        self.config = ConfigDict(
            sample_size=sample_size, in_channels=int(in_channels), out_channels=int(out_channels),
            center_input_sample=bool(center_input_sample), flip_sin_to_cos=bool(flip_sin_to_cos),
            freq_shift=int(freq_shift), down_block_types=list(down_block_types),
            mid_block_type=mid_block_type, up_block_types=list(up_block_types),
            block_out_channels=list(channels), layers_per_block=list(layer_counts),
            cross_attention_dim=list(cross_dims), attention_head_dim=list(legacy_heads),
            num_attention_heads=list(heads), transformer_layers_per_block=list(transformer_depths),
            norm_num_groups=int(norm_num_groups), norm_eps=float(norm_eps), dropout=float(dropout),
            act_fn=act_fn, time_embedding_dim=time_embed_dim,
            time_embedding_type=time_embedding_type, resnet_time_scale_shift=resnet_time_scale_shift,
            **metadata,
        )
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.time_proj = Timesteps(channels[0], flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(channels[0], time_embed_dim, act_fn=act_fn)

        self.down_blocks = nn.ModuleList()
        input_channel = channels[0]
        for index, (block_name, output_channel) in enumerate(zip(down_block_types, channels)):
            block_cls = CrossAttnDownBlock2D if block_name == 'CrossAttnDownBlock2D' else DownBlock2D
            block = block_cls(
                input_channel,
                output_channel,
                time_embed_dim,
                layer_counts[index],
                add_downsample=index < levels - 1,
                groups=norm_num_groups,
                cross_attention_dim=cross_dims[index],
                num_attention_heads=heads[index],
                transformer_layers=transformer_depths[index],
            )
            self.down_blocks.append(block)
            input_channel = output_channel

        if mid_block_type in {None, 'None'}:
            self.mid_block = None
        else:
            self.mid_block = UNetMidBlock2DCrossAttn(
                channels[-1], time_embed_dim, norm_num_groups,
                cross_dims[-1], heads[-1], transformer_depths[-1],
            )

        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        reversed_layers = list(reversed(layer_counts))
        reversed_cross = list(reversed(cross_dims))
        reversed_heads = list(reversed(heads))
        reversed_depths = list(reversed(transformer_depths))
        previous_output = channels[-1]
        for index, (block_name, output_channel) in enumerate(zip(up_block_types, reversed_channels)):
            input_channel = reversed_channels[min(index + 1, levels - 1)]
            block_cls = CrossAttnUpBlock2D if block_name == 'CrossAttnUpBlock2D' else UpBlock2D
            block = block_cls(
                input_channel,
                previous_output,
                output_channel,
                time_embed_dim,
                reversed_layers[index] + 1,
                add_upsample=index < levels - 1,
                groups=norm_num_groups,
                cross_attention_dim=reversed_cross[index],
                num_attention_heads=reversed_heads[index],
                transformer_layers=reversed_depths[index],
            )
            self.up_blocks.append(block)
            previous_output = output_channel

        self.conv_norm_out = nn.GroupNorm(_groups(channels[0], norm_num_groups), channels[0], eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[0], out_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, **_):
        self.gradient_checkpointing = True

    def gradient_checkpointing_enable(self, **kwargs):
        self.enable_gradient_checkpointing(**kwargs)

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float,
        encoder_hidden_states: torch.Tensor,
        class_labels=None,
        timestep_cond=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        added_cond_kwargs=None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        encoder_attention_mask=None,
        return_dict: bool = True,
        **_,
    ):
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        else:
            timestep = timestep.to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        temb = self.time_embedding(self.time_proj(timestep).to(sample.dtype))

        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            sample, residuals = down_block(
                sample,
                temb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
            down_block_res_samples += residuals

        if down_block_additional_residuals is not None:
            down_block_res_samples = tuple(
                base + addition
                for base, addition in zip(down_block_res_samples, down_block_additional_residuals)
            )
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                temb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        for index, up_block in enumerate(self.up_blocks):
            residual_count = len(up_block.resnets)
            residuals = down_block_res_samples[-residual_count:]
            down_block_res_samples = down_block_res_samples[:-residual_count]
            upsample_size = None
            if index < len(self.up_blocks) - 1 and down_block_res_samples:
                upsample_size = down_block_res_samples[-1].shape[-2:]
            sample = up_block(
                sample,
                residuals,
                temb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                upsample_size=upsample_size,
            )
        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))
        output = UNet2DConditionOutput(sample=sample)
        return output if return_dict else (sample,)
