# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MODIFIED BY HFTRAINER: repository-owned PyTorch implementation.  Framework
# dispatch, cache classes, generation mixins, model outputs, configuration and
# vision helpers were replaced with small local equivalents; published module
# and parameter names are intentionally unchanged for checkpoint compatibility.

"""Local Qwen3-VL conditioner used by MiniMax-H3.

The implementation keeps the official state-dict tree (``model.visual.*``,
``model.language_model.*``, ``lm_head.*``), while exposing an early-exit path
for H3's unnormalised ``hidden_states[50]`` conditioner feature. Text
attention uses PyTorch SDPA for the normal inference path so long multimodal
presentations do not materialize quadratic attention-score tensors; requesting
attention weights selects the numerically equivalent eager path.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..common import (
    LocalMiniMaxH3ModelMixin,
    load_config,
    resolve_pretrained_directory,
)
from ..configuration import ConfigDict
from ..outputs import ModelOutput
from .configuration import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig

_SDPA_SUPPORTS_GQA = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")


@dataclass
class BaseModelOutputWithDeepstackFeatures(ModelOutput):
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor | None = None
    deepstack_features: list[torch.Tensor] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class Qwen3VLModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.Tensor
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    rope_deltas: torch.Tensor | None = None


@dataclass
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    rope_deltas: torch.Tensor | None = None


def _activation(name: str, value: torch.Tensor) -> torch.Tensor:
    if name in {"silu", "swish"}:
        return F.silu(value)
    if name in {"gelu_pytorch_tanh", "gelu_new", "gelu_fast"}:
        return F.gelu(value, approximate="tanh")
    if name == "gelu":
        return F.gelu(value)
    raise ValueError(f"Unsupported Qwen3-VL activation {name!r}")


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    batch, heads, sequence, head_dim = hidden_states.shape
    if repeats == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None].expand(
        batch, heads, repeats, sequence, head_dim
    )
    return hidden_states.reshape(batch, heads * repeats, sequence, head_dim)


def _vision_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> torch.Tensor:
    """Official block-major H/W patch positions, repeated per temporal patch."""

    outputs: list[torch.Tensor] = []
    for temporal, height, width in grid_thw.detach().cpu().tolist():
        temporal, height, width = int(temporal), int(height), int(width)
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError(
                "vision grids must be divisible by spatial_merge_size, got "
                f"{height}x{width} and merge={spatial_merge_size}"
            )
        rows, columns = torch.meshgrid(
            torch.arange(height, device=grid_thw.device),
            torch.arange(width, device=grid_thw.device),
            indexing="ij",
        )
        block_shape = (
            height // spatial_merge_size,
            spatial_merge_size,
            width // spatial_merge_size,
            spatial_merge_size,
        )
        rows = rows.reshape(block_shape).transpose(1, 2).flatten()
        columns = columns.reshape(block_shape).transpose(1, 2).flatten()
        outputs.append(torch.stack((rows, columns), dim=-1).repeat(temporal, 1))
    if not outputs:
        return torch.empty((0, 2), dtype=torch.long, device=grid_thw.device)
    return torch.cat(outputs, dim=0)


def _axis_interpolation(
    index: torch.Tensor, size: torch.Tensor, side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    index = index.float()
    source = index * (side - 1) / torch.clamp(size.float() - 1, min=1)
    floor = torch.floor(source)
    offsets = torch.arange(2, device=index.device)
    raw_taps = floor.long()[:, None] + offsets
    taps = raw_taps.clamp(0, side - 1)
    weights = (1 - (source[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)
    return taps, weights


def _vision_interpolation_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear learned-position interpolation in merge-block patch order."""

    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    if not len(counts):
        empty_i = torch.empty((0, 4), dtype=torch.long, device=grid_thw.device)
        return empty_i, empty_i.float()
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts_per_item = F.pad(counts.cumsum(0)[:-1], (1, 0))
    starts = torch.repeat_interleave(starts_per_item, counts)
    within = (torch.arange(int(counts.sum()), device=grid_thw.device) - starts) % (
        heights * widths
    )
    merge = spatial_merge_size
    blocks_w = widths // merge
    in_column = within % merge
    in_row = (within // merge) % merge
    block_column = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    column = block_column * merge + in_column
    row_taps, row_weights = _axis_interpolation(row, heights, num_grid_per_side)
    column_taps, column_weights = _axis_interpolation(column, widths, num_grid_per_side)
    indices = (
        row_taps[:, :, None] * num_grid_per_side + column_taps[:, None, :]
    ).reshape(-1, 4)
    weights = (row_weights[:, :, None] * column_weights[:, None, :]).reshape(-1, 4)
    return indices, weights


def _vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0), value=0)


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(
            _activation(self.hidden_act, self.linear_fc1(hidden_states))
        )


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(self.proj.weight.dtype))
        return hidden_states.reshape(-1, self.embed_dim)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.theta = float(theta)
        self.register_buffer("inv_freq", self._make_inv_freq(), persistent=False)

    def _make_inv_freq(self, device: torch.device | str | None = None) -> torch.Tensor:
        return 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
                / self.dim
            )
        )

    def reset_inv_freq(self, device: torch.device | str) -> None:
        self.inv_freq = self._make_inv_freq(device)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(
        self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.use_postshuffle_norm = bool(use_postshuffle_norm)
        norm_size = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = self.norm(hidden_states.reshape(-1, self.hidden_size))
        else:
            hidden_states = self.norm(hidden_states).reshape(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


def apply_rotary_pos_emb_vision(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_dtype, key_dtype = query.dtype, key.dtype
    query, key = query.float(), key.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    return query.to(query_dtype), key.to(key_dtype)


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        sequence = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(sequence, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query, key = apply_rotary_pos_emb_vision(
            query, key, position_embeddings[0], position_embeddings[1]
        )
        boundaries = cu_seqlens.detach().cpu().tolist()
        outputs: list[torch.Tensor] = []
        weights_output: list[torch.Tensor] = []
        for start, end in itertools.pairwise(boundaries):
            q = query[start:end].transpose(0, 1)
            k = key[start:end].transpose(0, 1)
            v = value[start:end].transpose(0, 1)
            weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
            weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
            outputs.append(torch.matmul(weights, v).transpose(0, 1))
            if output_attentions:
                weights_output.append(weights)
        attention_output = torch.cat(outputs, dim=0).reshape(sequence, self.dim)
        attention_output = self.proj(attention_output)
        if not output_attentions:
            return attention_output, None
        # Packed entries can have different sizes; keep one padded block-diagonal
        # tensor only when all chunks match, otherwise return the flat tuple via
        # an object tensor is undesirable.  Vision callers do not request it in H3.
        if len({value.shape[-1] for value in weights_output}) == 1:
            return attention_output, torch.stack(weights_output)
        return attention_output, None


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention, weights = self.attn(
            self.norm1(hidden_states),
            cu_seqlens,
            position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attention
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states, weights


class Qwen3VLVisionModel(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = config.spatial_merge_size**2
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        if head_dim % 4:
            raise ValueError("vision attention head_dim must be divisible by four")
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [Qwen3VLVisionBlock(config) for _ in range(config.depth)]
        )
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
                for _ in self.deepstack_visual_indexes
            ]
        )
        self.gradient_checkpointing = False

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **_: Any,
    ) -> BaseModelOutputWithDeepstackFeatures | tuple[Any, ...]:
        if grid_thw is None:
            raise ValueError("grid_thw is required for Qwen3-VL vision inputs")
        grid_thw = torch.as_tensor(
            grid_thw, dtype=torch.long, device=hidden_states.device
        )
        expected_rows = int(grid_thw.prod(-1).sum())
        if hidden_states.shape[0] != expected_rows:
            raise ValueError(
                f"pixel patch rows ({hidden_states.shape[0]}) do not match grid_thw ({expected_rows})"
            )
        indices, weights = _vision_interpolation_indices_and_weights(
            grid_thw, self.num_grid_per_side, self.spatial_merge_size
        )
        position_ids = _vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = _vision_cu_seqlens(grid_thw)
        hidden_states = self.patch_embed(hidden_states)
        position = (self.pos_embed(indices) * weights[:, :, None]).sum(1)
        hidden_states = hidden_states + position.to(hidden_states.dtype)
        rotary = self.rotary_pos_emb(position_ids)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())

        collected_hidden = [hidden_states] if output_hidden_states else None
        collected_attentions: list[torch.Tensor] | None = (
            [] if output_attentions else None
        )
        deepstack_features: list[torch.Tensor] = []
        for layer_index, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:

                def checkpointed_block(
                    states: torch.Tensor, current_block: nn.Module = block
                ) -> torch.Tensor:
                    return current_block(
                        states,
                        cu_seqlens,
                        position_embeddings,
                        output_attentions=False,
                    )[0]

                hidden_states = checkpoint(
                    checkpointed_block, hidden_states, use_reentrant=False
                )
                attention = None
            else:
                hidden_states, attention = block(
                    hidden_states,
                    cu_seqlens,
                    position_embeddings,
                    output_attentions=output_attentions,
                )
            if collected_hidden is not None:
                collected_hidden.append(hidden_states)
            if collected_attentions is not None and attention is not None:
                collected_attentions.append(attention)
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack_features.append(
                    self.deepstack_merger_list[merger_index](hidden_states)
                )
        pooled = self.merger(hidden_states)
        output = BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            deepstack_features=deepstack_features,
            hidden_states=tuple(collected_hidden)
            if collected_hidden is not None
            else None,
            attentions=(
                tuple(collected_attentions)
                if collected_attentions is not None
                else None
            ),
        )
        return output if return_dict else output.to_tuple()


class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(
            values.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * values.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3VLTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
        super().__init__()
        self.config = config
        self.rope_theta = float(
            config.rope_parameters.get("rope_theta", config.rope_theta)
        )
        self.mrope_section = [
            int(value) for value in config.rope_parameters["mrope_section"]
        ]
        self.register_buffer("inv_freq", self._make_inv_freq(), persistent=False)

    def _make_inv_freq(self, device: torch.device | str | None = None) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    self.config.head_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / self.config.head_dim
            )
        )

    def reset_inv_freq(self, device: torch.device | str) -> None:
        self.inv_freq = self._make_inv_freq(device)

    @staticmethod
    def apply_interleaved_mrope(
        frequencies: torch.Tensor, sections: list[int]
    ) -> torch.Tensor:
        temporal = frequencies[0].clone()
        for dimension, offset in enumerate((1, 2), start=1):
            length = int(sections[dimension]) * 3
            temporal[..., slice(offset, length, 3)] = frequencies[
                dimension, ..., slice(offset, length, 3)
            ]
        return temporal

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("text position_ids must have shape [3, batch, sequence]")
        inv_freq = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        positions = position_ids[:, :, None, :].float()
        with torch.autocast(
            device_type=hidden_states.device.type
            if hidden_states.device.type not in {"mps"}
            else "cpu",
            enabled=False,
        ):
            frequencies = (inv_freq.to(hidden_states.device) @ positions).transpose(
                2, 3
            )
            frequencies = self.apply_interleaved_mrope(frequencies, self.mrope_section)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cos, sin = embedding.cos(), embedding.sin()
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.layer_type = config.layer_types[layer_idx]
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, sequence, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch, sequence, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, sequence, self.num_key_value_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rotary_pos_emb(
            query, key, position_embeddings[0], position_embeddings[1]
        )
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        present = (key, value) if use_cache else None

        if not output_attentions:
            dropout_p = self.attention_dropout if self.training else 0.0
            if self.num_key_value_groups == 1:
                output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=dropout_p,
                )
            elif _SDPA_SUPPORTS_GQA:
                # Modern torch dispatches this to its native grouped-query
                # kernels without repeating KV or materializing scores.
                output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=dropout_p,
                    enable_gqa=True,
                )
            else:
                # torch 2.0 predates native GQA. Repeating KV is linear in
                # sequence length and still lets SDPA avoid the quadratic
                # attention-weight allocation used by the eager fallback.
                output = F.scaled_dot_product_attention(
                    query,
                    repeat_kv(key, self.num_key_value_groups),
                    repeat_kv(value, self.num_key_value_groups),
                    attn_mask=attention_mask,
                    dropout_p=dropout_p,
                )
            output = output.transpose(1, 2).reshape(
                batch, sequence, self.num_heads * self.head_dim
            )
            return self.o_proj(output), None, present

        repeated_key = repeat_kv(key, self.num_key_value_groups)
        repeated_value = repeat_kv(value, self.num_key_value_groups)
        weights = torch.matmul(query, repeated_key.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(
            weights,
            p=self.attention_dropout if self.training else 0.0,
            training=self.training,
        )
        output = torch.matmul(weights, repeated_value)
        output = output.transpose(1, 2).reshape(
            batch, sequence, self.num_heads * self.head_dim
        )
        return self.o_proj(output), weights if output_attentions else None, present


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
        super().__init__()
        self.hidden_act = config.hidden_act
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            _activation(self.hidden_act, self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        residual = hidden_states
        attention, weights, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attention
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, weights, present


def _causal_attention_mask(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_length: int,
) -> torch.Tensor:
    batch, sequence, _ = hidden_states.shape
    key_length = past_length + sequence
    query_positions = torch.arange(
        past_length, key_length, device=hidden_states.device
    )[:, None]
    key_positions = torch.arange(key_length, device=hidden_states.device)[None, :]
    blocked = key_positions > query_positions
    minimum = torch.finfo(hidden_states.dtype).min
    mask = torch.zeros(
        (sequence, key_length), dtype=hidden_states.dtype, device=hidden_states.device
    ).masked_fill(blocked, minimum)
    mask = mask[None, None].expand(batch, 1, -1, -1).clone()
    if attention_mask is not None:
        attention_mask = attention_mask.to(hidden_states.device)
        if attention_mask.ndim != 2 or attention_mask.shape[0] != batch:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask.shape[1] == sequence and past_length:
            prefix = torch.ones(
                (batch, past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((prefix, attention_mask), dim=1)
        if attention_mask.shape[1] != key_length:
            raise ValueError(
                f"attention_mask length {attention_mask.shape[1]} != key length {key_length}"
            )
        mask = mask.masked_fill(attention_mask[:, None, None, :] == 0, minimum)
    return mask


class Qwen3VLTextModel(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3VLTextDecoderLayer(config, layer_index)
                for layer_index in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)
        self.gradient_checkpointing = False

    @staticmethod
    def _deepstack_process(
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        if int(visual_pos_masks.sum()) != visual_embeds.shape[0]:
            raise ValueError(
                "deepstack feature rows do not match visual placeholder positions"
            )
        output = hidden_states.clone()
        output[visual_pos_masks] = output[visual_pos_masks] + visual_embeds
        return output

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        conditioning_layer: int | None = None,
        **_: Any,
    ) -> Qwen3VLModelOutputWithPast | tuple[Any, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        use_cache = self.config.use_cache if use_cache is None else bool(use_cache)
        if self.gradient_checkpointing and self.training:
            use_cache = False
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one pair per decoder layer")
        past_length = int(past_key_values[0][0].shape[2]) if past_key_values else 0
        if position_ids is None:
            values = torch.arange(
                past_length,
                past_length + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            position_ids = values[None, None].expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        causal_mask = _causal_attention_mask(inputs_embeds, attention_mask, past_length)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        hidden_history: list[torch.Tensor] | None = (
            [hidden_states]
            if output_hidden_states or conditioning_layer is not None
            else None
        )
        attentions: list[torch.Tensor] | None = [] if output_attentions else None
        presents: list[tuple[torch.Tensor, torch.Tensor]] | None = (
            [] if use_cache else None
        )
        stop_layer = (
            len(self.layers) if conditioning_layer is None else int(conditioning_layer)
        )
        if stop_layer < 1 or stop_layer > len(self.layers):
            raise ValueError(
                f"conditioning_layer must be in [1, {len(self.layers)}], got {stop_layer}"
            )
        for layer_index in range(stop_layer):
            layer = self.layers[layer_index]
            past = past_key_values[layer_index] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:

                def checkpointed_layer(
                    states: torch.Tensor, current_layer: nn.Module = layer
                ) -> torch.Tensor:
                    return current_layer(
                        states,
                        position_embeddings,
                        attention_mask=causal_mask,
                        past_key_value=None,
                        use_cache=False,
                        output_attentions=False,
                    )[0]

                hidden_states = checkpoint(
                    checkpointed_layer, hidden_states, use_reentrant=False
                )
                weights, present = None, None
            else:
                hidden_states, weights, present = layer(
                    hidden_states,
                    position_embeddings,
                    attention_mask=causal_mask,
                    past_key_value=past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            # Transformers records the raw decoder-layer output in
            # hidden_states, then injects deepstack visual residuals for the
            # *next* layer. H3 consumes hidden_states[50], so preserve that
            # observable ordering while still feeding the residual forward.
            if hidden_history is not None:
                hidden_history.append(hidden_states)
            if deepstack_visual_embeds is not None and layer_index < len(
                deepstack_visual_embeds
            ):
                if visual_pos_masks is None:
                    raise ValueError(
                        "visual_pos_masks are required for deepstack features"
                    )
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_index],
                )
            if attentions is not None and weights is not None:
                attentions.append(weights)
            if presents is not None and present is not None:
                presents.append(present)

        # H3 consumes the exact unnormalised hidden_states[index].  Do not run
        # the final RMSNorm when early exiting.
        if conditioning_layer is None:
            hidden_states = self.norm(hidden_states)
            if hidden_history is not None:
                hidden_history[-1] = hidden_states
        output = Qwen3VLModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(presents) if presents is not None else None,
            hidden_states=tuple(hidden_history) if hidden_history is not None else None,
            attentions=tuple(attentions) if attentions is not None else None,
        )
        return output if return_dict else output.to_tuple()


class Qwen3VLModel(nn.Module):
    """Vision/text backbone with official checkpoint module names."""

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)
        self.rope_deltas: torch.Tensor | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.language_model.embed_tokens = embeddings

    @staticmethod
    def get_vision_position_ids(
        start_position: int,
        grid_thw: list[int] | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        temporal = int(grid_thw[0]) // temp_merge_size
        height = int(grid_thw[1]) // spatial_merge_size
        width = int(grid_thw[2]) // spatial_merge_size
        temporal_positions = torch.arange(temporal, device=device) * time_interval
        height_positions = torch.arange(height, device=device) + start_position
        width_positions = torch.arange(width, device=device) + start_position
        temporal_grid, height_grid, width_grid = torch.meshgrid(
            temporal_positions,
            height_positions,
            width_positions,
            indexing="ij",
        )
        positions = torch.stack(
            (temporal_grid, height_grid, width_grid), dim=0
        ).reshape(3, -1)
        positions[0] += start_position
        return positions

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or mm_token_type_ids.shape != input_ids.shape:
            raise ValueError(
                "input_ids and mm_token_type_ids must share [batch, sequence]"
            )
        if video_grid_thw is not None:
            video_grid_thw = torch.as_tensor(
                video_grid_thw, dtype=torch.long, device=input_ids.device
            )
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw = video_grid_thw.clone()
            video_grid_thw[:, 0] = 1
        if image_grid_thw is not None:
            image_grid_thw = torch.as_tensor(
                image_grid_thw, dtype=torch.long, device=input_ids.device
            )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        positions = torch.zeros(
            (3, *input_ids.shape),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        deltas: list[torch.Tensor] = []
        merge = self.config.vision_config.spatial_merge_size
        for batch_index in range(input_ids.shape[0]):
            types = mm_token_type_ids[batch_index]
            visible = (
                attention_mask[batch_index].bool()
                if attention_mask is not None
                else torch.ones_like(types, dtype=torch.bool)
            )
            visible_types = types[visible]
            groups: list[tuple[int, int, int]] = []
            for modality, group in itertools.groupby(
                enumerate(visible_types.detach().cpu().tolist()), lambda pair: pair[1]
            ):
                members = list(group)
                groups.append((int(modality), members[0][0], members[-1][0] + 1))
            current_position = 0
            row_positions: list[torch.Tensor] = []
            for modality, start, end in groups:
                length = end - start
                if modality == 0:
                    value = torch.arange(length, device=input_ids.device)
                    row_positions.append(value[None].expand(3, -1) + current_position)
                    current_position += length
                    continue
                if modality not in (1, 2) or grid_iters[modality] is None:
                    raise ValueError(
                        f"modality type {modality} has no matching vision grid"
                    )
                try:
                    grid = next(grid_iters[modality])
                except StopIteration as exc:
                    raise ValueError(
                        "more visual token runs than supplied grids"
                    ) from exc
                vision_positions = self.get_vision_position_ids(
                    current_position,
                    grid,
                    temp_merge_size=1,
                    spatial_merge_size=merge,
                    device=input_ids.device,
                )
                if vision_positions.shape[1] != length:
                    raise ValueError(
                        f"visual token run has {length} rows but its grid requires "
                        f"{vision_positions.shape[1]}"
                    )
                row_positions.append(vision_positions)
                current_position += max(int(grid[1]), int(grid[2])) // merge
            if row_positions:
                row_positions_tensor = torch.cat(row_positions, dim=1)
            else:
                row_positions_tensor = torch.empty(
                    (3, 0), dtype=input_ids.dtype, device=input_ids.device
                )
            positions[:, batch_index, visible] = row_positions_tensor.to(
                input_ids.dtype
            )
            maximum = (
                row_positions_tensor.max() + 1
                if row_positions_tensor.numel()
                else torch.tensor(0, device=input_ids.device)
            )
            deltas.append(maximum - int(visible.sum()))
        return positions, torch.stack(deltas).reshape(-1, 1)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs: Any,
    ) -> BaseModelOutputWithDeepstackFeatures:
        return self.visual(
            pixel_values.to(self.visual.dtype),
            grid_thw=image_grid_thw,
            return_dict=True,
            **kwargs,
        )

    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        **kwargs: Any,
    ) -> BaseModelOutputWithDeepstackFeatures:
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def _replace_visual_features(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        token_id: int,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids is None:
            token_embedding = self.get_input_embeddings()(
                torch.tensor(token_id, device=inputs_embeds.device)
            )
            mask = (inputs_embeds == token_embedding).all(-1)
        else:
            mask = input_ids == token_id
        if int(mask.sum()) != features.shape[0]:
            raise ValueError(
                f"visual feature/token mismatch: {features.shape[0]} features for "
                f"{int(mask.sum())} placeholders of token {token_id}"
            )
        output = inputs_embeds.clone()
        output[mask] = features.to(output.device, output.dtype)
        return output, mask

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        if has_multimodal and mm_token_type_ids is None:
            raise ValueError("mm_token_type_ids are required for multimodal M-RoPE")
        if input_ids is not None and has_multimodal:
            positions, self.rope_deltas = self.get_rope_index(
                input_ids,
                mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            return positions
        past_length = int(past_key_values[0][0].shape[2]) if past_key_values else 0
        sequence = inputs_embeds.shape[1]
        values = torch.arange(
            past_length,
            past_length + sequence,
            device=inputs_embeds.device,
        )
        if self.rope_deltas is not None and past_length:
            values = values[None] + self.rope_deltas.to(values.device)
            return values[None].expand(3, -1, -1)
        return values[None, None].expand(3, inputs_embeds.shape[0], -1)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        conditioning_layer: int | None = None,
        **kwargs: Any,
    ) -> Qwen3VLModelOutputWithPast | tuple[Any, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask: torch.Tensor | None = None
        video_mask: torch.Tensor | None = None
        deepstack_images: list[torch.Tensor] | None = None
        deepstack_videos: list[torch.Tensor] | None = None
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required with pixel_values")
            image_output = self.get_image_features(
                pixel_values, image_grid_thw, **kwargs
            )
            inputs_embeds, image_mask = self._replace_visual_features(
                input_ids,
                inputs_embeds,
                self.config.image_token_id,
                image_output.pooler_output,
            )
            deepstack_images = image_output.deepstack_features
        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("video_grid_thw is required with pixel_values_videos")
            video_output = self.get_video_features(
                pixel_values_videos, video_grid_thw, **kwargs
            )
            inputs_embeds, video_mask = self._replace_visual_features(
                input_ids,
                inputs_embeds,
                self.config.video_token_id,
                video_output.pooler_output,
            )
            deepstack_videos = video_output.deepstack_features

        visual_mask: torch.Tensor | None = None
        deepstack_visual: list[torch.Tensor] | None = None
        if image_mask is not None and video_mask is not None:
            visual_mask = image_mask | video_mask
            deepstack_visual = []
            image_joint = image_mask[visual_mask]
            video_joint = video_mask[visual_mask]
            if len(deepstack_images or []) != len(deepstack_videos or []):
                raise ValueError("image/video deepstack feature counts differ")
            for image_feature, video_feature in zip(
                deepstack_images or [], deepstack_videos or []
            ):
                joint = image_feature.new_zeros(
                    (int(visual_mask.sum()), image_feature.shape[-1])
                )
                joint[image_joint] = image_feature
                joint[video_joint] = video_feature.to(joint.device, joint.dtype)
                deepstack_visual.append(joint)
        elif image_mask is not None:
            visual_mask, deepstack_visual = image_mask, deepstack_images
        elif video_mask is not None:
            visual_mask, deepstack_visual = video_mask, deepstack_videos

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids,
                inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )
        output = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack_visual,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            conditioning_layer=conditioning_layer,
        )
        output.rope_deltas = self.rope_deltas
        return output if return_dict else output.to_tuple()


class Qwen3VLForConditionalGeneration(LocalMiniMaxH3ModelMixin, nn.Module):
    """Checkpoint-compatible local Qwen3-VL conditional language model."""

    config_name = "config.json"
    weights_name = "model.safetensors"
    component_name = "text_encoder"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: str | None = None,
        *,
        torch_dtype: torch.dtype | str | None = None,
        dtype: torch.dtype | str | None = None,
        **kwargs: Any,
    ) -> Qwen3VLForConditionalGeneration:
        # The 33B public conditioner declares BF16 inside text_config rather
        # than at the root.  Loading it into the mixin's default FP32 target
        # would double host memory before the first inference.
        if torch_dtype is None and dtype is None:
            directory = resolve_pretrained_directory(
                pretrained_model_name_or_path, subfolder
            )
            raw_config = load_config(directory)
            text_config = raw_config.get("text_config") or {}
            declared = text_config.get("dtype") or text_config.get("torch_dtype")
            if declared is not None:
                torch_dtype = declared
        return super().from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            dtype=dtype,
            **kwargs,
        )

    def __init__(
        self,
        text_config: dict[str, Any] | Qwen3VLTextConfig | None = None,
        vision_config: dict[str, Any] | Qwen3VLVisionConfig | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        tie_word_embeddings: bool = False,
        model_type: str = "qwen3_vl",
        config: Qwen3VLConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)
        if config is None:
            config = Qwen3VLConfig(
                text_config=text_config,
                vision_config=vision_config,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                vision_start_token_id=vision_start_token_id,
                vision_end_token_id=vision_end_token_id,
                tie_word_embeddings=tie_word_embeddings,
                model_type=model_type,
                **kwargs,
            )
        elif not isinstance(config, Qwen3VLConfig):
            config = Qwen3VLConfig(**dict(config))
        self.config = ConfigDict(config.to_dict())
        # Retain attribute-rich nested objects on the public config.
        self.config.text_config = config.text_config
        self.config.vision_config = config.vision_config
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.gradient_checkpointing = False
        self._initialize_weights(config)

    def _initialize_weights(self, config: Qwen3VLConfig) -> None:
        standard_deviation = float(config.text_config.initializer_range)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=standard_deviation)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        padding_index = config.text_config.pad_token_id
        if padding_index is not None:
            with torch.no_grad():
                self.model.language_model.embed_tokens.weight[padding_index].zero_()

    def _materialize_nonpersistent_buffers(self, device: torch.device | str) -> None:
        for module in self.modules():
            if isinstance(
                module, (Qwen3VLVisionRotaryEmbedding, Qwen3VLTextRotaryEmbedding)
            ):
                module.reset_inv_freq(device)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(embeddings)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, embeddings: nn.Linear) -> None:
        self.lm_head = embeddings

    def get_image_features(self, *args: Any, **kwargs: Any):
        return self.model.get_image_features(*args, **kwargs)

    def get_video_features(self, *args: Any, **kwargs: Any):
        return self.model.get_video_features(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        logits_to_keep: int | torch.Tensor = 0,
        conditioning_layer: int | None = None,
        **kwargs: Any,
    ) -> Qwen3VLCausalLMOutputWithPast | tuple[Any, ...]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            conditioning_layer=conditioning_layer,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int):
            selection = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        else:
            selection = logits_to_keep
        logits = self.lm_head(hidden_states[:, selection])
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous().float()
            shift_labels = labels[:, 1:].contiguous().to(logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        output = Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        attention_mask: torch.Tensor | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Small local autoregressive loop for smoke tests and offline use."""

        generated = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(generated)
        eos_token_id = (
            self.config.text_config.eos_token_id
            if eos_token_id is None
            else eos_token_id
        )
        # Recompute the short prefix each step.  This path is deliberately
        # simple; the forward API separately supports explicit KV tuples.
        for _ in range(int(max_new_tokens)):
            outputs = self(
                input_ids=generated,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
                **model_kwargs,
            )
            next_logits = outputs.logits[:, -1].float()
            if do_sample:
                if temperature <= 0:
                    raise ValueError("temperature must be positive")
                probabilities = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token)), dim=1
            )
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return generated


class MiniMaxH3Qwen3VLEncoder(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with MiniMax-H3's stable early-layer encoding interface."""

    @torch.no_grad()
    def encode(
        self,
        token_ids: list[int] | tuple[int, ...] | torch.Tensor,
        *,
        processor: Any,
        vision_inputs: dict[str, torch.Tensor] | None = None,
        conditioning_layer: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if conditioning_layer is None:
            conditioning_layer = 50
        token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("MiniMax-H3 encodes exactly one presentation at a time")
        compute_device = self.model.language_model.embed_tokens.weight.device
        input_ids = token_ids.to(compute_device)
        mm_types = processor.create_mm_token_type_ids(token_ids.cpu().tolist())
        mm_token_type_ids = torch.tensor(
            mm_types, dtype=torch.long, device=compute_device
        )
        kwargs: dict[str, torch.Tensor] = {}
        for name, value in (vision_inputs or {}).items():
            value = torch.as_tensor(value)
            if name.startswith("pixel_"):
                value = value.to(compute_device, self.dtype)
            else:
                value = value.to(compute_device)
            kwargs[name] = value
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
            conditioning_layer=int(conditioning_layer),
            return_dict=True,
            **kwargs,
        )
        result = outputs.last_hidden_state
        if device is not None or dtype is not None:
            result = result.to(
                device=device if device is not None else result.device,
                dtype=dtype if dtype is not None else result.dtype,
            )
        return result


__all__ = [
    "BaseModelOutputWithDeepstackFeatures",
    "MiniMaxH3Qwen3VLEncoder",
    "Qwen3VLCausalLMOutputWithPast",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLModelOutputWithPast",
    "Qwen3VLTextAttention",
    "Qwen3VLTextDecoderLayer",
    "Qwen3VLTextModel",
    "Qwen3VLTextRMSNorm",
    "Qwen3VLVisionModel",
]
