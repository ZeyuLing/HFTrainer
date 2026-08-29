# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
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
# MODIFIED BY HFTRAINER: external Diffusers/PEFT execution has been replaced
# by repository-local PyTorch SDPA, configuration, checkpoint, and training
# primitives.  Public constructor and checkpoint key names remain compatible.

"""Repository-local MiniMax-H3 joint video/audio transformer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from .common import (
    LocalMiniMaxH3ModelMixin,
    get_parameter_dtype,
    register_to_config,
)
from .outputs import MiniMaxH3TransformerOutput

# 0 = video, 1 = text, 2 = audio.
MINIMAX_H3_MODALITY_NUM = 3


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Sinusoidal embedding used by the official MiniMax-H3 checkpoint."""

    if timesteps.ndim != 1:
        raise ValueError(
            f"timesteps must be one-dimensional, got {list(timesteps.shape)}"
        )
    half_dim = embedding_dim // 2
    denominator = half_dim - downscale_freq_shift
    if denominator <= 0:
        raise ValueError("embedding_dim is too small for downscale_freq_shift")
    exponent = -math.log(max_period) * torch.arange(
        half_dim, device=timesteps.device, dtype=torch.float32
    )
    exponent = exponent / denominator
    frequencies = torch.exp(exponent)
    angles = timesteps.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat(
            (embedding[:, half_dim:], embedding[:, :half_dim]), dim=-1
        )
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class Timesteps(nn.Module):
    """Parameter-free timestep projection with official call semantics."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.flip_sin_to_cos = bool(flip_sin_to_cos)
        self.downscale_freq_shift = float(downscale_freq_shift)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class TimestepEmbedding(nn.Module):
    """Two-layer SiLU timestep MLP with official parameter names."""

    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


def _apply_rotary_emb(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate the leading MM-RoPE channels and pass the remainder through."""

    rotary_dim = cos.shape[-1]
    if rotary_dim > hidden_states.shape[-1]:
        raise ValueError(
            f"rotary dimension {rotary_dim} exceeds head dimension "
            f"{hidden_states.shape[-1]}"
        )
    if rotary_dim % 2:
        raise ValueError("rotary dimension must be even")
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]
    cos = cos.to(device=hidden_states.device, dtype=hidden_states.dtype)[
        None, :, None, :
    ]
    sin = sin.to(device=hidden_states.device, dtype=hidden_states.dtype)[
        None, :, None, :
    ]
    first, second = hidden_states_rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    rotary = hidden_states_rotary * cos + rotated * sin
    return torch.cat((rotary, hidden_states_pass), dim=-1).contiguous()


class MiniMaxH3RotaryPosEmbed(nn.Module):
    """Three-axis rotary embedding over packed ``(time, height, width)`` rows."""

    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10_000.0):
        super().__init__()
        if rope_freq_dim <= 0:
            raise ValueError("rope_freq_dim must be positive")
        if rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        self.rope_freq_dim = int(rope_freq_dim)
        self.rope_theta = float(rope_theta)
        self.register_buffer(
            "inv_freq", self._make_inv_freq(torch.device("cpu")), persistent=False
        )

    def _make_inv_freq(self, device: torch.device) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    2 * self.rope_freq_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / (2 * self.rope_freq_dim)
            )
        )

    def materialize(self, device: torch.device | str) -> None:
        self.inv_freq = self._make_inv_freq(torch.device(device))

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                "position_ids must have shape (sequence_length, 3), got "
                f"{list(position_ids.shape)}"
            )
        position_ids = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)
        frequencies = position_ids.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        time, height, width = frequencies.unbind(dim=1)
        frequencies = torch.cat((time, height, width), dim=-1)
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies.cos(), frequencies.sin()


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    """Six AdaLN parameters for each timestep/modality combination."""

    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(
            time_embed_dim,
            6 * hidden_size * MINIMAX_H3_MODALITY_NUM,
            bias=True,
        )

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        parameter_dtype = get_parameter_dtype(self.linear)
        projected = self.linear(F.silu(temb).to(parameter_dtype))
        projected = projected.view(-1, 6 * self.hidden_size)
        return projected.chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    """Final per-row RMSNorm with timestep-selected shift and scale."""

    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.linear(
            F.silu(temb).to(get_parameter_dtype(self.linear))
        ).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        selected_scale = scale.index_select(0, timestep_indices)
        selected_shift = shift.index_select(0, timestep_indices)
        return hidden_states * (1.0 + selected_scale) + selected_shift


class MiniMaxH3AttnProcessor:
    """Full non-causal self-attention implemented with PyTorch SDPA."""

    def __call__(
        self,
        attn: MiniMaxH3Attention,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, attn.head_dim))
        key = key.unflatten(-1, (attn.heads, attn.head_dim))
        value = value.unflatten(-1, (attn.heads, attn.head_dim))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=query.device)
            if attention_mask.ndim == 2 and attention_mask.shape == (
                hidden_states.shape[0],
                hidden_states.shape[1],
            ):
                attention_mask = attention_mask[:, None, None, :].to(torch.bool)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        return attn.to_out[1](output)


class MiniMaxH3Attention(nn.Module):
    """Checkpoint-compatible Q/K/V attention module."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dim_head: int,
        qk_norm_eps: float = 1e-5,
        processor: MiniMaxH3AttnProcessor | None = None,
    ):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(dim_head)
        self.inner_dim = self.heads * self.head_dim
        self.use_bias = False
        self.to_q = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, hidden_size, bias=False), nn.Dropout(0.0)]
        )
        self.processor = processor or MiniMaxH3AttnProcessor()

    def get_processor(self) -> MiniMaxH3AttnProcessor:
        return self.processor

    def set_processor(self, processor: MiniMaxH3AttnProcessor) -> None:
        if not callable(processor):
            raise TypeError("attention processor must be callable")
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb, attention_mask)


class SwiGLU(nn.Module):
    """Fused ``value * SiLU(gate)`` projection with official key layout."""

    def __init__(self, hidden_size: int, inner_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 2 * inner_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return value * F.silu(gate)


class MiniMaxH3FeedForward(nn.Module):
    """SwiGLU feed-forward whose state-dict keys match Diffusers ``FeedForward``."""

    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.net = nn.ModuleList(
            [
                SwiGLU(hidden_size, ffn_dim),
                nn.Dropout(0.0),
                nn.Linear(ffn_dim, hidden_size, bias=False),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Plain pre-norm transformer block for the projected text rows."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(hidden_size, ffn_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.ff(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ):
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_dim,
                    norm_eps,
                    qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)
        self.gradient_checkpointing = False
        self.gradient_checkpointing_kwargs: dict[str, Any] = {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        for block in self.refiner_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                options = dict(self.gradient_checkpointing_kwargs)
                options.setdefault("use_reentrant", False)
                hidden_states = checkpoint(block, hidden_states, **options)
            else:
                hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    """AdaLN-modulated packed-sequence transformer block."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = MiniMaxH3FeedForward(hidden_size, ffn_dim)
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(time_embed_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_proj(temb)

        residual = hidden_states
        normalized = self.norm1(hidden_states)
        normalized = normalized * (
            1.0 + scale_msa.index_select(0, adaln_indices)
        ) + shift_msa.index_select(0, adaln_indices)
        hidden_states = residual + gate_msa.index_select(0, adaln_indices) * self.attn(
            normalized, rotary_emb, attention_mask
        )

        residual = hidden_states
        normalized = self.norm2(hidden_states)
        normalized = normalized * (
            1.0 + scale_mlp.index_select(0, adaln_indices)
        ) + shift_mlp.index_select(0, adaln_indices)
        return residual + gate_mlp.index_select(0, adaln_indices) * self.ff(normalized)


class MiniMaxH3Transformer3DModel(LocalMiniMaxH3ModelMixin, nn.Module):
    """MiniMax-H3's full-attention joint video/audio denoiser.

    The caller builds one packed sequence layout shared across the batch.  Text,
    audio, and patchified video inputs are projected, scattered to that layout,
    processed jointly, and gathered back into the video/audio output streams.
    """

    _supports_gradient_checkpointing = True
    component_name = "transformer"
    _no_split_modules: ClassVar[list[str]] = [
        "MiniMaxH3TransformerBlock",
        "MiniMaxH3TokenRefinerBlock",
        "MiniMaxH3AdaLayerNormOut",
    ]
    _keep_in_fp32_modules: ClassVar[list[str]] = [
        "proj_in",
        "audio_proj_in",
        "time_embedder",
        "proj_out",
        "audio_proj_out",
        "rope",
    ]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        patch_size = tuple(int(value) for value in patch_size)
        if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
            raise ValueError("patch_size must contain three positive integers")
        if (
            min(
                num_attention_heads,
                attention_head_dim,
                hidden_size,
                ffn_dim,
                in_channels,
                audio_in_channels,
                text_dim,
                freq_dim,
                time_embed_hidden_dim,
                time_embed_dim,
            )
            <= 0
        ):
            raise ValueError("MiniMax-H3 dimensions must be positive")
        if num_layers < 0 or num_refiner_layers < 0:
            raise ValueError("layer counts cannot be negative")
        if 6 * rope_freq_dim > attention_head_dim:
            raise ValueError(
                "MM-RoPE rotates 6 * rope_freq_dim channels, which must not "
                "exceed attention_head_dim"
            )

        video_patch_dim = in_channels * math.prod(patch_size)
        self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)
        self.time_proj = Timesteps(freq_dim, True, 0.0)
        self.time_embedder = TimestepEmbedding(
            freq_dim, time_embed_hidden_dim, time_embed_dim
        )
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim, rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            ffn_dim,
            num_refiner_layers,
            norm_eps,
            qk_norm_eps,
            final_norm_eps,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_dim,
                    time_embed_dim,
                    norm_eps,
                    qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size, time_embed_dim, final_norm_eps
        )
        self.proj_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(hidden_size, audio_in_channels, bias=True)
        self.gradient_checkpointing = False

    @classmethod
    def _convert_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        """Accept both current Diffusers and original MiniMax config names."""

        values = dict(config)
        aliases = {
            "token_refiner_num_layers": "num_refiner_layers",
            "ffn_hidden_size": "ffn_dim",
            "latents_dim": "in_channels",
            "audio_latents_dim": "audio_in_channels",
            "timestep_input_dim": "freq_dim",
            "time_embed_hidden_size": "time_embed_hidden_dim",
            "rope_inv_freq_len": "rope_freq_dim",
        }
        for source, target in aliases.items():
            if source in values:
                values.setdefault(target, values.pop(source))
        for derived in ("adaln_out_features", "final_adaln_out_features"):
            values.pop(derived, None)
        return values

    @classmethod
    def _convert_checkpoint_tensor(
        cls, key: str, tensor: torch.Tensor, config: Mapping[str, Any]
    ) -> Mapping[str, torch.Tensor]:
        """Stream-convert original MiniMax/SGLang keys when necessary."""

        if key == "rope.inv_freq":
            return {}
        original_key = key
        if key.startswith("token_refiner.blocks."):
            key = key.replace(
                "token_refiner.blocks.", "token_refiner.refiner_blocks.", 1
            )
        elif key.startswith("blocks."):
            key = key.replace("blocks.", "transformer_blocks.", 1)
        replacements = (
            ("time_embedder.proj_in.", "time_embedder.linear_1."),
            ("time_embedder.proj_out.", "time_embedder.linear_2."),
            ("video_patch_proj.", "proj_in."),
            ("audio_patch_proj.", "audio_proj_in."),
            ("condition_proj.", "context_embedder."),
            ("final_layer.norm.", "norm_out.norm."),
            ("final_layer.adaln_proj.linear.", "norm_out.linear."),
            ("final_layer.video_out.", "proj_out."),
            ("final_layer.audio_out.", "audio_proj_out."),
            (".attn.q_norm.", ".attn.norm_q."),
            (".attn.k_norm.", ".attn.norm_k."),
            (".attn.out_proj.", ".attn.to_out.0."),
            (".mlp.fc2.", ".ff.net.2."),
        )
        for source, target in replacements:
            key = key.replace(source, target)

        if key.endswith(".attn.qkv_proj.weight"):
            heads = int(config["num_attention_heads"])
            head_dim = int(config["attention_head_dim"])
            expected_rows = 3 * heads * head_dim
            if tensor.shape[0] != expected_rows:
                raise ValueError(
                    f"{original_key} has {tensor.shape[0]} rows, expected {expected_rows}"
                )
            # Raw MiniMax shards interleave q/k/v inside every head.
            grouped = tensor.reshape(heads, 3 * head_dim, *tensor.shape[1:])
            query, key_tensor, value = grouped.split(head_dim, dim=1)
            ordered = torch.cat(
                [
                    item.reshape(heads * head_dim, *tensor.shape[1:])
                    for item in (query, key_tensor, value)
                ],
                dim=0,
            )
            query, key_tensor, value = ordered.split(heads * head_dim, dim=0)
            prefix = key.removesuffix("qkv_proj.weight")
            return {
                f"{prefix}to_q.weight": query.contiguous(),
                f"{prefix}to_k.weight": key_tensor.contiguous(),
                f"{prefix}to_v.weight": value.contiguous(),
            }

        if key.endswith(".mlp.fc1.weight"):
            gate, value = tensor.chunk(2, dim=0)
            key = key.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight")
            return {key: torch.cat((value, gate), dim=0).contiguous()}
        return {key: tensor}

    def _materialize_nonpersistent_buffers(self, device: torch.device | str) -> None:
        if self.rope.inv_freq.is_meta:
            self.rope.materialize(device)

    @property
    def attn_processors(self) -> dict[str, MiniMaxH3AttnProcessor]:
        return {
            f"{name}.processor": module.get_processor()
            for name, module in self.named_modules()
            if isinstance(module, MiniMaxH3Attention)
        }

    def set_attn_processor(
        self,
        processor: MiniMaxH3AttnProcessor | Mapping[str, MiniMaxH3AttnProcessor],
    ) -> None:
        processors = self.attn_processors
        if isinstance(processor, Mapping) and set(processor) != set(processors):
            raise ValueError(
                "Processor mapping keys must exactly match model.attn_processors"
            )
        for name, module in self.named_modules():
            if isinstance(module, MiniMaxH3Attention):
                value = (
                    processor[f"{name}.processor"]
                    if isinstance(processor, Mapping)
                    else processor
                )
                module.set_processor(value)

    @staticmethod
    def _validate_indices(
        *,
        sequence_length: int,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        num_video_tokens: int,
        num_audio_tokens: int,
        num_text_tokens: int,
    ) -> None:
        specifications = (
            ("video_indices", video_indices, num_video_tokens),
            ("audio_indices", audio_indices, num_audio_tokens),
            ("text_indices", text_indices, num_text_tokens),
        )
        for name, indices, expected_length in specifications:
            if indices.ndim != 1 or indices.numel() != expected_length:
                raise ValueError(
                    f"{name} must have {expected_length} entries, got "
                    f"shape {list(indices.shape)}"
                )
            if indices.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"{name} must contain integer indices")
            if indices.numel() and (
                int(indices.min()) < 0 or int(indices.max()) >= sequence_length
            ):
                raise IndexError(f"{name} contains an out-of-range packed position")
        combined = torch.cat((video_indices, audio_indices, text_indices))
        if (
            combined.numel() != sequence_length
            or torch.unique(combined).numel() != sequence_length
        ):
            raise ValueError(
                "video/audio/text indices must form a disjoint partition of the "
                "packed sequence"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        """Predict video and audio data-ward velocities for one packed layout."""

        if (
            hidden_states.ndim != 3
            or audio_hidden_states.ndim != 3
            or encoder_hidden_states.ndim != 3
        ):
            raise ValueError("all three modality inputs must be rank-three tensors")
        batch_size = hidden_states.shape[0]
        if (
            audio_hidden_states.shape[0] != batch_size
            or encoder_hidden_states.shape[0] != batch_size
        ):
            raise ValueError("all modality inputs must have the same batch size")
        expected_video_dim = self.proj_in.in_features
        if hidden_states.shape[-1] != expected_video_dim:
            raise ValueError(
                f"video rows must have {expected_video_dim} channels after patchifying"
            )
        if audio_hidden_states.shape[-1] != self.audio_proj_in.in_features:
            raise ValueError("audio latent channel dimension does not match config")
        if encoder_hidden_states.shape[-1] != self.context_embedder.in_features:
            raise ValueError("text encoder hidden dimension does not match config")
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"position_ids must be (seq_len, 3), got {list(position_ids.shape)}"
            )
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (
            sequence_length,
        ):
            raise ValueError(
                "token_tags and timestep_indices must both match the packed sequence"
            )
        if token_tags.dtype not in (
            torch.int32,
            torch.int64,
        ) or timestep_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_tags and timestep_indices must be integer tensors")
        if token_tags.numel() and (
            int(token_tags.min()) < 0
            or int(token_tags.max()) >= MINIMAX_H3_MODALITY_NUM
        ):
            raise ValueError(
                "token_tags values must be 0 (video), 1 (text), or 2 (audio)"
            )
        timestep = timestep.reshape(-1)
        if timestep.numel() == 0:
            raise ValueError("timestep must contain at least one distinct value")
        if timestep_indices.numel() and (
            int(timestep_indices.min()) < 0
            or int(timestep_indices.max()) >= timestep.numel()
        ):
            raise IndexError("timestep_indices addresses a missing timestep row")
        self._validate_indices(
            sequence_length=sequence_length,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            num_video_tokens=hidden_states.shape[1],
            num_audio_tokens=audio_hidden_states.shape[1],
            num_text_tokens=encoder_hidden_states.shape[1],
        )
        if attention_kwargs:
            unsupported = set(attention_kwargs) - {"scale"}
            if unsupported:
                raise TypeError(
                    "Unsupported attention kwargs: " + ", ".join(sorted(unsupported))
                )

        device = hidden_states.device
        structural = (
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
        )
        if any(tensor.device != device for tensor in structural):
            raise ValueError(
                "packed layout tensors must be on the same device as the latents"
            )
        if (
            audio_hidden_states.device != device
            or encoder_hidden_states.device != device
            or timestep.device != device
        ):
            raise ValueError("all model inputs must be on the same device")

        rotary_emb = self.rope(position_ids)
        video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        audio_embeds = self.audio_proj_in(
            audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in))
        )
        text_embeds = self.context_embedder(
            encoder_hidden_states.to(get_parameter_dtype(self.context_embedder))
        )
        text_embeds = self.token_refiner(text_embeds)

        packed = text_embeds.new_zeros(
            (batch_size, sequence_length, text_embeds.shape[-1])
        )
        packed = packed.index_copy(1, text_indices, text_embeds)
        packed = packed.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed = packed.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                cos, sin = rotary_emb

                def custom_forward(
                    states: torch.Tensor,
                    embedding: torch.Tensor,
                    indices: torch.Tensor,
                    rotary_cos: torch.Tensor,
                    rotary_sin: torch.Tensor,
                    current_block: MiniMaxH3TransformerBlock = block,
                ) -> torch.Tensor:
                    return current_block(
                        states,
                        embedding,
                        indices,
                        (rotary_cos, rotary_sin),
                    )

                packed = self._gradient_checkpointing_func(
                    custom_forward, packed, temb, adaln_indices, cos, sin
                )
            else:
                packed = block(packed, temb, adaln_indices, rotary_emb)

        packed = self.norm_out(packed, temb, timestep_indices).to(
            get_parameter_dtype(self.proj_out)
        )
        video_output = self.proj_out(packed).index_select(1, video_indices)
        audio_output = self.audio_proj_out(packed).index_select(1, audio_indices)
        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(
            sample=video_output, audio_sample=audio_output
        )


__all__ = [
    "MINIMAX_H3_MODALITY_NUM",
    "MiniMaxH3AdaLayerNormModulation",
    "MiniMaxH3AdaLayerNormOut",
    "MiniMaxH3Attention",
    "MiniMaxH3AttnProcessor",
    "MiniMaxH3RotaryPosEmbed",
    "MiniMaxH3TokenRefiner",
    "MiniMaxH3TokenRefinerBlock",
    "MiniMaxH3Transformer3DModel",
    "MiniMaxH3TransformerBlock",
    "MiniMaxH3TransformerOutput",
    "TimestepEmbedding",
    "Timesteps",
    "get_timestep_embedding",
]
