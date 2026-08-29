# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Local Gemma text backbone used by LTX-Video 2.5.

This is a compact PyTorch implementation of the text-only execution path used
by the packed LTX Gemma checkpoints.  It preserves the checkpoint module names
and hidden-state contract required by the LTX embedding processor while
keeping model construction inside HFTrainer.  Multimodal prompt enhancement is
an optional pipeline concern; normal text-to-video encoding never requires a
second model framework.

The architecture follows Google's Gemma family and the Apache-2.0 reference
implementation identified in ``THIRD_PARTY_NOTICES.md``.  This file is a
substantial HFTrainer rewrite, not an unmodified upstream module.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGemmaConfig:
    """Attribute-accessible, recursively parsed Gemma configuration."""

    def __init__(self, **values: Any):
        for name, value in values.items():
            if isinstance(value, Mapping):
                value = LocalGemmaConfig(**value)
            elif isinstance(value, list):
                value = [LocalGemmaConfig(**item) if isinstance(item, Mapping) else item for item in value]
            setattr(self, name, value)
        self.model_type = getattr(self, 'model_type', 'gemma4_unified')
        if hasattr(self, 'text_config'):
            text = self.text_config
        else:
            text = self
        _set_text_defaults(text)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> 'LocalGemmaConfig':
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        def convert(value):
            if isinstance(value, LocalGemmaConfig):
                return {name: convert(item) for name, item in vars(value).items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(self)

    def get_text_config(self):
        return getattr(self, 'text_config', self)


def _set_default(config: LocalGemmaConfig, name: str, value: Any) -> None:
    if not hasattr(config, name):
        setattr(config, name, value)


def _set_text_defaults(config: LocalGemmaConfig) -> None:
    defaults = {
        'vocab_size': 262_144,
        'hidden_size': 2304,
        'intermediate_size': 9216,
        'num_hidden_layers': 30,
        'num_attention_heads': 8,
        'num_key_value_heads': 4,
        'head_dim': 256,
        'global_head_dim': 512,
        'num_global_key_value_heads': None,
        'hidden_activation': 'gelu_pytorch_tanh',
        'max_position_embeddings': 262_144,
        'rms_norm_eps': 1e-6,
        'pad_token_id': 0,
        'eos_token_id': 1,
        'bos_token_id': 2,
        'attention_bias': False,
        'attention_dropout': 0.0,
        'sliding_window': 1024,
        'attention_k_eq_v': False,
        'num_kv_shared_layers': 0,
        'use_double_wide_mlp': False,
        'use_bidirectional_attention': 'vision',
        'final_logit_softcapping': None,
    }
    for name, value in defaults.items():
        _set_default(config, name, value)
    if getattr(config, 'num_global_key_value_heads', None) is None:
        config.num_global_key_value_heads = config.num_key_value_heads
    if not hasattr(config, 'layer_types') or config.layer_types is None:
        config.layer_types = [
            'sliding_attention' if (index + 1) % 6 else 'full_attention'
            for index in range(config.num_hidden_layers)
        ]
        config.layer_types[-1] = 'full_attention'
    if not hasattr(config, 'rope_parameters') or config.rope_parameters is None:
        config.rope_parameters = {
            'sliding_attention': {'rope_type': 'default', 'rope_theta': 10_000.0},
            'full_attention': {
                'rope_type': 'proportional',
                'partial_rotary_factor': 0.25,
                'rope_theta': 1_000_000.0,
            },
        }


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            normalized = normalized * self.weight.float()
        return normalized.to(hidden_states.dtype)


class ScaledEmbedding(nn.Embedding):
    def __init__(self, count: int, dim: int, padding_idx: int | None):
        super().__init__(count, dim, padding_idx=padding_idx)
        self.register_buffer('embed_scale', torch.tensor(dim**0.5), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


def _rope_frequencies(config: LocalGemmaConfig, layer_type: str) -> torch.Tensor:
    params = config.rope_parameters[layer_type]
    head_dim = config.global_head_dim if layer_type == 'full_attention' else config.head_dim
    partial = float(params.get('partial_rotary_factor', 1.0))
    rotated = int(partial * head_dim // 2)
    theta = float(params.get('rope_theta', 10_000.0))
    inv = 1.0 / (theta ** (torch.arange(0, 2 * rotated, 2, dtype=torch.float32) / head_dim))
    if rotated < head_dim // 2:
        inv = torch.cat([inv, torch.zeros(head_dim // 2 - rotated)])
    return inv / float(params.get('factor', 1.0))


class RotaryEmbedding(nn.Module):
    def __init__(self, config: LocalGemmaConfig):
        super().__init__()
        self.config = config
        for layer_type in set(config.layer_types):
            self.register_buffer(
                f'{layer_type}_inv_freq',
                _rope_frequencies(config, layer_type),
                persistent=False,
            )

    def forward(self, position_ids: torch.Tensor, layer_type: str, dtype: torch.dtype):
        inv = getattr(self, f'{layer_type}_inv_freq').to(position_ids.device)
        frequencies = position_ids.float().unsqueeze(-1) * inv.view(1, 1, -1)
        embedding = torch.cat([frequencies, frequencies], dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def _apply_rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return value * cos + _rotate_half(value) * sin


def _repeat_kv(value: torch.Tensor, repeat: int) -> torch.Tensor:
    if repeat == 1:
        return value
    batch, length, heads, dim = value.shape
    return value[:, :, :, None, :].expand(batch, length, heads, repeat, dim).reshape(
        batch, length, heads * repeat, dim
    )


class GemmaAttention(nn.Module):
    def __init__(self, config: LocalGemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == 'sliding_attention'
        self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim
        self.alternative_attention = bool(config.attention_k_eq_v and not self.is_sliding)
        self.num_kv_heads = (
            config.num_global_key_value_heads if self.alternative_attention else config.num_key_value_heads
        )
        self.num_kv_groups = config.num_attention_heads // self.num_kv_heads
        first_shared = config.num_hidden_layers - int(config.num_kv_shared_layers)
        self.is_kv_shared_layer = layer_idx >= first_shared > 0
        prior = config.layer_types[:first_shared]
        self.store_full_length_kv = (
            not self.is_kv_shared_layer
            and self.layer_type in prior
            and layer_idx == len(prior) - 1 - prior[::-1].index(self.layer_type)
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)
        if not self.is_kv_shared_layer:
            self.k_proj = nn.Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=config.attention_bias,
            )
            self.v_proj = None if self.alternative_attention else nn.Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=config.attention_bias,
            )
            self.k_norm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)
            self.v_norm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, length, -1, self.head_dim)
        query = _apply_rope(self.q_norm(query), cos, sin)
        if self.is_kv_shared_layer:
            key, value = shared_kv[self.layer_type]
            key, value = key.to(query.device), value.to(query.device)
        else:
            key = self.k_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim)
            value = key if self.v_proj is None else self.v_proj(hidden_states).view(
                batch, length, self.num_kv_heads, self.head_dim
            )
            key = _apply_rope(self.k_norm(key), cos, sin)
            value = self.v_norm(value)
            if self.store_full_length_kv:
                shared_kv[self.layer_type] = (key, value)
        key = _repeat_kv(key, self.num_kv_groups)
        value = _repeat_kv(value, self.num_kv_groups)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) + attention_mask
        probabilities = F.softmax(scores.float(), dim=-1).to(query.dtype)
        probabilities = F.dropout(
            probabilities,
            p=float(self.config.attention_dropout),
            training=self.training,
        )
        output = torch.matmul(probabilities, value).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output)


class GemmaMLP(nn.Module):
    def __init__(self, config: LocalGemmaConfig, layer_idx: int):
        super().__init__()
        first_shared = config.num_hidden_layers - int(config.num_kv_shared_layers)
        wide = bool(config.use_double_wide_mlp and layer_idx >= first_shared > 0)
        intermediate = config.intermediate_size * (2 if wide else 1)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)
        self.activation = config.hidden_activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        if self.activation in {'gelu_pytorch_tanh', 'gelu_new'}:
            gate = F.gelu(gate, approximate='tanh')
        elif self.activation in {'silu', 'swish'}:
            gate = F.silu(gate)
        else:
            gate = F.gelu(gate)
        return self.down_proj(gate * self.up_proj(hidden_states))


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: LocalGemmaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config, layer_idx)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.register_buffer('layer_scalar', torch.ones(1))

    def forward(self, hidden_states, cos, sin, mask, shared_kv):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), cos, sin, mask, shared_kv
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return (residual + self.post_feedforward_layernorm(hidden_states)) * self.layer_scalar


def _attention_mask(
    padding_mask: torch.Tensor | None,
    length: int,
    *,
    causal: bool,
    sliding_window: int | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    allowed = torch.ones(length, length, dtype=torch.bool, device=device)
    positions = torch.arange(length, device=device)
    if causal:
        allowed &= positions[:, None] >= positions[None, :]
    if sliding_window is not None:
        allowed &= positions[:, None] - positions[None, :] < sliding_window
    if padding_mask is None:
        padding_mask = torch.ones(1, length, dtype=torch.bool, device=device)
    else:
        padding_mask = padding_mask.to(device=device, dtype=torch.bool)
    allowed = allowed[None, None] & padding_mask[:, None, None, :]
    minimum = torch.finfo(dtype).min
    return torch.where(allowed, torch.zeros((), dtype=dtype, device=device), minimum)


@dataclass
class LocalGemmaOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


class LocalGemmaLanguageModel(nn.Module):
    def __init__(self, config: LocalGemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = ScaledEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            GemmaDecoderLayer(config, index) for index in range(config.num_hidden_layers)
        )
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> LocalGemmaOutput:
        del kwargs
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('Specify exactly one of input_ids or inputs_embeds.')
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length, _ = hidden.shape
        positions = torch.arange(length, device=hidden.device).unsqueeze(0).expand(batch, -1)
        masks = {
            layer_type: _attention_mask(
                attention_mask,
                length,
                causal=self.config.use_bidirectional_attention != 'all',
                sliding_window=(self.config.sliding_window if layer_type == 'sliding_attention' else None),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            for layer_type in set(self.config.layer_types)
        }
        rope = {
            layer_type: self.rotary_emb(positions, layer_type, hidden.dtype)
            for layer_type in set(self.config.layer_types)
        }
        states = [hidden] if output_hidden_states else []
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for index, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[index]
            cos, sin = rope[layer_type]
            hidden = layer(hidden, cos, sin, masks[layer_type], shared_kv)
            if output_hidden_states:
                states.append(hidden)
        hidden = self.norm(hidden)
        if states:
            states[-1] = hidden
        return LocalGemmaOutput(hidden, tuple(states))


class LocalGemmaModel(nn.Module):
    def __init__(self, config: LocalGemmaConfig):
        super().__init__()
        self.language_model = LocalGemmaLanguageModel(config.get_text_config())

    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)


class LocalGemmaForConditionalGeneration(nn.Module):
    """Checkpoint-compatible text-only Gemma conditional-generation shell."""

    _no_split_modules = ['GemmaDecoderLayer']

    def __init__(self, config: LocalGemmaConfig):
        super().__init__()
        self.config = config
        self.model = LocalGemmaModel(config)
        text = config.get_text_config()
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        self.tie_weights()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def tie_weights(self):
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True, **kwargs):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        return SimpleNamespace(
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states,
            logits=self.lm_head(output.last_hidden_state),
        )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=32, **kwargs):
        temperature = float(kwargs.get('temperature', 1.0))
        do_sample = bool(kwargs.get('do_sample', False))
        eos = self.config.get_text_config().eos_token_id
        ids = input_ids
        mask = attention_mask
        for _ in range(int(max_new_tokens)):
            output = self.forward(ids, attention_mask=mask)
            logits = output.logits[:, -1] / max(temperature, 1e-6)
            token = (
                torch.multinomial(F.softmax(logits.float(), -1), 1)
                if do_sample
                else logits.argmax(-1, keepdim=True)
            )
            ids = torch.cat([ids, token], dim=1)
            if mask is not None:
                mask = torch.cat([mask, torch.ones_like(token)], dim=1)
            if isinstance(eos, int) and bool((token == eos).all()):
                break
        return ids


def build_local_gemma_model(config: LocalGemmaConfig | Mapping[str, Any]):
    if not isinstance(config, LocalGemmaConfig):
        config = LocalGemmaConfig.from_dict(config)
    if config.model_type not in {'gemma4_unified', 'gemma4', 'gemma3'}:
        raise ValueError(f'Unsupported local Gemma model_type: {config.model_type!r}')
    return LocalGemmaForConditionalGeneration(config)


def default_rope_parameters(config: LocalGemmaConfig, *, layer_type: str, **kwargs):
    del kwargs
    return _rope_frequencies(config, layer_type), 1.0


def proportional_rope_parameters(
    config: LocalGemmaConfig,
    *,
    layer_type: str,
    head_dim_key: str = 'head_dim',
    **kwargs,
):
    del head_dim_key, kwargs
    return _rope_frequencies(config, layer_type), 1.0


ROPE_INIT_FUNCTIONS = {
    'default': default_rope_parameters,
    'proportional': proportional_rope_parameters,
}


__all__ = [
    'LocalGemmaConfig',
    'LocalGemmaForConditionalGeneration',
    'ROPE_INIT_FUNCTIONS',
    'build_local_gemma_model',
]
