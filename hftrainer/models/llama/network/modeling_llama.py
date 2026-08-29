"""Pure-PyTorch LLaMA implementation owned by HFTrainer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hftrainer.models.llama.checkpoint import load_state_dict, save_state_dict
from hftrainer.models.llama.configuration import LlamaConfig
from hftrainer.registry import MODEL_COMPONENTS


PastKeyValues = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: torch.Tensor
    past_key_values: Optional[PastKeyValues] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None

    def to_tuple(self) -> tuple:
        return tuple(value for value in (
            self.last_hidden_state, self.past_key_values,
            self.hidden_states, self.attentions,
        ) if value is not None)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        return self.to_tuple()[item]


@dataclass
class CausalLMOutputWithPast:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[PastKeyValues] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None

    def to_tuple(self) -> tuple:
        return tuple(value for value in (
            self.loss, self.logits, self.past_key_values,
            self.hidden_states, self.attentions,
        ) if value is not None)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        return self.to_tuple()[item]


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * values.to(dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        factor = 1.0
        if config.rope_scaling:
            scaling_type = config.rope_scaling.get('rope_type', config.rope_scaling.get('type'))
            if scaling_type != 'linear':
                raise ValueError(
                    f'Unsupported rope_scaling {scaling_type!r}; local LLaMA currently supports linear scaling.'
                )
            factor = float(config.rope_scaling.get('factor', 1.0))
            if factor <= 0:
                raise ValueError('rope_scaling.factor must be positive.')
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.factor = factor

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        positions = position_ids.float() / self.factor
        frequencies = positions.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    half = tensor.shape[-1] // 2
    return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def _repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return states
    batch, heads, length, dim = states.shape
    return (
        states[:, :, None, :, :]
        .expand(batch, heads, groups, length, dim)
        .reshape(batch, heads * groups, length, dim)
    )


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = {'silu': F.silu, 'gelu': F.gelu, 'relu': F.relu}[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        batch, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        cosine, sine = self.rotary_emb(position_ids, query.dtype)
        query, key = _apply_rotary(query, key, cosine, sine)
        past_length = 0
        if past_key_value is not None:
            past_length = past_key_value[0].shape[-2]
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)
        present = (key, value) if use_cache else None
        repeated_key = _repeat_kv(key, self.num_key_value_groups)
        repeated_value = _repeat_kv(value, self.num_key_value_groups)
        scores = torch.matmul(query, repeated_key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        key_length = repeated_key.shape[-2]
        query_positions = past_length + torch.arange(query_length, device=hidden_states.device)
        key_positions = torch.arange(key_length, device=hidden_states.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~allowed.view(1, 1, query_length, key_length), torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (batch, key_length):
                raise ValueError(
                    f'attention_mask must be {(batch, key_length)}, got {tuple(attention_mask.shape)}.'
                )
            scores = scores.masked_fill(
                attention_mask[:, None, None, :].eq(0), torch.finfo(scores.dtype).min
            )
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        probabilities = F.dropout(probabilities, p=self.attention_dropout, training=self.training)
        output = torch.matmul(probabilities, repeated_value)
        output = output.transpose(1, 2).contiguous().view(
            batch, query_length, self.num_heads * self.head_dim
        )
        return self.o_proj(output), present, probabilities if output_attentions else None


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        attention, present, probabilities = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
            output_attentions,
        )
        hidden_states = residual + attention
        return (
            hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)),
            present,
            probabilities,
        )


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, index) for index in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[PastKeyValues] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('Provide exactly one of input_ids or inputs_embeds.')
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, query_length, _ = hidden_states.shape
        past_length = 0 if not past_key_values else past_key_values[0][0].shape[-2]
        key_length = past_length + query_length
        if attention_mask is None:
            attention_mask = torch.ones(
                batch, key_length, dtype=torch.long, device=hidden_states.device
            )
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + query_length,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0)
        use_cache = self.config.use_cache if use_cache is None else bool(use_cache)
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False
        all_hidden = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None
        for index, layer in enumerate(self.layers):
            if all_hidden is not None:
                all_hidden += (hidden_states,)
            past = past_key_values[index] if past_key_values else None
            hidden_states, present, probabilities = layer(
                hidden_states,
                attention_mask,
                position_ids,
                past,
                use_cache,
                output_attentions,
            )
            if next_cache is not None:
                next_cache += (present,)
            if all_attentions is not None:
                all_attentions += (probabilities,)
        hidden_states = self.norm(hidden_states)
        if all_hidden is not None:
            all_hidden += (hidden_states,)
        output = BaseModelOutputWithPast(
            hidden_states, next_cache, all_hidden, all_attentions
        )
        return output if return_dict else output.to_tuple()


@MODEL_COMPONENTS.register_module()
class LocalLlamaForCausalLM(nn.Module):
    """Local LLaMA decoder with training, generation and artifact I/O."""

    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig | dict | None = None, **config_kwargs: Any):
        super().__init__()
        if config is None:
            config = LlamaConfig.from_dict(config_kwargs)
        elif isinstance(config, dict):
            config = LlamaConfig.from_dict(config, **config_kwargs)
        elif config_kwargs:
            config = LlamaConfig.from_dict(config.to_dict(), **config_kwargs)
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.tie_weights()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        del kwargs
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.model.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[PastKeyValues] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ):
        if kwargs:
            raise TypeError(f'Unsupported local LLaMA forward options: {sorted(kwargs)}')
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            if labels.shape != logits.shape[:2]:
                raise ValueError(
                    f'labels must have shape {tuple(logits.shape[:2])}, got {tuple(labels.shape)}.'
                )
            shift_logits = logits[:, :-1].contiguous().float()
            shift_labels = labels[:, 1:].contiguous().to(logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        output = CausalLMOutputWithPast(
            loss, logits, outputs.past_key_values,
            outputs.hidden_states, outputs.attentions,
        )
        return output if return_dict else output.to_tuple()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if kwargs:
            raise TypeError(f'Unsupported local generation options: {sorted(kwargs)}')
        if max_new_tokens < 0:
            raise ValueError('max_new_tokens must be non-negative.')
        if do_sample and temperature <= 0:
            raise ValueError('temperature must be positive when sampling.')
        sequences = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(sequences)
        else:
            attention_mask = attention_mask.clone()
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        pad = self.config.pad_token_id if pad_token_id is None else pad_token_id
        if pad is None:
            pad = eos if eos is not None else 0
        finished = torch.zeros(sequences.shape[0], dtype=torch.bool, device=sequences.device)
        for _ in range(max_new_tokens):
            outputs = self(
                input_ids=sequences,
                attention_mask=attention_mask,
                position_ids=attention_mask.long().cumsum(-1).sub(1).clamp_min(0),
                use_cache=False,
            )
            column_ids = torch.arange(
                attention_mask.shape[1], device=attention_mask.device
            ).expand_as(attention_mask)
            last_positions = column_ids.masked_fill(attention_mask.eq(0), -1).max(-1).values
            last_positions = last_positions.clamp_min(0)
            logits = outputs.logits[
                torch.arange(sequences.shape[0], device=sequences.device), last_positions
            ]
            if do_sample:
                logits = logits / temperature
                if top_k is not None and 0 < top_k < logits.shape[-1]:
                    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
                    logits = torch.full_like(logits, float('-inf')).scatter(
                        -1, sorted_indices, sorted_logits
                    )
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, pad), next_token)
            sequences = torch.cat((sequences, next_token[:, None]), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, (~finished).to(attention_mask.dtype)[:, None]), dim=-1
            )
            if eos is not None:
                finished |= next_token.eq(eos)
                if finished.all():
                    break
        return sequences

    @classmethod
    def from_config(cls, config: LlamaConfig | dict | None = None, **kwargs: Any):
        return cls(config=config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: Any = None,
        output_loading_info: bool = False,
        **kwargs: Any,
    ):
        for ignored in ('local_files_only', 'low_cpu_mem_usage'):
            kwargs.pop(ignored, None)
        if kwargs:
            raise TypeError(f'Unsupported local LLaMA load options: {sorted(kwargs)}')
        config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        state = load_state_dict(pretrained_model_name_or_path)
        state = {
            key: value for key, value in state.items()
            if not key.endswith('rotary_emb.inv_freq')
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = {'lm_head.weight'} if config.tie_word_embeddings else set()
        real_missing = [key for key in missing if key not in allowed_missing]
        if real_missing or unexpected:
            raise RuntimeError(
                f'LLaMA checkpoint mismatch: missing={real_missing[:8]}, unexpected={unexpected[:8]}.'
            )
        if config.tie_word_embeddings:
            model.tie_weights()
        if torch_dtype not in (None, 'auto'):
            dtype = (
                getattr(torch, torch_dtype.removeprefix('torch.'))
                if isinstance(torch_dtype, str) else torch_dtype
            )
            model.to(dtype=dtype)
        info = {'missing_keys': real_missing, 'unexpected_keys': list(unexpected)}
        return (model, info) if output_loading_info else model

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise TypeError(f'Unsupported local LLaMA save options: {sorted(kwargs)}')
        self.config.save_pretrained(save_directory)
        save_state_dict(self.state_dict(), save_directory, safe_serialization)


LlamaForCausalLM = LocalLlamaForCausalLM


__all__ = [
    'BaseModelOutputWithPast',
    'CausalLMOutputWithPast',
    'LlamaForCausalLM',
    'LlamaModel',
    'LlamaRMSNorm',
    'LocalLlamaForCausalLM',
]
