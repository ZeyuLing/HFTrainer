"""PyTorch UMT5-compatible encoder used by the local Wan bundle."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .common import BaseModelOutput, LocalWanModelMixin, WanConfig


class UMT5LayerNorm(nn.Module):
    """T5 RMS normalization (no bias and no mean subtraction)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(self.weight.dtype)


def _relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> torch.Tensor:
    relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = relative_position.abs()
    else:
        relative_position = -torch.minimum(
            relative_position, torch.zeros_like(relative_position)
        )
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float().clamp(min=1) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.minimum(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )
    return relative_buckets + torch.where(
        is_small, relative_position, relative_position_if_large
    )


class UMT5Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        dropout_rate: float,
        has_relative_attention_bias: bool,
        relative_attention_num_buckets: int,
        relative_attention_max_distance: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.key_value_proj_dim = d_kv
        self.n_heads = num_heads
        self.inner_dim = num_heads * d_kv
        self.dropout = dropout_rate
        self.q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, d_model, bias=False)
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                relative_attention_num_buckets, num_heads
            )

    def compute_bias(
        self,
        query_length: int,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        context_position = torch.arange(query_length, device=device)[:, None]
        memory_position = torch.arange(key_length, device=device)[None, :]
        relative_position = memory_position - context_position
        buckets = _relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length, _ = hidden_states.shape

        def shape(states: torch.Tensor) -> torch.Tensor:
            return states.view(
                batch_size, sequence_length, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        query = shape(self.q(hidden_states))
        key = shape(self.k(hidden_states))
        value = shape(self.v(hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2))

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    sequence_length,
                    sequence_length,
                    hidden_states.device,
                    scores.dtype,
                )
            else:
                position_bias = torch.zeros(
                    1,
                    self.n_heads,
                    sequence_length,
                    sequence_length,
                    device=hidden_states.device,
                    dtype=scores.dtype,
                )
        scores = scores + position_bias
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                mask = attention_mask[:, None, None, :]
            elif attention_mask.ndim == 3:
                mask = attention_mask[:, None]
            else:
                mask = attention_mask
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attention_weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attention_weights = F.dropout(
            attention_weights, p=self.dropout, training=self.training
        )
        attended = torch.matmul(attention_weights, value)
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.inner_dim
        )
        output = self.o(attended)
        return output, position_bias, attention_weights if output_attentions else None


class UMT5DenseGatedActDense(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, dropout_rate: float, gated: bool = True
    ):
        super().__init__()
        self.gated = gated
        if gated:
            self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
            self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        else:
            self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = dropout_rate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.gated:
            hidden_states = F.gelu(
                self.wi_0(hidden_states), approximate="tanh"
            ) * self.wi_1(hidden_states)
        else:
            hidden_states = F.gelu(self.wi(hidden_states), approximate="tanh")
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return self.wo(hidden_states)


class UMT5LayerSelfAttention(nn.Module):
    def __init__(self, config: WanConfig, has_relative_attention_bias: bool):
        super().__init__()
        self.SelfAttention = UMT5Attention(
            d_model=config.d_model,
            d_kv=config.d_kv,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            has_relative_attention_bias=has_relative_attention_bias,
            relative_attention_num_buckets=config.relative_attention_num_buckets,
            relative_attention_max_distance=config.relative_attention_max_distance,
        )
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = config.dropout_rate

    def forward(
        self, hidden_states, attention_mask, position_bias, output_attentions=False
    ):
        normed = self.layer_norm(hidden_states)
        attention_output, position_bias, attention_weights = self.SelfAttention(
            normed,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + F.dropout(
            attention_output, p=self.dropout, training=self.training
        )
        return hidden_states, position_bias, attention_weights


class UMT5LayerFF(nn.Module):
    def __init__(self, config: WanConfig):
        super().__init__()
        self.DenseReluDense = UMT5DenseGatedActDense(
            config.d_model,
            config.d_ff,
            config.dropout_rate,
            gated=config.is_gated_act,
        )
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = config.dropout_rate

    def forward(self, hidden_states):
        forwarded = self.DenseReluDense(self.layer_norm(hidden_states))
        return hidden_states + F.dropout(
            forwarded, p=self.dropout, training=self.training
        )


class UMT5Block(nn.Module):
    def __init__(self, config: WanConfig, has_relative_attention_bias: bool):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                UMT5LayerSelfAttention(config, has_relative_attention_bias),
                UMT5LayerFF(config),
            ]
        )

    def forward(
        self, hidden_states, attention_mask, position_bias, output_attentions=False
    ):
        hidden_states, position_bias, attention_weights = self.layer[0](
            hidden_states,
            attention_mask,
            position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias, attention_weights


class UMT5Stack(nn.Module):
    def __init__(self, config: WanConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.block = nn.ModuleList(
            [
                UMT5Block(config, has_relative_attention_bias=index == 0)
                for index in range(config.num_layers)
            ]
        )
        self.final_layer_norm = UMT5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = config.dropout_rate


class UMT5EncoderModel(LocalWanModelMixin, nn.Module):
    """Encoder-only UMT5 architecture with official configuration field names."""

    component_name = "text_encoder"

    def __init__(
        self,
        vocab_size: int = 256384,
        d_model: int = 4096,
        d_kv: int = 64,
        d_ff: int = 10240,
        num_layers: int = 24,
        num_heads: int = 64,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        feed_forward_proj: str = "gated-gelu",
        dense_act_fn: str = "gelu_new",
        is_gated_act: bool | None = None,
        initializer_factor: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        is_encoder_decoder: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        super().__init__()
        kwargs.pop("hidden_size", None)
        relative_attention_max_distance = int(
            kwargs.pop("max_distance", relative_attention_max_distance)
        )
        if d_model <= 0 or d_kv <= 0 or d_ff <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("UMT5 dimensions and layer counts must be positive")
        if is_gated_act is None:
            is_gated_act = feed_forward_proj.startswith("gated-")
        self.config = WanConfig(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            hidden_size=int(d_model),
            d_kv=int(d_kv),
            d_ff=int(d_ff),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            relative_attention_num_buckets=int(relative_attention_num_buckets),
            relative_attention_max_distance=int(relative_attention_max_distance),
            dropout_rate=float(dropout_rate),
            layer_norm_epsilon=float(layer_norm_epsilon),
            feed_forward_proj=feed_forward_proj,
            dense_act_fn=dense_act_fn,
            is_gated_act=bool(is_gated_act),
            initializer_factor=float(initializer_factor),
            pad_token_id=int(pad_token_id),
            eos_token_id=int(eos_token_id),
            tie_word_embeddings=bool(tie_word_embeddings),
            is_encoder_decoder=bool(is_encoder_decoder),
            use_cache=bool(use_cache),
            **kwargs,
        )
        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder = UMT5Stack(self.config, self.shared)
        self.gradient_checkpointing = False
        self.post_init()

    def post_init(self):
        factor = self.config.initializer_factor
        nn.init.normal_(self.shared.weight, mean=0.0, std=factor)
        for module in self.modules():
            if isinstance(module, UMT5Attention):
                nn.init.normal_(
                    module.q.weight,
                    mean=0.0,
                    std=factor * (self.config.d_model * self.config.d_kv) ** -0.5,
                )
                nn.init.normal_(
                    module.k.weight, mean=0.0, std=factor * self.config.d_model**-0.5
                )
                nn.init.normal_(
                    module.v.weight, mean=0.0, std=factor * self.config.d_model**-0.5
                )
                nn.init.normal_(
                    module.o.weight,
                    mean=0.0,
                    std=factor * (self.config.num_heads * self.config.d_kv) ** -0.5,
                )
            elif isinstance(module, UMT5DenseGatedActDense):
                if module.gated:
                    nn.init.normal_(
                        module.wi_0.weight, std=factor * self.config.d_model**-0.5
                    )
                    nn.init.normal_(
                        module.wi_1.weight, std=factor * self.config.d_model**-0.5
                    )
                else:
                    nn.init.normal_(
                        module.wi.weight, std=factor * self.config.d_model**-0.5
                    )
                nn.init.normal_(module.wo.weight, std=factor * self.config.d_ff**-0.5)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.shared

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.shared = new_embeddings
        self.encoder.embed_tokens = new_embeddings

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        if kwargs:
            # Accept common encoder-only compatibility kwargs when they are None.
            unsupported = {
                key: value for key, value in kwargs.items() if value is not None
            }
            if unsupported:
                raise TypeError(
                    f"Unsupported UMT5 forward kwargs: {sorted(unsupported)}"
                )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "Exactly one of input_ids or inputs_embeds must be provided"
            )
        if inputs_embeds is None:
            inputs_embeds = self.encoder.embed_tokens(input_ids)
        hidden_states = F.dropout(
            inputs_embeds, p=self.config.dropout_rate, training=self.training
        )
        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2], device=hidden_states.device, dtype=torch.long
            )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None
        for block in self.encoder.block:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if (
                self.gradient_checkpointing
                and self.training
                and hidden_states.requires_grad
            ):

                def custom_forward(states, mask, module=block):
                    return module(states, mask, None, output_attentions=False)[0]

                hidden_states = checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                )
                position_bias = None
                attention_weights = None
            else:
                hidden_states, position_bias, attention_weights = block(
                    hidden_states,
                    attention_mask,
                    position_bias,
                    output_attentions=output_attentions,
                )
            if output_attentions:
                all_attentions += (attention_weights,)

        hidden_states = self.encoder.final_layer_norm(hidden_states)
        hidden_states = F.dropout(
            hidden_states, p=self.config.dropout_rate, training=self.training
        )
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            values = (hidden_states,)
            if output_hidden_states:
                values += (all_hidden_states,)
            if output_attentions:
                values += (all_attentions,)
            return values
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
