"""Local CLIP text tower used for Stable Diffusion conditioning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hftrainer.registry import MODEL_COMPONENTS

from ..checkpoint import LocalComponentMixin
from .configuration import ConfigDict
from .outputs import TextEncoderOutput


class QuickGELU(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(1.702 * value)


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.register_buffer(
            'position_ids',
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError('Either input_ids or inputs_embeds must be provided.')
            inputs_embeds = self.token_embedding(input_ids)
        sequence_length = inputs_embeds.shape[-2]
        if position_ids is None:
            position_ids = self.position_ids[:, :sequence_length]
        return inputs_embeds + self.position_embedding(position_ids)


class CLIPAttention(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        if self.embed_dim % self.num_heads:
            raise ValueError('hidden_size must be divisible by num_attention_heads.')
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = float(config.attention_dropout)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = tensor.shape
        return tensor.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query = self._shape(self.q_proj(hidden_states)) * self.scale
        key = self._shape(self.k_proj(hidden_states))
        value = self._shape(self.v_proj(hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2))
        if causal_attention_mask is not None:
            scores = scores + causal_attention_mask.to(scores.dtype)
        if attention_mask is not None:
            scores = scores + attention_mask.to(scores.dtype)
        probabilities = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).reshape(hidden_states.shape)
        return self.out_proj(output), probabilities


class CLIPMLP(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation_fn = (
            QuickGELU() if config.hidden_act == 'quick_gelu' else nn.GELU()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        return residual + self.mlp(hidden_states)


class CLIPEncoder(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.layers = nn.ModuleList(
            [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_attention_mask=causal_attention_mask,
                )
        return hidden_states


class CLIPTextTransformer(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.config = config
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> TextEncoderOutput:
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        batch, sequence = hidden_states.shape[:2]
        min_value = torch.finfo(hidden_states.dtype).min
        causal_mask = torch.full(
            (sequence, sequence),
            min_value,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)[None, None, :, :]
        if attention_mask is not None:
            expanded = attention_mask[:, None, None, :].to(hidden_states.dtype)
            attention_mask = (1.0 - expanded) * min_value
        hidden_states = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_mask,
        )
        hidden_states = self.final_layer_norm(hidden_states)
        if input_ids is not None:
            pooled_indices = input_ids.to(torch.int32).argmax(dim=-1)
            pooled = hidden_states[torch.arange(batch, device=hidden_states.device), pooled_indices]
        else:
            pooled = hidden_states[:, -1]
        return TextEncoderOutput(last_hidden_state=hidden_states, pooler_output=pooled)


@MODEL_COMPONENTS.register_module(name='CLIPTextModel', force=True)
class CLIPTextModel(LocalComponentMixin, nn.Module):
    """SD1.5-compatible CLIP text encoder implemented with ordinary torch layers."""

    component_kind = 'clip_text_encoder'

    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        projection_dim: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 77,
        hidden_act: str = 'quick_gelu',
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        initializer_factor: float = 1.0,
        initializer_range: float = 0.02,
        pad_token_id: int = 1,
        bos_token_id: int = 49406,
        eos_token_id: int = 49407,
        **metadata: Any,
    ):
        nn.Module.__init__(self)
        self.config = ConfigDict(
            vocab_size=int(vocab_size),
            hidden_size=int(hidden_size),
            intermediate_size=int(intermediate_size),
            projection_dim=int(projection_dim),
            num_hidden_layers=int(num_hidden_layers),
            num_attention_heads=int(num_attention_heads),
            max_position_embeddings=int(max_position_embeddings),
            hidden_act=hidden_act,
            layer_norm_eps=float(layer_norm_eps),
            attention_dropout=float(attention_dropout),
            dropout=float(dropout),
            initializer_factor=float(initializer_factor),
            initializer_range=float(initializer_range),
            pad_token_id=int(pad_token_id),
            bos_token_id=int(bos_token_id),
            eos_token_id=int(eos_token_id),
            **metadata,
        )
        self.text_model = CLIPTextTransformer(self.config)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def gradient_checkpointing_enable(self, **_):
        self.text_model.encoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.text_model.encoder.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_dict: bool = True,
        **_,
    ):
        output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return output if return_dict else (output.last_hidden_state, output.pooler_output)
