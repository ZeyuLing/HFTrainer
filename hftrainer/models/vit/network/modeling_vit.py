"""Pure-PyTorch Vision Transformer with published ViT checkpoint key layout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hftrainer.models.vit.checkpoint import load_state_dict, save_state_dict
from hftrainer.models.vit.configuration import ViTConfig
from hftrainer.registry import MODEL_COMPONENTS


@dataclass
class ImageClassifierOutput:
    """Minimal attribute and tuple compatible image-classifier output."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None

    def to_tuple(self) -> tuple:
        return tuple(value for value in (
            self.loss, self.logits, self.hidden_states, self.attentions
        ) if value is not None)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        return self.to_tuple()[item]


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(f'pixel_values must be BCHW, got {tuple(pixel_values.shape)}')
        if pixel_values.shape[1] != self.projection.in_channels:
            raise ValueError(
                f'Expected {self.projection.in_channels} channels, got {pixel_values.shape[1]}.'
            )
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class ViTEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.patch_embeddings.num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        patch = self.patch_embeddings.patch_size
        grid_h, grid_w = height // patch, width // patch
        expected = self.patch_embeddings.num_patches
        if grid_h * grid_w == expected:
            return self.position_embeddings
        source = int(math.sqrt(expected))
        if source * source != expected:
            raise ValueError('Cannot interpolate non-square pretrained position embeddings.')
        cls_position = self.position_embeddings[:, :1]
        patches = self.position_embeddings[:, 1:].reshape(1, source, source, -1)
        patches = patches.permute(0, 3, 1, 2)
        patches = F.interpolate(patches, size=(grid_h, grid_w), mode='bicubic', align_corners=False)
        patches = patches.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
        return torch.cat((cls_position, patches), dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(pixel_values.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self._position_embedding(
            pixel_values.shape[-2], pixel_values.shape[-1]
        ).to(embeddings)
        return self.dropout(embeddings)


class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _transpose(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        return tensor.view(shape).permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: torch.Tensor, output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        query = self._transpose(self.query(hidden_states))
        key = self._transpose(self.key(hidden_states))
        value = self._transpose(self.value(hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        probabilities = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(probabilities, value).permute(0, 2, 1, 3).contiguous()
        context = context.view(hidden_states.shape[0], hidden_states.shape[1], self.all_head_size)
        return context, probabilities if output_attentions else None


class ViTSelfOutput(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states))


class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False):
        context, probabilities = self.attention(hidden_states, output_attentions)
        return self.output(context), probabilities


class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        activations = {
            'gelu': F.gelu,
            'gelu_new': lambda x: F.gelu(x, approximate='tanh'),
            'relu': F.relu,
        }
        if config.hidden_act not in activations:
            raise ValueError(f'Unsupported ViT activation: {config.hidden_act}')
        self.intermediate_act_fn = activations[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states))


class ViTLayer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False):
        attention, probabilities = self.attention(
            self.layernorm_before(hidden_states), output_attentions
        )
        hidden_states = hidden_states + attention
        feed_forward = self.output(self.intermediate(self.layernorm_after(hidden_states)))
        return hidden_states + feed_forward, probabilities


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        all_hidden = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for layer in self.layer:
            if all_hidden is not None:
                all_hidden += (hidden_states,)
            hidden_states, attention = layer(hidden_states, output_attentions)
            if all_attentions is not None:
                all_attentions += (attention,)
        if all_hidden is not None:
            all_hidden += (hidden_states,)
        return hidden_states, all_hidden, all_attentions


class ViTModel(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any):
        hidden_states = self.embeddings(pixel_values)
        hidden_states, all_hidden, all_attentions = self.encoder(
            hidden_states,
            bool(kwargs.get('output_hidden_states', False)),
            bool(kwargs.get('output_attentions', False)),
        )
        return self.layernorm(hidden_states), all_hidden, all_attentions


@MODEL_COMPONENTS.register_module()
class LocalViTForImageClassification(nn.Module):
    """ViT classifier owned by HFTrainer and loadable from local artifacts."""

    config_class = ViTConfig

    def __init__(self, config: ViTConfig | dict | None = None, **config_kwargs: Any):
        super().__init__()
        if config is None:
            config = ViTConfig.from_dict(config_kwargs)
        elif isinstance(config, dict):
            config = ViTConfig.from_dict(config, **config_kwargs)
        elif config_kwargs:
            config = ViTConfig.from_dict(config.to_dict(), **config_kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(module, ViTEmbeddings):
            nn.init.trunc_normal_(module.cls_token, std=self.config.initializer_range)
            nn.init.trunc_normal_(module.position_embeddings, std=self.config.initializer_range)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        del kwargs
        self.gradient_checkpointing = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ):
        del kwargs
        hidden, all_hidden, all_attentions = self.vit(
            pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = self.classifier(hidden[:, 0])
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(-1), labels.to(logits.dtype))
            else:
                loss = F.cross_entropy(logits, labels.long())
        output = ImageClassifierOutput(loss, logits, all_hidden, all_attentions)
        return output if return_dict else output.to_tuple()

    @classmethod
    def from_config(cls, config: ViTConfig | dict | None = None, **kwargs: Any):
        return cls(config=config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_labels: Optional[int] = None,
        ignore_mismatched_sizes: bool = False,
        torch_dtype: Any = None,
        output_loading_info: bool = False,
        **kwargs: Any,
    ):
        for ignored in ('local_files_only', 'low_cpu_mem_usage'):
            kwargs.pop(ignored, None)
        if kwargs:
            raise TypeError(f'Unsupported local ViT load options: {sorted(kwargs)}')
        overrides = {'num_labels': num_labels} if num_labels is not None else {}
        config = ViTConfig.from_pretrained(pretrained_model_name_or_path, **overrides)
        model = cls(config)
        state = load_state_dict(pretrained_model_name_or_path)
        own = model.state_dict()
        mismatched = [
            (key, tuple(value.shape), tuple(own[key].shape))
            for key, value in state.items()
            if key in own and value.shape != own[key].shape
        ]
        if mismatched and not ignore_mismatched_sizes:
            raise RuntimeError(
                f'ViT checkpoint has shape mismatches: {mismatched[:8]}. '
                'Use ignore_mismatched_sizes=True only for classifier head replacement.'
            )
        if ignore_mismatched_sizes:
            replaceable_head_keys = {'classifier.weight', 'classifier.bias'}
            invalid_mismatches = [
                item for item in mismatched if item[0] not in replaceable_head_keys
            ]
            if invalid_mismatches:
                raise RuntimeError(
                    'ViT checkpoint has non-classifier shape mismatches: '
                    f'{invalid_mismatches[:8]}. Only classifier.weight and '
                    'classifier.bias may be replaced.'
                )
            state = dict(state)
            for key, _, _ in mismatched:
                state.pop(key)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if ignore_mismatched_sizes:
            invalid_missing = [
                key for key in missing if key not in replaceable_head_keys
            ]
            if invalid_missing or unexpected:
                raise RuntimeError(
                    'ViT checkpoint mismatch outside the replaceable classifier head: '
                    f'missing={invalid_missing[:8]}, unexpected={unexpected[:8]}. '
                    'Only classifier.weight and classifier.bias may be absent or '
                    'shape-mismatched.'
                )
        elif missing or unexpected:
            raise RuntimeError(
                f'ViT checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}. '
                'Use ignore_mismatched_sizes=True only for intentional head replacement.'
            )
        if torch_dtype not in (None, 'auto'):
            dtype = (
                getattr(torch, torch_dtype.removeprefix('torch.'))
                if isinstance(torch_dtype, str) else torch_dtype
            )
            model.to(dtype=dtype)
        info = {
            'missing_keys': list(missing),
            'unexpected_keys': list(unexpected),
            'mismatched_keys': mismatched,
        }
        return (model, info) if output_loading_info else model

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise TypeError(f'Unsupported local ViT save options: {sorted(kwargs)}')
        self.config.save_pretrained(save_directory)
        save_state_dict(self.state_dict(), save_directory, safe_serialization)


ViTForImageClassification = LocalViTForImageClassification


__all__ = [
    'ImageClassifierOutput',
    'LocalViTForImageClassification',
    'ViTForImageClassification',
    'ViTModel',
]
