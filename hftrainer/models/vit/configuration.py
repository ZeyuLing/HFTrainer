"""Configuration for the repository-local Vision Transformer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass
class ViTConfig:
    """Serializable Vision Transformer configuration.

    Field names intentionally follow the commonly published ViT checkpoint
    schema so existing ``config.json`` files can be consumed without a model
    framework dependency.
    """

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = 'gelu'
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    qkv_bias: bool = True
    num_labels: int = 1000
    problem_type: Optional[str] = None
    id2label: Dict[str, str] = field(default_factory=dict)
    label2id: Dict[str, int] = field(default_factory=dict)
    model_type: str = 'vit'
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.image_size = int(self.image_size)
        self.patch_size = int(self.patch_size)
        self.num_channels = int(self.num_channels)
        self.hidden_size = int(self.hidden_size)
        self.num_hidden_layers = int(self.num_hidden_layers)
        self.num_attention_heads = int(self.num_attention_heads)
        self.intermediate_size = int(self.intermediate_size)
        self.num_labels = int(self.num_labels)
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError('image_size and patch_size must be positive.')
        if self.image_size % self.patch_size:
            raise ValueError('image_size must be divisible by patch_size.')
        if self.hidden_size % self.num_attention_heads:
            raise ValueError('hidden_size must be divisible by num_attention_heads.')
        if self.num_hidden_layers <= 0 or self.intermediate_size <= 0:
            raise ValueError('num_hidden_layers and intermediate_size must be positive.')

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], **overrides: Any) -> 'ViTConfig':
        data = dict(values)
        data.update({key: value for key, value in overrides.items() if value is not None})
        if 'num_labels' not in data:
            labels = data.get('id2label') or data.get('label2id')
            if isinstance(labels, Mapping) and labels:
                data['num_labels'] = len(labels)
        known = {item.name for item in fields(cls)} - {'_extra'}
        init = {key: value for key, value in data.items() if key in known}
        init['_extra'] = {key: value for key, value in data.items() if key not in known}
        return cls(**init)

    @classmethod
    def from_pretrained(cls, path: str | Path, **overrides: Any) -> 'ViTConfig':
        directory = Path(path)
        config_path = directory / 'config.json' if directory.is_dir() else directory
        if not config_path.is_file():
            raise FileNotFoundError(
                f'ViT config not found at {config_path}. Only local artifacts are supported.'
            )
        with config_path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        return cls.from_dict(data, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        extra = data.pop('_extra', {})
        data.update(extra)
        return data

    def save_pretrained(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'config.json').open('w', encoding='utf-8') as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
