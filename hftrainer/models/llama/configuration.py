"""Configuration for the repository-local LLaMA language model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass
class LlamaConfig:
    """Serializable LLaMA configuration using established artifact field names."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    hidden_act: str = 'silu'
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[int] = 2
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    head_dim: Optional[int] = None
    model_type: str = 'llama'
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in (
            'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
            'num_attention_heads', 'max_position_embeddings',
        ):
            setattr(self, name, int(getattr(self, name)))
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.num_key_value_heads = int(self.num_key_value_heads)
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads:
                raise ValueError('hidden_size must be divisible by num_attention_heads.')
            self.head_dim = self.hidden_size // self.num_attention_heads
        self.head_dim = int(self.head_dim)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError('num_attention_heads must be divisible by num_key_value_heads.')
        if min(
            self.vocab_size, self.hidden_size, self.intermediate_size,
            self.num_hidden_layers, self.num_attention_heads, self.num_key_value_heads,
            self.max_position_embeddings, self.head_dim,
        ) <= 0:
            raise ValueError('All LLaMA dimensions must be positive.')
        if self.head_dim % 2:
            raise ValueError('head_dim must be even for rotary embeddings.')
        if self.hidden_act not in {'silu', 'gelu', 'relu'}:
            raise ValueError(f'Unsupported LLaMA activation: {self.hidden_act}')

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], **overrides: Any) -> 'LlamaConfig':
        data = dict(values)
        data.update({key: value for key, value in overrides.items() if value is not None})
        known = {item.name for item in fields(cls)} - {'_extra'}
        init = {key: value for key, value in data.items() if key in known}
        init['_extra'] = {key: value for key, value in data.items() if key not in known}
        return cls(**init)

    @classmethod
    def from_pretrained(cls, path: str | Path, **overrides: Any) -> 'LlamaConfig':
        root = Path(path)
        config_path = root / 'config.json' if root.is_dir() else root
        if not config_path.is_file():
            raise FileNotFoundError(
                f'LLaMA config not found at {config_path}. Only local artifacts are supported.'
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
