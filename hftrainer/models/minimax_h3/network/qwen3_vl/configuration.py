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
# MODIFIED BY HFTRAINER: dependency-free, JSON-compatible configuration
# objects for the repository-owned Qwen3-VL conditioner runtime.

"""Qwen3-VL configuration objects without a Transformers dependency."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..configuration import ConfigDict


class Qwen3VLVisionConfig(ConfigDict):
    """Configuration of the Qwen3-VL vision tower.

    Defaults match the public Qwen3-VL configuration.  MiniMax-H3 overrides
    ``out_hidden_size`` to 5120 in its released ``text_encoder/config.json``.
    Unknown JSON fields are retained so a load/save round trip is lossless.
    """

    model_type = "qwen3_vl_vision"

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list[int] | tuple[int, ...] = (8, 16, 24),
        initializer_range: float = 0.02,
        model_type: str = "qwen3_vl_vision",
        **kwargs: Any,
    ) -> None:
        values = {
            "depth": int(depth),
            "hidden_size": int(hidden_size),
            "hidden_act": str(hidden_act),
            "intermediate_size": int(intermediate_size),
            "num_heads": int(num_heads),
            "in_channels": int(in_channels),
            "patch_size": int(patch_size),
            "spatial_merge_size": int(spatial_merge_size),
            "temporal_patch_size": int(temporal_patch_size),
            "out_hidden_size": int(out_hidden_size),
            "num_position_embeddings": int(num_position_embeddings),
            "deepstack_visual_indexes": [
                int(value) for value in deepstack_visual_indexes
            ],
            "initializer_range": float(initializer_range),
            "model_type": (
                "qwen3_vl_vision" if model_type == "qwen3_vl" else str(model_type)
            ),
        }
        values.update(kwargs)
        super().__init__(values)
        if self.hidden_size % self.num_heads:
            raise ValueError("vision hidden_size must be divisible by num_heads")
        if self.spatial_merge_size < 1 or self.temporal_patch_size < 1:
            raise ValueError("vision patch merge sizes must be positive")
        if int(self.num_position_embeddings**0.5) ** 2 != self.num_position_embeddings:
            raise ValueError(
                "num_position_embeddings must describe a square learned grid"
            )
        if any(
            index < 0 or index >= self.depth for index in self.deepstack_visual_indexes
        ):
            raise ValueError(
                "deepstack_visual_indexes must name existing vision blocks"
            )


class Qwen3VLTextConfig(ConfigDict):
    """Configuration of the causal GQA language tower."""

    model_type = "qwen3_vl_text"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 32,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 128000,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        rope_parameters: Mapping[str, Any] | None = None,
        rope_scaling: Mapping[str, Any] | None = None,
        rope_theta: float = 500000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        layer_types: list[str] | tuple[str, ...] | None = None,
        model_type: str = "qwen3_vl_text",
        **kwargs: Any,
    ) -> None:
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if rope_parameters is None:
            rope_parameters = dict(rope_scaling or {})
            rope_parameters.setdefault("rope_type", "default")
            rope_parameters.setdefault("rope_theta", float(rope_theta))
            rope_parameters.setdefault("mrope_section", [24, 20, 20])
        else:
            rope_parameters = dict(rope_parameters)
            rope_parameters.setdefault("rope_theta", float(rope_theta))
            rope_parameters.setdefault("rope_type", "default")
        if layer_types is None:
            layer_types = ["full_attention"] * int(num_hidden_layers)
        values = {
            "vocab_size": int(vocab_size),
            "hidden_size": int(hidden_size),
            "intermediate_size": int(intermediate_size),
            "num_hidden_layers": int(num_hidden_layers),
            "num_attention_heads": int(num_attention_heads),
            "num_key_value_heads": int(num_key_value_heads),
            "head_dim": int(head_dim),
            "hidden_act": str(hidden_act),
            "max_position_embeddings": int(max_position_embeddings),
            "initializer_range": float(initializer_range),
            "rms_norm_eps": float(rms_norm_eps),
            "use_cache": bool(use_cache),
            "rope_parameters": ConfigDict(rope_parameters),
            "rope_scaling": ConfigDict(dict(rope_scaling or rope_parameters)),
            "rope_theta": float(rope_theta),
            "attention_bias": bool(attention_bias),
            "attention_dropout": float(attention_dropout),
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "layer_types": [str(value) for value in layer_types],
            "model_type": str(model_type),
        }
        values.update(kwargs)
        super().__init__(values)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        sections = list(self.rope_parameters.get("mrope_section", ()))
        if (
            len(sections) != 3
            or sum(int(value) for value in sections) != self.head_dim // 2
        ):
            raise ValueError(
                "mrope_section must contain three values summing to head_dim // 2"
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per decoder layer")


class Qwen3VLConfig(ConfigDict):
    """Composite Qwen3-VL config preserving official nested JSON fields."""

    model_type = "qwen3_vl"

    def __init__(
        self,
        text_config: Mapping[str, Any] | Qwen3VLTextConfig | None = None,
        vision_config: Mapping[str, Any] | Qwen3VLVisionConfig | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        tie_word_embeddings: bool = False,
        model_type: str = "qwen3_vl",
        **kwargs: Any,
    ) -> None:
        if not isinstance(text_config, Qwen3VLTextConfig):
            text_config = Qwen3VLTextConfig(**dict(text_config or {}))
        if not isinstance(vision_config, Qwen3VLVisionConfig):
            vision_config = Qwen3VLVisionConfig(**dict(vision_config or {}))
        values = {
            "text_config": text_config,
            "vision_config": vision_config,
            "image_token_id": int(image_token_id),
            "video_token_id": int(video_token_id),
            "vision_start_token_id": int(vision_start_token_id),
            "vision_end_token_id": int(vision_end_token_id),
            "tie_word_embeddings": bool(tie_word_embeddings),
            "model_type": str(model_type),
        }
        values.update(kwargs)
        super().__init__(values)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Qwen3VLConfig:
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        result = dict(self)
        result["text_config"] = self.text_config.to_dict()
        result["vision_config"] = self.vision_config.to_dict()
        return result


__all__ = ["Qwen3VLConfig", "Qwen3VLTextConfig", "Qwen3VLVisionConfig"]
