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
# MODIFIED BY HFTRAINER: public exports for the local Qwen3-VL runtime.

from .configuration import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from .modeling import (
    BaseModelOutputWithDeepstackFeatures,
    MiniMaxH3Qwen3VLEncoder,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLVisionModel,
)

__all__ = [
    "BaseModelOutputWithDeepstackFeatures",
    "MiniMaxH3Qwen3VLEncoder",
    "Qwen3VLCausalLMOutputWithPast",
    "Qwen3VLConfig",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLModelOutputWithPast",
    "Qwen3VLTextAttention",
    "Qwen3VLTextConfig",
    "Qwen3VLTextDecoderLayer",
    "Qwen3VLTextModel",
    "Qwen3VLTextRMSNorm",
    "Qwen3VLVisionConfig",
    "Qwen3VLVisionModel",
]
