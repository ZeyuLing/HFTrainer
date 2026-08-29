# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Loader utilities for model weights, LoRAs, and safetensor operations."""

from hftrainer.models.ltx_video.network.loader.fuse_loras import apply_loras
from hftrainer.models.ltx_video.network.loader.helpers import (
    create_meta_model,
    load_state_dict,
    parse_model_version,
    read_model_config,
    read_model_metadata,
)
from hftrainer.models.ltx_video.network.loader.module_ops import ModuleOps
from hftrainer.models.ltx_video.network.loader.primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from hftrainer.models.ltx_video.network.loader.registry import (
    DummyRegistry,
    ModelRegistry,
    Registry,
    module_registry_key,
)
from hftrainer.models.ltx_video.network.loader.sd_ops import (
    LTXV_LORA_COMFY_RENAMING_MAP,
    ContentMatching,
    ContentReplacement,
    KeyValueOperation,
    KeyValueOperationResult,
    SDKeyValueOperation,
    SDOps,
)
from hftrainer.models.ltx_video.network.loader.sft_loader import SafetensorsModelStateDictLoader, SafetensorsStateDictLoader
from hftrainer.models.ltx_video.network.loader.single_gpu_model_builder import SingleGPUModelBuilder

__all__ = [
    "LTXV_LORA_COMFY_RENAMING_MAP",
    "ContentMatching",
    "ContentReplacement",
    "DummyRegistry",
    "KeyValueOperation",
    "KeyValueOperationResult",
    "LoRAAdaptableProtocol",
    "LoraPathStrengthAndSDOps",
    "LoraStateDictWithStrength",
    "ModelBuilderProtocol",
    "ModelRegistry",
    "ModuleOps",
    "Registry",
    "SDKeyValueOperation",
    "SDOps",
    "SafetensorsModelStateDictLoader",
    "SafetensorsStateDictLoader",
    "SingleGPUModelBuilder",
    "StateDict",
    "StateDictLoader",
    "apply_loras",
    "create_meta_model",
    "load_state_dict",
    "module_registry_key",
    "parse_model_version",
    "read_model_config",
    "read_model_metadata",
]
