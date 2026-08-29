"""LTX-2.5 checkpoint bundle with repository-owned model contracts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.models.ltx_video.checkpoints import (
    LTX25InferenceCheckpoints,
    normalize_lora_specs,
)
from hftrainer.models.ltx_video.component_loader import LTXComponentStore
from hftrainer.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class LTXVideoBundle(ModelBundle):
    """Own the artifacts and model-level settings for LTX-Video 2.5.

    The bundle deliberately has no dependency on ``hftrainer.pipelines`` or
    ``hftrainer.trainers``. Numerical model code lives under
    ``hftrainer.models.ltx_video.network``; inference graph construction is
    owned by :class:`hftrainer.pipelines.ltx_video.LTXVideoPipeline`.
    """

    SUPPORTED_MODES = ('distilled', 'dev_two_stage')

    def __init__(
        self,
        transformer_path: str,
        text_encoder_path: str,
        video_vae_path: str,
        audio_vae_path: str,
        spatial_upsampler_path: str,
        duration_head_path: str | None = None,
        distilled_lora_path: str | None = None,
        loras: Iterable[Any] | None = None,
        mode: str = 'distilled',
        device: str | torch.device | None = None,
        quantization: str | Any | None = None,
        quantization_checkpoint_path: str | None = None,
        offload_mode: str | Any = 'none',
        alloc_trim_strategy: str | Any = 'trim',
        compilation_config: dict | Any | None = None,
        prompt_enhancer_gemma_root: str | None = None,
        diffvae_optimization: str | Any = 'chunked_eager',
        validate_paths: bool = True,
        strict_checkpoint_roles: bool = True,
        components: LTXComponentStore | None = None,
    ):
        super().__init__()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {self.SUPPORTED_MODES}; got {mode!r}")

        self.mode = mode
        self.checkpoints = LTX25InferenceCheckpoints(
            transformer_path=str(Path(transformer_path).expanduser()),
            text_encoder_path=str(Path(text_encoder_path).expanduser()),
            video_vae_path=str(Path(video_vae_path).expanduser()),
            audio_vae_path=str(Path(audio_vae_path).expanduser()),
            spatial_upsampler_path=str(Path(spatial_upsampler_path).expanduser()),
            duration_head_path=(
                str(Path(duration_head_path).expanduser()) if duration_head_path else None
            ),
            distilled_lora_path=(
                str(Path(distilled_lora_path).expanduser()) if distilled_lora_path else None
            ),
        )
        self.lora_specs = normalize_lora_specs(loras)
        self.device_name = device
        self.quantization = quantization
        self.quantization_checkpoint_path = quantization_checkpoint_path
        self.offload_mode = offload_mode
        self.alloc_trim_strategy = alloc_trim_strategy
        self.compilation_config = compilation_config
        self.prompt_enhancer_gemma_root = prompt_enhancer_gemma_root
        self.diffvae_optimization = diffvae_optimization
        self.validate_paths = bool(validate_paths)
        self.strict_checkpoint_roles = bool(strict_checkpoint_roles)
        self._components = components
        self.validate()

    @property
    def components(self) -> LTXComponentStore:
        """Return the bundle-owned lazy store for local LTX components.

        Constructing a bundle must not allocate a 22B model. The registry owns
        reusable repository-local model shells and optional weight caches only
        when a trainer or pipeline asks for them. All inference blocks receive
        this same registry instead of creating private model stores.
        """

        if self._components is None:
            self._components = LTXComponentStore()
        return self._components

    @property
    def component_registry(self):
        """Compatibility view of the inference registry owned by ``components``."""

        return self.components.inference_registry

    def clear_components(self) -> None:
        """Release model shells and tensors held by this bundle's local cache."""

        if self._components is not None:
            self._components.clear()

    def validate(self) -> None:
        """Validate artifact roles without importing an inference backend."""

        self.checkpoints.validate(
            mode=self.mode,
            require_files=self.validate_paths,
            strict_roles=self.strict_checkpoint_roles,
        )
        if self.validate_paths:
            for spec in self.lora_specs:
                if not Path(spec.path).expanduser().is_file():
                    raise FileNotFoundError(f"Missing LTX LoRA: {spec.path}")

    def artifact_paths(self) -> dict[str, str | None]:
        """Return the typed split-checkpoint roles consumed by the pipeline."""

        return {
            'transformer_path': self.checkpoints.transformer_path,
            'text_encoder_path': self.checkpoints.text_encoder_path,
            'video_vae_path': self.checkpoints.video_vae_path,
            'audio_vae_path': self.checkpoints.audio_vae_path,
            'spatial_upsampler_path': self.checkpoints.spatial_upsampler_path,
            'duration_head_path': self.checkpoints.duration_head_path,
            'distilled_lora_path': self.checkpoints.distilled_lora_path,
        }
