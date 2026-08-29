"""LTX-2.5 model bundle backed by Lightricks' native pipeline packages."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.models.ltx_video.checkpoints import (
    LTX25InferenceCheckpoints,
    normalize_lora_specs,
)
from hftrainer.models.ltx_video.runtime import require_ltx_torch_capabilities
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.optional import require_modules

_LTX_INSTALL_HINT = 'python -m pip install -e ".[ltx-video]"'


@MODEL_BUNDLES.register_module()
class LTXVideoBundle(ModelBundle):
    """Own an LTX-2.5 split pack and lazily construct its native backend.

    The official pipeline owns the actual 22B modules and their memory/offload
    lifecycle.  This Bundle deliberately owns the component *contract* rather
    than registering a second copy of those modules in PyTorch.  Training uses
    :class:`hftrainer.trainers.ltx_video.LTXVideoTrainer`, which delegates to
    the official trainer's complete optimization loop.
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
        load_model: bool = False,
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
        object.__setattr__(self, '_backend', None)
        object.__setattr__(self, '_backend_api', None)

        self.validate()
        if load_model:
            self.load_model()

    def validate(self) -> None:
        self.checkpoints.validate(
            mode=self.mode,
            require_files=self.validate_paths,
            strict_roles=self.strict_checkpoint_roles,
        )
        if self.validate_paths:
            for spec in self.lora_specs:
                if not Path(spec.path).expanduser().is_file():
                    raise FileNotFoundError(f"Missing LTX LoRA: {spec.path}")

    @staticmethod
    def _import_backend_api() -> SimpleNamespace:
        require_ltx_torch_capabilities('LTX-2.5 inference')
        modules = require_modules(
            [
                'ltx_core.allocator_trim_strategy',
                'ltx_core.components.guiders',
                'ltx_core.loader',
                'ltx_core.model.transformer.compiling',
                'ltx_core.model.video_vae',
                'ltx_core.model.video_vae.transformer',
                'ltx_pipelines.distilled',
                'ltx_pipelines.ti2vid_two_stages',
                'ltx_pipelines.utils.args',
                'ltx_pipelines.utils.media_io',
                'ltx_pipelines.utils.model_paths',
                'ltx_pipelines.utils.quantization_factory',
                'ltx_pipelines.utils.types',
            ],
            feature='LTX-2.5 inference',
            install_hint=_LTX_INSTALL_HINT,
        )
        loader = modules['ltx_core.loader']
        return SimpleNamespace(
            AllocatorTrimStrategy=modules[
                'ltx_core.allocator_trim_strategy'
            ].AllocatorTrimStrategy,
            MultiModalGuiderParams=modules[
                'ltx_core.components.guiders'
            ].MultiModalGuiderParams,
            LoraPathStrengthAndSDOps=loader.LoraPathStrengthAndSDOps,
            LTXV_LORA_COMFY_RENAMING_MAP=loader.LTXV_LORA_COMFY_RENAMING_MAP,
            CompilationConfig=modules[
                'ltx_core.model.transformer.compiling'
            ].CompilationConfig,
            AUTO_TILING=modules['ltx_core.model.video_vae'].AUTO_TILING,
            get_video_chunks_number=modules[
                'ltx_core.model.video_vae'
            ].get_video_chunks_number,
            DiffVAEMode=modules[
                'ltx_core.model.video_vae.transformer'
            ].DiffVAEMode,
            DistilledPipeline=modules['ltx_pipelines.distilled'].DistilledPipeline,
            TI2VidTwoStagesPipeline=modules[
                'ltx_pipelines.ti2vid_two_stages'
            ].TI2VidTwoStagesPipeline,
            ImageConditioningInput=modules[
                'ltx_pipelines.utils.args'
            ].ImageConditioningInput,
            encode_video=modules['ltx_pipelines.utils.media_io'].encode_video,
            ModelPaths=modules['ltx_pipelines.utils.model_paths'].ModelPaths,
            QuantizationKind=modules[
                'ltx_pipelines.utils.quantization_factory'
            ].QuantizationKind,
            OffloadMode=modules['ltx_pipelines.utils.types'].OffloadMode,
            DEFAULT_AUTO_DURATION=modules[
                'ltx_pipelines.utils.types'
            ].DEFAULT_AUTO_DURATION,
        )

    @property
    def backend_api(self) -> SimpleNamespace:
        api = self._backend_api
        if api is None:
            api = self._import_backend_api()
            object.__setattr__(self, '_backend_api', api)
        return api

    @staticmethod
    def _enum_value(enum_cls, value, field_name: str):
        if isinstance(value, enum_cls):
            return value
        try:
            return enum_cls(value)
        except (TypeError, ValueError) as exc:
            choices = [item.value for item in enum_cls]
            raise ValueError(
                f"Invalid {field_name}={value!r}; expected one of {choices}."
            ) from exc

    def _resolve_quantization(self, api):
        if self.quantization is None or not isinstance(self.quantization, str):
            return self.quantization
        try:
            kind = api.QuantizationKind(self.quantization)
        except ValueError as exc:
            choices = [item.value for item in api.QuantizationKind]
            raise ValueError(
                f"Invalid quantization={self.quantization!r}; expected one of {choices}."
            ) from exc
        checkpoint = self.quantization_checkpoint_path or self.checkpoints.transformer_path
        return kind.to_policy(checkpoint_path=checkpoint)

    def _resolve_compilation(self, api):
        value = self.compilation_config
        if value is None or isinstance(value, api.CompilationConfig):
            return value
        if hasattr(value, 'to_dict'):
            value = value.to_dict()
        if isinstance(value, dict):
            return api.CompilationConfig(**value)
        raise TypeError('compilation_config must be a mapping or CompilationConfig.')

    def _official_loras(self, api, specs=None):
        return [
            api.LoraPathStrengthAndSDOps(
                spec.path,
                spec.strength,
                api.LTXV_LORA_COMFY_RENAMING_MAP,
            )
            for spec in (self.lora_specs if specs is None else specs)
        ]

    def build_model_paths(self, api=None):
        api = api or self.backend_api
        return api.ModelPaths.from_split(
            transformer_path=self.checkpoints.transformer_path,
            text_encoder_path=self.checkpoints.text_encoder_path,
            video_vae_path=self.checkpoints.video_vae_path,
            audio_vae_path=self.checkpoints.audio_vae_path,
            duration_head_path=self.checkpoints.duration_head_path,
        )

    def load_model(self):
        """Construct and cache the selected official inference pipeline."""

        if self._backend is not None:
            return self._backend

        self.validate()
        if (
            self.device_name is not None
            and torch.device(self.device_name).type == 'cuda'
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "LTX-2.5 inference is configured for a CUDA device, but "
                "torch.cuda.is_available() is false. Install a CUDA-enabled "
                "PyTorch build that matches the host driver, or explicitly choose "
                "a supported non-CUDA device configuration."
            )
        api = self.backend_api
        common = {
            'model_paths': self.build_model_paths(api),
            'spatial_upsampler_path': self.checkpoints.spatial_upsampler_path,
            'loras': self._official_loras(api),
            'device': (
                torch.device(self.device_name) if self.device_name is not None else None
            ),
            'quantization': self._resolve_quantization(api),
            'compilation_config': self._resolve_compilation(api),
            'offload_mode': self._enum_value(
                api.OffloadMode,
                self.offload_mode,
                'offload_mode',
            ),
            'alloc_trim_strategy': self._enum_value(
                api.AllocatorTrimStrategy,
                self.alloc_trim_strategy,
                'alloc_trim_strategy',
            ),
            'prompt_enhancer_gemma_root': self.prompt_enhancer_gemma_root,
            'diffvae_optimization': self._enum_value(
                api.DiffVAEMode,
                self.diffvae_optimization,
                'diffvae_optimization',
            ),
        }
        if self.mode == 'distilled':
            # `loras` is required by the current official constructor even for
            # an empty list; keep it explicit to avoid the model-card snippet bug.
            backend = api.DistilledPipeline(**common)
        else:
            distilled_spec = normalize_lora_specs(
                [{'path': self.checkpoints.distilled_lora_path, 'strength': 1.0}]
            )
            backend = api.TI2VidTwoStagesPipeline(
                distilled_lora=self._official_loras(api, distilled_spec),
                **common,
            )
        object.__setattr__(self, '_backend', backend)
        return backend

    @property
    def backend(self):
        return self.load_model()

    def unload_model(self) -> None:
        """Release HFTrainer's reference to the official backend."""

        object.__setattr__(self, '_backend', None)

    def make_image_conditionings(self, values: Iterable[Any] | None):
        api = self.backend_api
        result = []
        for value in values or ():
            if isinstance(value, api.ImageConditioningInput):
                result.append(value)
            elif isinstance(value, (str, Path)):
                result.append(api.ImageConditioningInput(str(value), 0, 1.0, None))
            elif isinstance(value, dict):
                result.append(
                    api.ImageConditioningInput(
                        str(value['path']),
                        int(value.get('frame_idx', 0)),
                        float(value.get('strength', 1.0)),
                        value.get('crf'),
                    )
                )
            elif isinstance(value, (tuple, list)) and 3 <= len(value) <= 4:
                result.append(api.ImageConditioningInput(*value))
            else:
                raise TypeError(
                    "images entries must be paths, mappings, ImageConditioningInput, "
                    "or (path, frame_idx, strength[, crf]) tuples."
                )
        return result

    def make_guider_params(self, values: dict | Any | None, *, modality: str):
        api = self.backend_api
        if isinstance(values, api.MultiModalGuiderParams):
            return values
        defaults = {
            'video': {
                'cfg_scale': 3.0,
                'stg_scale': 1.0,
                'rescale_scale': 0.7,
                'modality_scale': 3.0,
                'skip_step': 0,
                'stg_blocks': [28],
            },
            'audio': {
                'cfg_scale': 7.0,
                'stg_scale': 1.0,
                'rescale_scale': 0.7,
                'modality_scale': 3.0,
                'skip_step': 0,
                'stg_blocks': [28],
            },
        }[modality]
        if values:
            if hasattr(values, 'to_dict'):
                values = values.to_dict()
            defaults.update(dict(values))
        return api.MultiModalGuiderParams(**defaults)
