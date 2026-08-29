"""HFTrainer-owned inference orchestration for LTX-Video 2.5."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from hftrainer.models.ltx_video.bundle import LTXVideoBundle
from hftrainer.models.ltx_video.checkpoints import (
    normalize_lora_specs,
    validate_ltx25_generation_shape,
)
from hftrainer.models.ltx_video.runtime import require_ltx_torch_capabilities
from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES

_UNSET = object()


@PIPELINES.register_module()
class LTXVideoPipeline(BasePipeline):
    """Build and execute the repository-local LTX-Video inference graph.

    Model artifact roles and model-level settings stay in ``LTXVideoBundle``.
    Backend construction, conditioning conversion, denoising orchestration and
    output encoding belong here, so the model package never imports a pipeline.
    """

    BUNDLE_CLS = LTXVideoBundle

    def __init__(
        self,
        bundle: LTXVideoBundle,
        *,
        height: int = 512,
        width: int = 768,
        num_frames: int | None = 121,
        frame_rate: float = 24.0,
        seed: int = 42,
        num_inference_steps: int = 30,
        negative_prompt: str = '',
        video_guider: dict | Any | None = None,
        audio_guider: dict | Any | None = None,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        generated_keyframes: int | list[int] = 0,
        max_batch_size: int = 1,
    ):
        super().__init__(bundle=bundle)
        self.defaults = {
            'height': int(height),
            'width': int(width),
            'num_frames': num_frames,
            'frame_rate': float(frame_rate),
            'seed': int(seed),
            'num_inference_steps': int(num_inference_steps),
            'negative_prompt': str(negative_prompt),
            'video_guider': video_guider,
            'audio_guider': audio_guider,
            'enhance_prompt': bool(enhance_prompt),
            'enhance_static_cache': bool(enhance_static_cache),
            'generated_keyframes': generated_keyframes,
            'max_batch_size': int(max_batch_size),
        }
        self._backend_api: SimpleNamespace | None = None
        self._backend: Any | None = None

    @staticmethod
    def _import_backend_api() -> SimpleNamespace:
        """Import only repository-owned numerical and pipeline components."""

        require_ltx_torch_capabilities('LTX-Video 2.5 inference')
        from hftrainer.models.ltx_video.network.allocator_trim_strategy import (
            AllocatorTrimStrategy,
        )
        from hftrainer.models.ltx_video.network.components.guiders import (
            MultiModalGuiderParams,
        )
        from hftrainer.models.ltx_video.network.loader import (
            LoraPathStrengthAndSDOps,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )
        from hftrainer.models.ltx_video.network.model.transformer.compiling import (
            CompilationConfig,
        )
        from hftrainer.models.ltx_video.network.model.video_vae import (
            AUTO_TILING,
            get_video_chunks_number,
        )
        from hftrainer.models.ltx_video.network.model.video_vae.transformer import (
            DiffVAEMode,
        )
        from hftrainer.pipelines.ltx_video.backend.distilled import DistilledPipeline
        from hftrainer.pipelines.ltx_video.backend.ti2vid_two_stages import (
            TI2VidTwoStagesPipeline,
        )
        from hftrainer.pipelines.ltx_video.backend.utils.args import (
            ImageConditioningInput,
        )
        from hftrainer.pipelines.ltx_video.backend.utils.media_io import encode_video
        from hftrainer.pipelines.ltx_video.backend.utils.model_paths import ModelPaths
        from hftrainer.pipelines.ltx_video.backend.utils.quantization_factory import (
            QuantizationKind,
        )
        from hftrainer.pipelines.ltx_video.backend.utils.types import (
            DEFAULT_AUTO_DURATION,
            OffloadMode,
        )

        return SimpleNamespace(
            AllocatorTrimStrategy=AllocatorTrimStrategy,
            MultiModalGuiderParams=MultiModalGuiderParams,
            LoraPathStrengthAndSDOps=LoraPathStrengthAndSDOps,
            LTXV_LORA_COMFY_RENAMING_MAP=LTXV_LORA_COMFY_RENAMING_MAP,
            CompilationConfig=CompilationConfig,
            AUTO_TILING=AUTO_TILING,
            get_video_chunks_number=get_video_chunks_number,
            DiffVAEMode=DiffVAEMode,
            DistilledPipeline=DistilledPipeline,
            TI2VidTwoStagesPipeline=TI2VidTwoStagesPipeline,
            ImageConditioningInput=ImageConditioningInput,
            encode_video=encode_video,
            ModelPaths=ModelPaths,
            QuantizationKind=QuantizationKind,
            OffloadMode=OffloadMode,
            DEFAULT_AUTO_DURATION=DEFAULT_AUTO_DURATION,
        )

    @property
    def backend_api(self) -> SimpleNamespace:
        if self._backend_api is None:
            self._backend_api = self._import_backend_api()
        return self._backend_api

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
        value = self.bundle.quantization
        if value is None or not isinstance(value, str):
            return value
        try:
            kind = api.QuantizationKind(value)
        except ValueError as exc:
            choices = [item.value for item in api.QuantizationKind]
            raise ValueError(
                f"Invalid quantization={value!r}; expected one of {choices}."
            ) from exc
        checkpoint = (
            self.bundle.quantization_checkpoint_path
            or self.bundle.checkpoints.transformer_path
        )
        return kind.to_policy(checkpoint_path=checkpoint)

    def _resolve_compilation(self, api):
        value = self.bundle.compilation_config
        if value is None or isinstance(value, api.CompilationConfig):
            return value
        if hasattr(value, 'to_dict'):
            value = value.to_dict()
        if isinstance(value, dict):
            return api.CompilationConfig(**value)
        raise TypeError('compilation_config must be a mapping or CompilationConfig.')

    @staticmethod
    def _local_loras(api, specs):
        return [
            api.LoraPathStrengthAndSDOps(
                spec.path,
                spec.strength,
                api.LTXV_LORA_COMFY_RENAMING_MAP,
            )
            for spec in specs
        ]

    def _build_model_paths(self, api):
        checkpoints = self.bundle.checkpoints
        return api.ModelPaths.from_split(
            transformer_path=checkpoints.transformer_path,
            text_encoder_path=checkpoints.text_encoder_path,
            video_vae_path=checkpoints.video_vae_path,
            audio_vae_path=checkpoints.audio_vae_path,
            duration_head_path=checkpoints.duration_head_path,
        )

    def load_backend(self):
        """Construct and cache the selected local inference graph."""

        if self._backend is not None:
            return self._backend

        self.bundle.validate()
        device_name = self.bundle.device_name
        if (
            device_name is not None
            and torch.device(device_name).type == 'cuda'
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "LTX-Video 2.5 inference is configured for CUDA, but "
                "torch.cuda.is_available() is false. Install a CUDA-enabled "
                "PyTorch build matching the host driver, or choose a supported "
                "non-CUDA device."
            )

        api = self.backend_api
        common = {
            'model_paths': self._build_model_paths(api),
            'spatial_upsampler_path': self.bundle.checkpoints.spatial_upsampler_path,
            'loras': self._local_loras(api, self.bundle.lora_specs),
            'registry': self.bundle.component_registry,
            'device': torch.device(device_name) if device_name is not None else None,
            'quantization': self._resolve_quantization(api),
            'compilation_config': self._resolve_compilation(api),
            'offload_mode': self._enum_value(
                api.OffloadMode,
                self.bundle.offload_mode,
                'offload_mode',
            ),
            'alloc_trim_strategy': self._enum_value(
                api.AllocatorTrimStrategy,
                self.bundle.alloc_trim_strategy,
                'alloc_trim_strategy',
            ),
            'prompt_enhancer_gemma_root': self.bundle.prompt_enhancer_gemma_root,
            'diffvae_optimization': self._enum_value(
                api.DiffVAEMode,
                self.bundle.diffvae_optimization,
                'diffvae_optimization',
            ),
        }
        if self.bundle.mode == 'distilled':
            backend = api.DistilledPipeline(**common)
        else:
            distilled_specs = normalize_lora_specs(
                [
                    {
                        'path': self.bundle.checkpoints.distilled_lora_path,
                        'strength': 1.0,
                    }
                ]
            )
            backend = api.TI2VidTwoStagesPipeline(
                distilled_lora=self._local_loras(api, distilled_specs),
                **common,
            )
        self._backend = backend
        return backend

    @property
    def backend(self):
        return self.load_backend()

    def unload_backend(self) -> None:
        """Release the cached inference graph and loaded component references."""

        self._backend = None

    def _make_image_conditionings(self, values: Iterable[Any] | None):
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

    def _make_guider_params(self, values: dict | Any | None, *, modality: str):
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

    def infer_text_to_video(
        self,
        prompt: str,
        *,
        output_path: str | Path | None = None,
        images: Iterable[Any] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | str | None | object = _UNSET,
        frame_rate: float | None = None,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        negative_prompt: str | None = None,
        video_guider: dict | Any | None = None,
        audio_guider: dict | Any | None = None,
        enhance_prompt: bool | None = None,
        enhance_static_cache: bool | None = None,
        generated_keyframes: int | list[int] | None = None,
        max_batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Run text/image-to-video inference and optionally encode an MP4."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt must be a non-empty string.')

        height = self.defaults['height'] if height is None else int(height)
        width = self.defaults['width'] if width is None else int(width)
        if num_frames is _UNSET:
            requested_frames = self.defaults['num_frames']
        elif num_frames is None or str(num_frames).lower() == 'auto':
            requested_frames = None
        else:
            requested_frames = int(num_frames)
        frame_rate = self.defaults['frame_rate'] if frame_rate is None else float(frame_rate)
        seed = self.defaults['seed'] if seed is None else int(seed)
        if frame_rate <= 0:
            raise ValueError('frame_rate must be positive.')
        validate_ltx25_generation_shape(
            height=height,
            width=width,
            num_frames=requested_frames,
            has_duration_head=self.bundle.checkpoints.duration_head_path is not None,
        )

        api = self.backend_api
        local_images = self._make_image_conditionings(images)
        common = {
            'prompt': prompt,
            'seed': seed,
            'height': height,
            'width': width,
            'frame_rate': frame_rate,
            'images': local_images,
            'num_frames': (
                requested_frames
                if requested_frames is not None
                else api.DEFAULT_AUTO_DURATION
            ),
            'tiling_config': api.AUTO_TILING,
            'enhance_prompt': (
                self.defaults['enhance_prompt']
                if enhance_prompt is None
                else bool(enhance_prompt)
            ),
            'enhance_static_cache': (
                self.defaults['enhance_static_cache']
                if enhance_static_cache is None
                else bool(enhance_static_cache)
            ),
            'generated_keyframes': (
                self.defaults['generated_keyframes']
                if generated_keyframes is None
                else generated_keyframes
            ),
        }

        if self.bundle.mode == 'distilled':
            if negative_prompt not in (None, ''):
                raise ValueError(
                    "The distilled LTX-Video 2.5 pipeline uses CFG=1 and does not "
                    "accept a negative prompt. Use mode='dev_two_stage' for "
                    "guided inference."
                )
            if num_inference_steps is not None:
                raise ValueError(
                    "The distilled LTX-Video 2.5 schedule is fixed at eight steps; "
                    "num_inference_steps must be omitted."
                )
            video, audio, actual_frames, tiling = self.backend(**common)
        else:
            common.update(
                negative_prompt=(
                    self.defaults['negative_prompt']
                    if negative_prompt is None
                    else str(negative_prompt)
                ),
                num_inference_steps=(
                    self.defaults['num_inference_steps']
                    if num_inference_steps is None
                    else int(num_inference_steps)
                ),
                video_guider_params=self._make_guider_params(
                    self.defaults['video_guider'] if video_guider is None else video_guider,
                    modality='video',
                ),
                audio_guider_params=self._make_guider_params(
                    self.defaults['audio_guider'] if audio_guider is None else audio_guider,
                    modality='audio',
                ),
                max_batch_size=(
                    self.defaults['max_batch_size']
                    if max_batch_size is None
                    else int(max_batch_size)
                ),
            )
            video, audio, actual_frames, tiling = self.backend(**common)

        result = {
            'video': video,
            'audio': audio,
            'num_frames': int(actual_frames),
            'frame_rate': frame_rate,
            'height': height,
            'width': width,
            'seed': seed,
            'mode': self.bundle.mode,
            'tiling_config': tiling,
            'output_path': None,
        }
        if output_path is not None:
            output_path = Path(output_path).expanduser()
            api.encode_video(
                video=video,
                fps=frame_rate,
                audio=audio,
                output_path=str(output_path),
                video_chunks_number=api.get_video_chunks_number(actual_frames, tiling),
            )
            result['output_path'] = str(output_path.resolve())
            result['video'] = None
        return result

    def __call__(self, prompt: str, **kwargs) -> dict[str, Any]:
        return self.infer_text_to_video(prompt, **kwargs)
