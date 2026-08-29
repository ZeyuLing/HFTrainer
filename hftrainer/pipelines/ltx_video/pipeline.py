"""Unified HFTrainer inference surface for native LTX-2.5 pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hftrainer.models.ltx_video.bundle import LTXVideoBundle
from hftrainer.models.ltx_video.checkpoints import validate_ltx25_generation_shape
from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES

_UNSET = object()


@PIPELINES.register_module()
class LTXVideoPipeline(BasePipeline):
    """Generate synchronized video/audio with LTX-2.5.

    ``distilled`` mode is the official fixed eight-step fast pipeline and does
    not accept CFG/negative-prompt controls. ``dev_two_stage`` is the guided
    dev-transformer pipeline and is the correct route for LoRAs produced by the
    training adapter.
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

        api = self.bundle.backend_api
        official_images = self.bundle.make_image_conditionings(images)
        common = {
            'prompt': prompt,
            'seed': seed,
            'height': height,
            'width': width,
            'frame_rate': frame_rate,
            'images': official_images,
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
                    "The distilled LTX-2.5 pipeline uses CFG=1 and does not accept "
                    "a negative prompt. Use mode='dev_two_stage' for guided inference."
                )
            if num_inference_steps is not None:
                raise ValueError(
                    "The distilled LTX-2.5 schedule is fixed at eight steps; "
                    "num_inference_steps must be omitted."
                )
            video, audio, actual_frames, tiling = self.bundle.backend(**common)
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
                video_guider_params=self.bundle.make_guider_params(
                    self.defaults['video_guider'] if video_guider is None else video_guider,
                    modality='video',
                ),
                audio_guider_params=self.bundle.make_guider_params(
                    self.defaults['audio_guider'] if audio_guider is None else audio_guider,
                    modality='audio',
                ),
                max_batch_size=(
                    self.defaults['max_batch_size']
                    if max_batch_size is None
                    else int(max_batch_size)
                ),
            )
            video, audio, actual_frames, tiling = self.bundle.backend(**common)

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
            # The iterator is consumed by encode_video and must not be presented
            # as reusable output.
            result['video'] = None
        return result

    def __call__(self, prompt: str, **kwargs) -> dict[str, Any]:
        return self.infer_text_to_video(prompt, **kwargs)
