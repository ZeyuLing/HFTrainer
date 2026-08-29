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
# MODIFIED BY HFTRAINER: local image/video preprocessing and MiniMax-H3
# presentation assembly without transformers, tokenizers, or diffusers.

"""Qwen3-VL media processor and MiniMax-H3 presentation helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvF

from .tokenizer import BatchEncoding, MiniMaxH3Tokenizer


def _smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    num_frames: int = 1,
    temporal_factor: int = 1,
) -> tuple[int, int]:
    if height <= 0 or width <= 0 or num_frames <= 0 or temporal_factor <= 0:
        raise ValueError("media dimensions must be positive")
    if num_frames < temporal_factor:
        raise ValueError(
            f"num_frames={num_frames} must be at least temporal_factor={temporal_factor}"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute media aspect ratio must be at most 200")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    rounded_height = max(factor, round(height / factor) * factor)
    rounded_width = max(factor, round(width / factor) * factor)
    rounded_frames = math.ceil(num_frames / temporal_factor) * temporal_factor
    pixels = rounded_frames * rounded_height * rounded_width
    if pixels > max_pixels:
        scale = math.sqrt((num_frames * height * width) / max_pixels)
        rounded_height = max(factor, math.floor(height / scale / factor) * factor)
        rounded_width = max(factor, math.floor(width / scale / factor) * factor)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        rounded_height = max(factor, math.ceil(height * scale / factor) * factor)
        rounded_width = max(factor, math.ceil(width * scale / factor) * factor)
    return rounded_height, rounded_width


def _image_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, (str, Path)):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Pillow is a project dependency
            raise RuntimeError("Loading image paths requires Pillow") from exc
        with Image.open(value) as image:
            value = np.asarray(image.convert("RGB")).copy()
    elif hasattr(value, "convert") and hasattr(value, "size"):
        value = np.asarray(value.convert("RGB")).copy()
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(np.ascontiguousarray(value))
    if not torch.is_tensor(value):
        raise TypeError(f"unsupported image input {type(value).__name__}")
    if value.ndim == 2:
        value = value.unsqueeze(0).repeat(3, 1, 1)
    elif value.ndim == 3:
        if value.shape[-1] == 3:
            value = value.permute(2, 0, 1)
        elif value.shape[0] == 3:
            pass
        elif value.shape[-1] in (1, 4):
            value = value.permute(2, 0, 1)
        elif value.shape[0] not in (1, 4):
            raise ValueError("cannot infer image channel dimension")
    if value.ndim != 3 or value.shape[0] not in (1, 3, 4):
        raise ValueError("images must be HWC or CHW with 1, 3, or 4 channels")
    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)
    if value.shape[0] == 4:
        value = value[:3]
    if value.is_floating_point():
        value = value.float()
        upper = 1.0 if not value.numel() or float(value.max()) <= 1.0 else 255.0
        return value.clamp(0, upper).contiguous()
    return value.clamp(0, 255).to(torch.uint8).contiguous()


def _video_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        return torch.stack([_image_tensor(frame) for frame in value], dim=0)
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(np.ascontiguousarray(value))
    if not torch.is_tensor(value):
        raise TypeError(
            "videos must be frame sequences or THWC/TCHW/CTHW tensors; "
            "decode compressed files before calling the local processor"
        )
    if value.ndim != 4:
        raise ValueError("videos must have four dimensions")
    if value.shape[-1] == 3:
        value = value.permute(0, 3, 1, 2)
    elif value.shape[1] == 3:
        pass
    elif value.shape[0] == 3:
        value = value.permute(1, 0, 2, 3)
    elif value.shape[-1] in (1, 4):
        value = value.permute(0, 3, 1, 2)
    elif value.shape[1] in (1, 4):
        pass
    elif value.shape[0] in (1, 4):
        value = value.permute(1, 0, 2, 3)
    else:
        raise ValueError("cannot infer video channel dimension")
    if value.shape[1] == 1:
        value = value.repeat(1, 3, 1, 1)
    if value.shape[1] == 4:
        value = value[:, :3]
    if value.is_floating_point():
        value = value.float()
        upper = 1.0 if not value.numel() or float(value.max()) <= 1.0 else 255.0
        return value.clamp(0, upper).contiguous()
    return value.clamp(0, 255).to(torch.uint8).contiguous()


def _is_single_frame(value: Any) -> bool:
    if hasattr(value, "convert") and hasattr(value, "size"):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim in (2, 3)
    return torch.is_tensor(value) and value.ndim in (2, 3)


class _QwenVisionProcessor:
    def __init__(
        self,
        *,
        size: Mapping[str, int] | None = None,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        image_mean: Sequence[float] = (0.5, 0.5, 0.5),
        image_std: Sequence[float] = (0.5, 0.5, 0.5),
        is_video: bool = False,
        do_resize: bool = True,
        **kwargs: Any,
    ) -> None:
        default_size = (
            {"shortest_edge": 4096, "longest_edge": 25165824}
            if is_video
            else {"shortest_edge": 65536, "longest_edge": 16777216}
        )
        self.size = dict(size or default_size)
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.merge_size = int(merge_size)
        self.image_mean = tuple(float(value) for value in image_mean)
        self.image_std = tuple(float(value) for value in image_std)
        self.is_video = bool(is_video)
        self.do_resize = bool(do_resize)
        self.extra_config = dict(kwargs)
        if self.patch_size < 1 or self.temporal_patch_size < 1 or self.merge_size < 1:
            raise ValueError("patch and merge sizes must be positive")

    def _normalize_resize(
        self, frames: torch.Tensor, *, do_resize: bool | None = None
    ) -> torch.Tensor:
        do_resize = self.do_resize if do_resize is None else bool(do_resize)
        unit_scaled = frames.is_floating_point() and (
            not frames.numel() or float(frames.max()) <= 1.0
        )
        height, width = frames.shape[-2:]
        factor = self.patch_size * self.merge_size
        if do_resize:
            target_height, target_width = _smart_resize(
                height,
                width,
                factor=factor,
                min_pixels=int(self.size["shortest_edge"]),
                max_pixels=int(self.size["longest_edge"]),
                num_frames=frames.shape[0] if self.is_video else 1,
                temporal_factor=self.temporal_patch_size if self.is_video else 1,
            )
            if (target_height, target_width) != (height, width):
                # Keep uint8 pixels through torchvision's bicubic kernel and
                # rescale only afterwards. Qwen's released fast processors do
                # the same; converting to [0,1] before interpolation changes
                # rounding at reference-video resize boundaries.
                frames = tvF.resize(
                    frames,
                    [target_height, target_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
        elif height % factor or width % factor:
            raise ValueError(
                f"media size {height}x{width} must be divisible by patch_size*merge_size={factor}"
            )
        frames = frames.float()
        pixel_scale = 1.0 if unit_scaled else 255.0
        mean = (
            torch.tensor(self.image_mean, dtype=frames.dtype, device=frames.device)[
                None, :, None, None
            ]
            * pixel_scale
        )
        std = (
            torch.tensor(self.image_std, dtype=frames.dtype, device=frames.device)[
                None, :, None, None
            ]
            * pixel_scale
        )
        return (frames - mean) / std

    def _patchify(self, videos: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        # videos: [batch, frames, channels, height, width]
        batch, num_frames, channels, height, width = videos.shape
        padding = -num_frames % self.temporal_patch_size
        if padding:
            videos = torch.cat(
                (videos, videos[:, -1:].expand(-1, padding, -1, -1, -1)), dim=1
            )
            num_frames += padding
        grid_t = num_frames // self.temporal_patch_size
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        if grid_h % self.merge_size or grid_w % self.merge_size:
            raise ValueError("vision patch grid must be divisible by merge_size")
        patches = videos.reshape(
            batch,
            grid_t,
            self.temporal_patch_size,
            channels,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        patches = patches.reshape(
            batch,
            grid_t * grid_h * grid_w,
            channels * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return patches, grid_t, grid_h, grid_w

    def __call__(
        self,
        *,
        images: Any = None,
        videos: Any = None,
        return_tensors: str = "pt",
        do_resize: bool | None = None,
        do_sample_frames: bool = False,
        **_: Any,
    ) -> BatchEncoding:
        del do_sample_frames
        if return_tensors != "pt":
            raise ValueError("the local Qwen vision processor returns PyTorch tensors")
        if self.is_video:
            if videos is None:
                raise ValueError("videos are required")
            if torch.is_tensor(videos) or isinstance(videos, np.ndarray):
                values = [videos]
            else:
                values = list(videos)
                if values and _is_single_frame(values[0]):
                    values = [values]
            processed, grids = [], []
            for value in values:
                frames = self._normalize_resize(
                    _video_tensor(value), do_resize=do_resize
                )
                patches, temporal, height, width = self._patchify(frames.unsqueeze(0))
                processed.append(patches.reshape(-1, patches.shape[-1]))
                grids.append((temporal, height, width))
            return BatchEncoding(
                pixel_values_videos=torch.cat(processed, dim=0),
                video_grid_thw=torch.tensor(grids, dtype=torch.long),
            )
        if images is None:
            raise ValueError("images are required")
        values = (
            [images]
            if torch.is_tensor(images)
            or isinstance(images, np.ndarray)
            or hasattr(images, "convert")
            or isinstance(images, (str, Path))
            else list(images)
        )
        processed, grids = [], []
        for value in values:
            image = self._normalize_resize(
                _image_tensor(value).unsqueeze(0), do_resize=do_resize
            )
            repeated = image[:, None].expand(-1, self.temporal_patch_size, -1, -1, -1)
            patches, temporal, height, width = self._patchify(repeated)
            processed.append(patches.reshape(-1, patches.shape[-1]))
            grids.append((temporal, height, width))
        return BatchEncoding(
            pixel_values=torch.cat(processed, dim=0),
            image_grid_thw=torch.tensor(grids, dtype=torch.long),
        )

    def get_number_of_image_patches(self, height: int, width: int, *_: Any) -> int:
        target_h, target_w = _smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=int(self.size["shortest_edge"]),
            max_pixels=int(self.size["longest_edge"]),
        )
        return target_h // self.patch_size * (target_w // self.patch_size)

    def to_config(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            **self.extra_config,
        }


@dataclass(frozen=True)
class MiniMaxH3Presentation:
    token_ids: tuple[int, ...]
    token_tags: tuple[int, ...]
    vision_inputs: dict[str, torch.Tensor]
    presentation: str


class MiniMaxH3Processor:
    """Own Qwen3-VL processor plus exact H3 request serialization."""

    text_tag = 1
    vision_tag = 0

    def __init__(
        self,
        tokenizer: MiniMaxH3Tokenizer | None = None,
        image_processor: _QwenVisionProcessor | Mapping[str, Any] | None = None,
        video_processor: _QwenVisionProcessor | Mapping[str, Any] | None = None,
        *,
        video_sample_fps: float = 2.0,
    ) -> None:
        self.tokenizer = tokenizer or MiniMaxH3Tokenizer()
        self.image_processor = (
            image_processor
            if isinstance(image_processor, _QwenVisionProcessor)
            else _QwenVisionProcessor(**dict(image_processor or {}), is_video=False)
        )
        self.video_processor = (
            video_processor
            if isinstance(video_processor, _QwenVisionProcessor)
            else _QwenVisionProcessor(**dict(video_processor or {}), is_video=True)
        )
        self.video_sample_fps = float(video_sample_fps)
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.image_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.vision_start_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        tokenizer: MiniMaxH3Tokenizer | None = None,
        **kwargs: Any,
    ) -> MiniMaxH3Processor:
        root = Path(directory).expanduser()
        if tokenizer is None:
            tokenizer = MiniMaxH3Tokenizer.from_pretrained(root)

        def read(name: str) -> dict[str, Any]:
            path = root / name
            return (
                json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            )

        image_config = read("preprocessor_config.json")
        video_config = read("video_preprocessor_config.json")
        for config in (image_config, video_config):
            for key in tuple(config):
                if key in {
                    "processor_class",
                    "image_processor_type",
                    "video_processor_type",
                    "do_convert_rgb",
                    "do_normalize",
                    "do_rescale",
                    "do_sample_frames",
                    "resample",
                    "rescale_factor",
                }:
                    config.pop(key)
        return cls(
            tokenizer=tokenizer,
            image_processor=image_config,
            video_processor=video_config,
            **kwargs,
        )

    def save_pretrained(self, directory: str | Path) -> tuple[str, str]:
        root = Path(directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        image_path = root / "preprocessor_config.json"
        video_path = root / "video_preprocessor_config.json"
        image_config = self.image_processor.to_config()
        image_config.update(
            processor_class="Qwen3VLProcessor",
            image_processor_type="Qwen2VLImageProcessorFast",
        )
        video_config = self.video_processor.to_config()
        video_config.update(
            processor_class="Qwen3VLProcessor",
            video_processor_type="Qwen3VLVideoProcessor",
        )
        image_path.write_text(
            json.dumps(image_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        video_path.write_text(
            json.dumps(video_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(image_path), str(video_path)

    def create_mm_token_type_ids(
        self, input_ids: Sequence[Sequence[int]] | torch.Tensor
    ) -> list[list[int]]:
        if torch.is_tensor(input_ids):
            input_ids = input_ids.detach().cpu().tolist()
        output: list[list[int]] = []
        for row in input_ids:
            output.append(
                [
                    1
                    if int(token_id) == self.image_token_id
                    else 2
                    if int(token_id) == self.video_token_id
                    else 0
                    for token_id in row
                ]
            )
        return output

    def _text(self, value: str) -> tuple[list[int], list[int], str]:
        ids = self.tokenizer.encode(value, add_special_tokens=False)
        return ids, [self.text_tag] * len(ids), value

    def _vision(
        self, token: Literal["<|image_pad|>", "<|video_pad|>"], count: int
    ) -> tuple[list[int], list[int], str]:
        ids = [self.vision_start_token_id]
        ids += [int(self.tokenizer.convert_tokens_to_ids(token))] * int(count)
        ids += [self.vision_end_token_id]
        value = self.vision_start_token + token * int(count) + self.vision_end_token
        return ids, [self.vision_tag] * len(ids), value

    @staticmethod
    def _reference_value(reference: Any, name: str, default: Any = None) -> Any:
        if isinstance(reference, Mapping):
            return reference.get(name, default)
        return getattr(reference, name, default)

    @classmethod
    def _reference_kind(cls, reference: Any) -> str:
        kind = cls._reference_value(reference, "kind") or cls._reference_value(
            reference, "type"
        )
        if kind is None:
            if cls._reference_value(reference, "image") is not None:
                kind = "image"
            elif cls._reference_value(reference, "frames") is not None:
                kind = "video"
            elif cls._reference_value(reference, "waveform") is not None:
                kind = "audio"
        kind = str(kind).lower()
        if kind not in {"image", "video", "audio"}:
            raise ValueError(f"unsupported MiniMax-H3 reference kind {kind!r}")
        return kind

    @classmethod
    def _has_audio(cls, reference: Any, kind: str) -> bool:
        explicit = cls._reference_value(reference, "has_audio")
        if explicit is not None:
            return bool(explicit)
        return kind == "audio" or any(
            cls._reference_value(reference, name) is not None
            for name in ("audio", "waveform")
        )

    def _sample_reference_video(
        self, reference: Any
    ) -> tuple[torch.Tensor, list[float]]:
        frames = self._reference_value(reference, "frames")
        if frames is None:
            frames = self._reference_value(reference, "video")
        values = _video_tensor(frames)
        fps = float(self._reference_value(reference, "fps", self.video_sample_fps))
        if fps <= 0 or self.video_sample_fps <= 0:
            raise ValueError("reference video FPS values must be positive")
        stride = fps / self.video_sample_fps
        indices: list[int] = []
        cursor = 0.0
        while round(cursor) < values.shape[0]:
            current = round(cursor)
            if not indices or current > indices[-1]:
                indices.append(current)
            cursor += stride
        temporal_patch = self.video_processor.temporal_patch_size
        if len(indices) < temporal_patch:
            raise ValueError(
                f"reference video yields {len(indices)} sampled frames, fewer than "
                f"temporal_patch_size={temporal_patch}"
            )
        sampled = values[indices]
        timestamps = [index / self.video_sample_fps for index in range(len(indices))]
        timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
        block_timestamps = [
            (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
            for index in range(0, len(timestamps), temporal_patch)
        ]
        return sampled, block_timestamps

    def encode_presentation(
        self,
        prompt: str,
        *,
        mode: str = "t2va",
        first_frame: Any = None,
        last_frame: Any = None,
        references: Sequence[Any] = (),
    ) -> MiniMaxH3Presentation:
        if not isinstance(prompt, str):
            raise TypeError("MiniMax-H3 prompt must be one string")
        mode = mode.lower()
        if mode not in {"t2va", "i2va", "fl2va", "ref2va"}:
            raise ValueError("mode must be t2va, i2va, fl2va, or ref2va")
        token_ids: list[int] = []
        token_tags: list[int] = []
        rendered: list[str] = []

        def emit(segment: tuple[list[int], list[int], str]) -> None:
            token_ids.extend(segment[0])
            token_tags.extend(segment[1])
            rendered.append(segment[2])

        vision_inputs: dict[str, torch.Tensor] = {}
        if mode == "t2va":
            if first_frame is not None or last_frame is not None or references:
                raise ValueError("t2va does not accept frames or references")
        elif mode in {"i2va", "fl2va"}:
            if references:
                raise ValueError("keyframe modes do not accept omni references")
            keyframes = [
                value for value in (first_frame, last_frame) if value is not None
            ]
            if mode == "i2va" and len(keyframes) != 1:
                raise ValueError("i2va requires exactly one frame")
            if not keyframes:
                raise ValueError(f"{mode} requires at least one keyframe")
            features = self.image_processor(images=keyframes, return_tensors="pt")
            vision_inputs.update(features)
            merge_area = self.image_processor.merge_size**2
            for index, grid in enumerate(features.image_grid_thw):
                emit(self._text(f"<Picture {index + 1}>: "))
                emit(self._vision("<|image_pad|>", int(grid.prod()) // merge_area))
        else:
            references = list(references)
            if not references:
                raise ValueError("ref2va requires at least one reference")
            image_references = [
                reference
                for reference in references
                if self._reference_kind(reference) == "image"
            ]
            image_counts: list[int] = []
            if image_references:
                images = [
                    self._reference_value(reference, "image", reference)
                    for reference in image_references
                ]
                features = self.image_processor(images=images, return_tensors="pt")
                vision_inputs.update(features)
                merge_area = self.image_processor.merge_size**2
                image_counts = [
                    int(grid.prod()) // merge_area for grid in features.image_grid_thw
                ]

            video_references = [
                reference
                for reference in references
                if self._reference_kind(reference) == "video"
            ]
            video_counts: list[int] = []
            video_timestamps: list[list[float]] = []
            if video_references:
                sampled = [
                    self._sample_reference_video(value) for value in video_references
                ]
                video_timestamps = [value[1] for value in sampled]
                features = self.video_processor(
                    videos=[value[0] for value in sampled],
                    return_tensors="pt",
                    do_sample_frames=False,
                )
                vision_inputs.update(features)
                merge_area = self.video_processor.merge_size**2
                video_counts = [
                    int(grid[1] * grid[2]) // merge_area
                    for grid in features.video_grid_thw
                ]
                for timestamps, grid in zip(video_timestamps, features.video_grid_thw):
                    if len(timestamps) != int(grid[0]):
                        raise ValueError(
                            "video timestamp blocks do not match processed grid"
                        )

            counts = {"image": 0, "video": 0, "audio": 0}
            for reference in references:
                kind = self._reference_kind(reference)
                if self._has_audio(reference, kind):
                    counts["audio"] += 1
                    emit(self._text(f"<Audio {counts['audio']}>: "))
                if kind == "image":
                    counts["image"] += 1
                    emit(self._text(f"<Picture {counts['image']}>: "))
                    emit(
                        self._vision("<|image_pad|>", image_counts[counts["image"] - 1])
                    )
                elif kind == "video":
                    counts["video"] += 1
                    index = counts["video"] - 1
                    emit(self._text(f"<Video {counts['video']}>: "))
                    for timestamp in video_timestamps[index]:
                        emit(self._text(f"<{timestamp:.1f} seconds>"))
                        emit(self._vision("<|video_pad|>", video_counts[index]))
        emit(self._text(prompt))
        if len(token_ids) != len(token_tags):
            raise RuntimeError("presentation token/tag lengths diverged")
        return MiniMaxH3Presentation(
            token_ids=tuple(token_ids),
            token_tags=tuple(token_tags),
            vision_inputs=vision_inputs,
            presentation="".join(rendered),
        )

    @property
    def model_input_names(self) -> list[str]:
        return [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "mm_token_type_ids",
        ]


Qwen3VLProcessor = MiniMaxH3Processor

__all__ = [
    "MiniMaxH3Presentation",
    "MiniMaxH3Processor",
    "Qwen3VLProcessor",
]
