# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

"""MiniMax-H3 packed-sequence geometry shared by training and inference.

MODIFIED BY HFTRAINER: extracted from the upstream modular-pipeline blocks,
made state-free, and expressed with repository-owned dataclasses.  The
float64/numpy arithmetic is intentional: it preserves the released model's
rotary-position convention rather than merely producing close coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
FPS = 24
AUDIO_SAMPLE_RATE = 32_000
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@dataclass(frozen=True)
class MiniMaxH3ReferenceGeometry:
    """Encoded geometry of one Ref2VA reference in request order."""

    kind: Literal["image", "video", "audio"]
    latent_frames: int = 0
    latent_height: int = 0
    latent_width: int = 0
    audio_latents: int = 0

    @property
    def has_audio(self) -> bool:
        return self.audio_latents > 0


@dataclass(frozen=True)
class MiniMaxH3PackedLayout:
    """Structural tensors consumed by ``MiniMaxH3Transformer3DModel``."""

    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int = 0
    num_condition_audio_rows: int = 0

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    @property
    def target_video_indices(self) -> torch.Tensor:
        return self.video_indices[self.num_condition_video_rows :]

    @property
    def target_audio_indices(self) -> torch.Tensor:
        return self.audio_indices[self.num_condition_audio_rows :]

    def to(self, device: torch.device | str) -> MiniMaxH3PackedLayout:
        return MiniMaxH3PackedLayout(
            position_ids=self.position_ids.to(device),
            token_tags=self.token_tags.to(device),
            video_indices=self.video_indices.to(device),
            audio_indices=self.audio_indices.to(device),
            text_indices=self.text_indices.to(device),
            num_condition_video_rows=self.num_condition_video_rows,
            num_condition_audio_rows=self.num_condition_audio_rows,
        )


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    canvas_multiple: int = 32,
    short_edge: int = 768,
    max_pixels: int = 768 * 1344,
    min_aspect_ratio: float = 1 / 4,
    max_aspect_ratio: float = 4,
) -> tuple[int, int]:
    """Resolve a display ratio to the released H3-Base canvas convention."""

    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(
            f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}."
        )
    if canvas_multiple <= 0 or short_edge <= 0 or max_pixels <= 0:
        raise ValueError("Canvas limits must be positive.")
    ratio = aspect_width / aspect_height
    if not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(
            "MiniMax-H3 supports aspect ratios from "
            f"1:{1 / min_aspect_ratio:g} to {max_aspect_ratio:g}:1, got "
            f"{aspect_width}:{aspect_height} ({ratio:g})."
        )
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    multiple = int(canvas_multiple)
    return (
        max(multiple, round(height / multiple) * multiple),
        max(multiple, round(width / multiple) * multiple),
    )


def align_num_frames(
    num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5
) -> int:
    """Round upward to the next legal ``17*n+5`` H3 video length."""

    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if frames_per_chunk <= 0 or not 0 <= latents_per_chunk < frames_per_chunk:
        raise ValueError("Invalid VAE chunk geometry.")
    remainder = num_frames % frames_per_chunk
    return num_frames + (latents_per_chunk - remainder) % frames_per_chunk


def video_latent_num_frames(
    num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5
) -> int:
    """Map legal pixel-frame counts ``17*n+5`` to latent counts ``5*n+2``."""

    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError(
            f"num_frames must be of the form {frames_per_chunk}*n+"
            f"{latents_per_chunk}, got {num_frames}."
        )
    return (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2


def audio_latent_num_frames(
    num_frames: int,
    fps: float = FPS,
    latents_per_second: int = AUDIO_LATENTS_PER_SECOND,
) -> int:
    if num_frames <= 0 or fps <= 0 or latents_per_second <= 0:
        raise ValueError("Audio/video rates and frame count must be positive.")
    return round(num_frames / fps * latents_per_second)


def patchify_video_latents(
    latents: torch.Tensor, patch_size: Sequence[int] = (1, 2, 2)
) -> torch.Tensor:
    """Pack ``B,C,T,H,W`` video latents into frame-major transformer rows."""

    if latents.ndim != 5:
        raise ValueError(f"Expected BCTHW latents, got {tuple(latents.shape)}.")
    patch_t, patch_h, patch_w = (int(value) for value in patch_size)
    batch, channels, frames, height, width = latents.shape
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"Latents of shape {tuple(latents.shape)} are not divisible by "
            f"{(patch_t, patch_h, patch_w)}."
        )
    latents = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(
        batch, -1, channels * patch_t * patch_h * patch_w
    ).contiguous()


def unpatchify_video_latents(
    rows: torch.Tensor,
    *,
    channels: int,
    num_frames: int,
    height: int,
    width: int,
    patch_size: Sequence[int] = (1, 2, 2),
) -> torch.Tensor:
    """Inverse of :func:`patchify_video_latents`."""

    if rows.ndim == 2:
        rows = rows.unsqueeze(0)
    if rows.ndim != 3:
        raise ValueError(f"Expected BND packed rows, got {tuple(rows.shape)}.")
    patch_t, patch_h, patch_w = (int(value) for value in patch_size)
    expected_rows = (num_frames // patch_t) * (height // patch_h) * (width // patch_w)
    expected_dim = channels * patch_t * patch_h * patch_w
    if rows.shape[1:] != (expected_rows, expected_dim):
        raise ValueError(
            f"Packed shape {tuple(rows.shape)} does not describe "
            f"C={channels}, T/H/W={num_frames}/{height}/{width}, "
            f"patch={(patch_t, patch_h, patch_w)}."
        )
    values = rows.reshape(
        rows.shape[0],
        num_frames // patch_t,
        height // patch_h,
        width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    values = values.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return values.reshape(rows.shape[0], channels, num_frames, height, width)


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False)
    return torch.from_numpy(grid * _ROPE_SPATIAL_SCALE).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE
            * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        (torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0))
    )


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
    audio_channels: int,
) -> None:
    if rows.start == rows.stop:
        return
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(audio_channels)
    channel_positions = []
    for channel in range(audio_channels):
        edge = width_grid[0] if channel == 0 else width_grid[-1]
        channel_positions.append(
            torch.full((num_audio_latents,), float(edge), dtype=torch.float64)
        )
    position_ids[rows, 2] = torch.cat(channel_positions)


def build_fl2va_layout(
    text_token_tags: torch.Tensor,
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: Sequence[int] = (1, 2, 2),
    keyframe_anchors: Sequence[Literal["first", "last"]] = (),
    audio_channels: int = AUDIO_CHANNELS,
) -> MiniMaxH3PackedLayout:
    """Build ``[text | keyframes | target audio | target video]``."""

    if text_token_tags.ndim != 1:
        raise ValueError("text_token_tags must be one-dimensional.")
    patch_t, patch_h, patch_w = (int(value) for value in patch_size)
    if num_latent_frames % patch_t:
        raise ValueError("Video latent frames are not divisible by patch_t.")
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text = int(text_token_tags.numel())
    num_condition = len(keyframe_anchors) * rows_per_frame
    num_audio = num_audio_latents * audio_channels
    num_video = num_latent_frames // patch_t * rows_per_frame
    sequence_length = num_text + num_condition + num_audio + num_video

    condition_start = num_text
    audio_start = condition_start + num_condition
    video_start = audio_start + num_audio
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)
    frame_grid, width_grid = _frame_position_grid(
        latent_height, latent_width, patch_h, patch_w
    )
    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text)
        elif anchor == "last":
            spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for offset, multiplier in enumerate(_ROPE_FRAMES_PER_LATENT):
                spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= multiplier
            anchor_time = float(num_text) + float(spans.sum()) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"Unknown keyframe anchor {anchor!r}.")
        rows = slice(
            condition_start + index * rows_per_frame,
            condition_start + (index + 1) * rows_per_frame,
        )
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    _fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_audio_latents,
        float(num_text),
        width_grid,
        audio_channels,
    )
    video_positions = torch.empty(
        num_latent_frames, rows_per_frame, 3, dtype=torch.float64
    )
    video_positions[:, :, 0] = _temporal_position_grid(
        num_latent_frames, float(num_text)
    )[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)

    video_indices = torch.cat(
        (
            torch.arange(condition_start, audio_start),
            torch.arange(video_start, sequence_length),
        )
    )
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.long()
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG
    return MiniMaxH3PackedLayout(
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition,
        0,
    )


def build_ref2va_layout(
    text_token_tags: torch.Tensor,
    references: Sequence[MiniMaxH3ReferenceGeometry],
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: Sequence[int] = (1, 2, 2),
    audio_channels: int = AUDIO_CHANNELS,
) -> MiniMaxH3PackedLayout:
    """Build ``[text | ordered references | target audio | target video]``."""

    if text_token_tags.ndim != 1:
        raise ValueError("text_token_tags must be one-dimensional.")
    _, patch_h, patch_w = (int(value) for value in patch_size)
    num_text = int(text_token_tags.numel())
    target_video_rows = (
        num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    )
    target_audio_rows = num_audio_latents * audio_channels
    reference_video_rows = sum(
        ref.latent_frames
        * (ref.latent_height // patch_h)
        * (ref.latent_width // patch_w)
        for ref in references
        if ref.kind in ("image", "video")
    )
    reference_audio_rows = sum(
        ref.audio_latents * audio_channels for ref in references if ref.has_audio
    )
    sequence_length = (
        num_text
        + reference_video_rows
        + reference_audio_rows
        + target_audio_rows
        + target_video_rows
    )
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(
        latent_height, latent_width, patch_h, patch_w
    )
    video_parts: list[torch.Tensor] = []
    audio_parts: list[torch.Tensor] = []
    cursor = num_text
    rotary_time = float(num_text)
    for reference in references:
        if reference.kind == "image":
            if reference.latent_frames != 1:
                raise ValueError("An image reference must encode to one latent frame.")
            rows_per_frame = (reference.latent_height // patch_h) * (
                reference.latent_width // patch_w
            )
            rows = slice(cursor, cursor + rows_per_frame)
            cursor = rows.stop
            video_parts.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(
                reference.latent_height,
                reference.latent_width,
                patch_h,
                patch_w,
            )
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0
        elif reference.kind == "audio":
            if not reference.has_audio:
                raise ValueError("An audio reference needs audio latents.")
            count = reference.audio_latents * audio_channels
            rows = slice(cursor, cursor + count)
            cursor = rows.stop
            audio_parts.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(
                position_ids,
                rows,
                reference.audio_latents,
                rotary_time,
                target_width_grid,
                audio_channels,
            )
            rotary_time += float(reference.audio_latents)
        elif reference.kind == "video":
            audio_count = reference.audio_latents * audio_channels
            rows_per_frame = (reference.latent_height // patch_h) * (
                reference.latent_width // patch_w
            )
            video_count = reference.latent_frames * rows_per_frame
            audio_rows = slice(cursor, cursor + audio_count)
            video_rows = slice(audio_rows.stop, audio_rows.stop + video_count)
            cursor = video_rows.stop
            if audio_count:
                audio_parts.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_parts.append(torch.arange(video_rows.start, video_rows.stop))
            frame_grid, width_grid = _frame_position_grid(
                reference.latent_height,
                reference.latent_width,
                patch_h,
                patch_w,
            )
            _fill_audio_positions(
                position_ids,
                audio_rows,
                reference.audio_latents,
                rotary_time,
                width_grid,
                audio_channels,
            )
            frame_time = _temporal_position_grid(reference.latent_frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(
                frame_grid.shape[0]
            )
            position_ids[video_rows, 1:] = frame_grid.repeat(reference.latent_frames, 1)
            video_span = sum(
                _ROPE_FRAME_RESCALE
                * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
                for index in range(reference.latent_frames)
            )
            rotary_time += max(float(reference.audio_latents), video_span)
        else:
            raise ValueError(f"Unknown reference kind {reference.kind!r}.")

    audio_start = cursor
    video_start = audio_start + target_audio_rows
    _fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_audio_latents,
        rotary_time,
        target_width_grid,
        audio_channels,
    )
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(
        target_frame_grid.shape[0]
    )
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(
        (*video_parts, torch.arange(video_start, sequence_length))
    )
    audio_indices = torch.cat((*audio_parts, torch.arange(audio_start, video_start)))
    text_indices = torch.arange(num_text)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.long()
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG
    return MiniMaxH3PackedLayout(
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        reference_video_rows,
        reference_audio_rows,
    )


def build_row_timesteps(
    layout: MiniMaxH3PackedLayout,
    *,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float = 0.999,
    condition_audio_timestep: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce one per-row timestep assignment to values plus inverse indices."""

    row_timesteps = torch.full(
        (layout.sequence_length,), float(video_timestep), dtype=torch.float32
    )
    row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = (
        condition_video_timestep
    )
    row_timesteps[layout.audio_indices[layout.num_condition_audio_rows :]] = (
        audio_timestep
    )
    row_timesteps[layout.audio_indices[: layout.num_condition_audio_rows]] = (
        condition_audio_timestep
    )
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_LATENTS_PER_SECOND",
    "AUDIO_SAMPLE_RATE",
    "AUDIO_TAG",
    "FPS",
    "TEXT_TAG",
    "VIDEO_TAG",
    "MiniMaxH3PackedLayout",
    "MiniMaxH3ReferenceGeometry",
    "align_num_frames",
    "audio_latent_num_frames",
    "build_fl2va_layout",
    "build_ref2va_layout",
    "build_row_timesteps",
    "patchify_video_latents",
    "resolve_canvas_size",
    "unpatchify_video_latents",
    "video_latent_num_frames",
]
