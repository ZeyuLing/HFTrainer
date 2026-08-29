# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Typed, rate-preserving media references for MiniMax-H3 Ref2VA.

MODIFIED BY HFTRAINER: the file/URL decoder is implemented with standard
library I/O plus optional PyAV and never imports an external model framework.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from PIL import Image

from hftrainer.models.minimax_h3.network.layout import FPS


@dataclass
class MiniMaxH3Reference:
    """Marker base class; list order is semantically significant."""


@dataclass
class MiniMaxH3ImageReference(MiniMaxH3Reference):
    image: Image.Image | np.ndarray | torch.Tensor

    kind = "image"
    has_audio = False

    @classmethod
    def from_file(cls, media: str | os.PathLike) -> MiniMaxH3ImageReference:
        with _local_media_file(media) as path, Image.open(path) as image:
            return cls(image=image.convert("RGB").copy())


@dataclass
class MiniMaxH3VideoReference(MiniMaxH3Reference):
    frames: list[Image.Image] | np.ndarray | torch.Tensor
    fps: float | None = None
    audio: torch.Tensor | None = None
    sample_rate: int | None = None

    kind = "video"

    def __post_init__(self) -> None:
        self.fps = float(FPS if self.fps is None else self.fps)
        if self.fps <= 0:
            raise ValueError("A video reference needs a positive frame rate.")
        if (
            self.audio is not None
            and self.sample_rate is not None
            and int(self.sample_rate) <= 0
        ):
            raise ValueError("A soundtrack needs a positive sample rate.")

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    @classmethod
    def from_file(cls, media: str | os.PathLike) -> MiniMaxH3VideoReference:
        frames, fps, audio, sample_rate = _decode_video_file(media)
        return cls(frames=frames, fps=fps, audio=audio, sample_rate=sample_rate)


@dataclass
class MiniMaxH3AudioReference(MiniMaxH3Reference):
    audio: torch.Tensor
    sample_rate: int | None = None

    kind = "audio"
    has_audio = True

    def __post_init__(self) -> None:
        if self.audio.ndim not in (1, 2):
            raise ValueError(
                "Reference audio must be [samples] or [channels, samples]."
            )
        if self.sample_rate is not None and int(self.sample_rate) <= 0:
            raise ValueError("Reference audio needs a positive sample rate.")

    @classmethod
    def from_file(cls, media: str | os.PathLike) -> MiniMaxH3AudioReference:
        audio, sample_rate = _decode_audio_file(media)
        return cls(audio=audio, sample_rate=sample_rate)


@contextlib.contextmanager
def _local_media_file(media: str | os.PathLike):
    value = str(media)
    if not value.startswith(("http://", "https://")):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Reference media does not exist: {path}")
        yield str(path.resolve())
        return

    suffix = Path(unquote(urlparse(value).path)).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        temporary_path = handle.name
    try:
        with (
            urllib.request.urlopen(value, timeout=30) as response,
            open(temporary_path, "wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        yield temporary_path
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temporary_path)


def _import_av():
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "Decoding MiniMax-H3 reference files requires PyAV. Install "
            "HFTrainer's minimax-h3 extra, or pass decoded tensors."
        ) from exc
    return av


def _decode_audio_stream(container, stream) -> tuple[torch.Tensor, int]:
    sample_rate = getattr(stream, "rate", None)
    if sample_rate is None:
        sample_rate = getattr(
            getattr(stream, "codec_context", None), "sample_rate", None
        )
    if sample_rate is None or int(sample_rate) <= 0:
        raise ValueError("Could not determine the reference audio sample rate.")
    sample_rate = int(sample_rate)

    layout = getattr(stream, "layout", None)
    if layout is None:
        layout = getattr(getattr(stream, "codec_context", None), "layout", None)
    if layout is None:
        raise ValueError("Could not determine the reference audio channel layout.")

    # Packed formats such as stereo ``s16`` expose one interleaved ndarray row.
    # Converting every decoded frame to planar float first both preserves the
    # channel dimension and gives consistently normalized samples in [-1, 1].
    av = _import_av()
    resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
    chunks: list[torch.Tensor] = []

    def append_resampled(frames) -> None:
        if frames is None:
            return
        if not isinstance(frames, (list, tuple)):
            frames = (frames,)
        for frame in frames:
            values = torch.from_numpy(frame.to_ndarray()).float()
            if values.ndim == 1:
                values = values.unsqueeze(0)
            chunks.append(values)

    for frame in container.decode(stream):
        append_resampled(resampler.resample(frame))
    append_resampled(resampler.resample(None))

    if not chunks:
        return torch.empty(0, 0, dtype=torch.float32), sample_rate
    return torch.cat(chunks, dim=-1).contiguous(), sample_rate


def _decode_video_file(
    media: str | os.PathLike,
) -> tuple[np.ndarray, float, torch.Tensor | None, int | None]:
    av = _import_av()
    # Keep a downloaded URL in one temporary file for both decode passes. The
    # selected-stream video decode drains its container, so re-open that same
    # local file for audio instead of fetching a remote reference twice.
    with _local_media_file(media) as path:
        with av.open(path) as container:
            video_stream = next(iter(container.streams.video), None)
            if video_stream is None:
                raise ValueError(f"No video stream found in {media}.")
            rate = video_stream.average_rate or video_stream.guessed_rate
            fps = float(rate) if rate is not None else float(FPS)
            frames = []
            rotation = 0.0
            for frame in container.decode(video_stream):
                rotation = float(getattr(frame, "rotation", 0.0) or 0.0)
                frames.append(frame.to_rgb().to_ndarray())

        audio = None
        sample_rate = None
        with av.open(path) as container:
            audio_stream = next(iter(container.streams.audio), None)
            if audio_stream is not None:
                audio, sample_rate = _decode_audio_stream(container, audio_stream)
    if not frames:
        raise ValueError(f"No frames decoded from {media}.")

    frames = np.stack(frames)
    # PyAV reports the counterclockwise display-matrix rotation. Undo it as
    # FFmpeg does for display, snapped to the nearest quarter turn.
    turns = round(rotation / 90.0) % 4
    if turns:
        frames = np.ascontiguousarray(np.rot90(frames, k=-turns, axes=(1, 2)))
    return frames, fps, audio, sample_rate


def _decode_audio_file(media: str | os.PathLike) -> tuple[torch.Tensor, int]:
    av = _import_av()
    with _local_media_file(media) as path, av.open(path) as container:
        stream = next(iter(container.streams.audio), None)
        if stream is None:
            raise ValueError(f"No audio stream found in {media}.")
        return _decode_audio_stream(container, stream)


__all__ = [
    "MiniMaxH3AudioReference",
    "MiniMaxH3ImageReference",
    "MiniMaxH3Reference",
    "MiniMaxH3VideoReference",
]
