"""Regression tests for MiniMax-H3 reference media decoding."""

from __future__ import annotations

import contextlib
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import hftrainer.pipelines.minimax_h3.references as reference_module
from hftrainer.pipelines.minimax_h3.references import (
    _decode_audio_file,
    _decode_video_file,
)

SAMPLE_RATE = 8_000


def _stereo_pcm(num_samples: int) -> np.ndarray:
    left = np.full(num_samples, 8_192, dtype=np.int16)
    right = np.full(num_samples, -16_384, dtype=np.int16)
    return np.column_stack((left, right))


def _write_stereo_wav(path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(samples.astype("<i2", copy=False).tobytes())


def test_decode_packed_stereo_wav_preserves_channels_and_normalized_amplitude(
    tmp_path,
):
    pytest.importorskip("av")
    samples = _stereo_pcm(257)
    media = tmp_path / "packed-stereo.wav"
    _write_stereo_wav(media, samples)

    audio, sample_rate = _decode_audio_file(media)

    expected = torch.from_numpy(samples.T.copy()).float() / 32_768
    assert sample_rate == SAMPLE_RATE
    assert audio.shape == (2, 257)
    assert audio.is_contiguous()
    torch.testing.assert_close(audio, expected, rtol=0, atol=1 / 32_768)


def test_decode_empty_stereo_wav_preserves_empty_stream_contract(tmp_path):
    pytest.importorskip("av")
    media = tmp_path / "empty-stereo.wav"
    _write_stereo_wav(media, _stereo_pcm(0))

    audio, sample_rate = _decode_audio_file(media)

    assert sample_rate == SAMPLE_RATE
    assert audio.dtype == torch.float32
    assert audio.shape == (0, 0)


def test_decode_video_container_audio_preserves_stereo_layout(tmp_path):
    av = pytest.importorskip("av")
    samples = _stereo_pcm(257)
    media = tmp_path / "stereo-reference.mkv"

    with av.open(str(media), mode="w") as container:
        video_stream = container.add_stream("ffv1", rate=2)
        video_stream.width = 4
        video_stream.height = 4
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("pcm_s16le", rate=SAMPLE_RATE)
        audio_stream.layout = "stereo"

        for value in (32, 224):
            pixels = np.full((4, 4, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        packed = samples.reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(packed, format="s16", layout="stereo")
        frame.sample_rate = SAMPLE_RATE
        for packet in audio_stream.encode(frame):
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)

    frames, fps, audio, sample_rate = _decode_video_file(media)

    assert frames.shape == (2, 4, 4, 3)
    assert fps == pytest.approx(2.0)
    assert sample_rate == SAMPLE_RATE
    assert audio is not None
    assert audio.shape == (2, 257)
    expected = torch.from_numpy(samples.T.copy()).float() / 32_768
    torch.testing.assert_close(audio, expected, rtol=0, atol=1 / 32_768)


def test_decode_video_applies_display_rotation_and_localizes_url_once(monkeypatch):
    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    class _Frame:
        rotation = 90.0

        def to_rgb(self):
            return self

        def to_ndarray(self):
            return pixels.copy()

    video_stream = SimpleNamespace(average_rate=3, guessed_rate=None)

    class _Container:
        def __init__(self, *, video=(), audio=(), frames=()):
            self.streams = SimpleNamespace(video=list(video), audio=list(audio))
            self.frames = list(frames)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        def decode(self, stream):
            del stream
            return iter(self.frames)

    class _AV:
        def __init__(self):
            self.opened = []

        def open(self, path):
            self.opened.append(path)
            if len(self.opened) == 1:
                return _Container(video=[video_stream], frames=[_Frame(), _Frame()])
            return _Container()

    av = _AV()
    localized = []

    @contextlib.contextmanager
    def _one_local_file(media):
        localized.append(media)
        yield "one-downloaded-reference.mp4"

    monkeypatch.setattr(reference_module, "_import_av", lambda: av)
    monkeypatch.setattr(reference_module, "_local_media_file", _one_local_file)

    frames, fps, audio, sample_rate = _decode_video_file(
        "https://example.invalid/rotated.mp4"
    )

    expected = np.rot90(np.stack([pixels, pixels]), k=-1, axes=(1, 2))
    assert np.array_equal(frames, expected)
    assert frames.flags.c_contiguous
    assert fps == pytest.approx(3.0)
    assert audio is None
    assert sample_rate is None
    assert localized == ["https://example.invalid/rotated.mp4"]
    assert av.opened == [
        "one-downloaded-reference.mp4",
        "one-downloaded-reference.mp4",
    ]
