"""Tests for repository-local LTX audio preprocessing."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch

from hftrainer.models.ltx_video.network.model.audio_vae.ops import AudioProcessor
from hftrainer.models.ltx_video.network.types import Audio


def test_audio_processor_uses_no_external_audio_runtime():
    path = Path(
        'hftrainer/models/ltx_video/network/model/audio_vae/ops.py'
    )
    tree = ast.parse(path.read_text(encoding='utf-8'))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
    assert all(name.split('.', 1)[0] != 'torchaudio' for name in imports)


def test_audio_processor_resamples_and_builds_finite_log_mels():
    source_rate = 8_000
    target_rate = 16_000
    time = torch.arange(source_rate, dtype=torch.float32) / source_rate
    waveform = torch.sin(2 * math.pi * 440.0 * time)[None, None, :]
    processor = AudioProcessor(
        target_sample_rate=target_rate,
        mel_bins=32,
        mel_hop_length=160,
        n_fft=400,
    )

    resampled = processor.resample_audio(
        Audio(waveform=waveform, sampling_rate=source_rate)
    )
    mel = processor.waveform_to_mel(
        Audio(waveform=waveform, sampling_rate=source_rate)
    )

    assert resampled.sampling_rate == target_rate
    assert resampled.waveform.shape == (1, 1, target_rate)
    assert mel.shape[:2] == (1, 1)
    assert mel.shape[-1] == 32
    assert torch.isfinite(mel).all()
