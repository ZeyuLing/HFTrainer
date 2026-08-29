# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Pure-PyTorch audio preprocessing used by the local LTX audio VAE."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from hftrainer.models.ltx_video.network.types import Audio


def _hz_to_slaney_mel(frequencies: torch.Tensor) -> torch.Tensor:
    linear_scale = frequencies / (200.0 / 3.0)
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / (200.0 / 3.0)
    log_step = math.log(6.4) / 27.0
    log_scale = min_log_mel + torch.log(
        frequencies.clamp_min(torch.finfo(frequencies.dtype).tiny) / min_log_hz
    ) / log_step
    return torch.where(frequencies >= min_log_hz, log_scale, linear_scale)


def _slaney_mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    linear_hz = mels * (200.0 / 3.0)
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / (200.0 / 3.0)
    log_step = math.log(6.4) / 27.0
    log_hz = min_log_hz * torch.exp(log_step * (mels - min_log_mel))
    return torch.where(mels >= min_log_mel, log_hz, linear_hz)


def _slaney_mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> torch.Tensor:
    """Create the Slaney-normalized triangular filter bank used by LTX."""

    dtype = torch.float64
    frequencies = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1, dtype=dtype)
    mel_min = _hz_to_slaney_mel(torch.tensor(0.0, dtype=dtype))
    mel_max = _hz_to_slaney_mel(torch.tensor(sample_rate / 2.0, dtype=dtype))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, dtype=dtype)
    hz_points = _slaney_mel_to_hz(mel_points)

    lower = hz_points[:-2]
    center = hz_points[1:-1]
    upper = hz_points[2:]
    down = (frequencies[:, None] - lower[None, :]) / (center - lower)[None, :]
    up = (upper[None, :] - frequencies[:, None]) / (upper - center)[None, :]
    filters = torch.clamp(torch.minimum(down, up), min=0.0)
    filters *= (2.0 / (upper - lower))[None, :]
    return filters.to(torch.float32)


def _resample_waveform(
    waveform: torch.Tensor,
    source_rate: int,
    target_rate: int,
    *,
    kernel_radius: int = 12,
) -> torch.Tensor:
    """Band-limited, differentiable resampling implemented only with PyTorch."""

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError('Audio sampling rates must be positive integers.')
    if source_rate == target_rate:
        return waveform

    source_length = waveform.shape[-1]
    target_length = max(1, math.ceil(source_length * target_rate / source_rate))
    compute_dtype = (
        waveform.dtype
        if waveform.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    device = waveform.device
    positions = (
        torch.arange(target_length, device=device, dtype=compute_dtype)
        * (source_rate / target_rate)
    )
    centers = positions.floor().to(torch.long)
    offsets = torch.arange(
        -kernel_radius + 1,
        kernel_radius + 1,
        device=device,
        dtype=torch.long,
    )
    indices = centers[:, None] + offsets[None, :]
    distances = positions[:, None] - indices.to(compute_dtype)

    cutoff = min(1.0, target_rate / source_rate) * 0.99
    sinc = cutoff * torch.sinc(cutoff * distances)
    window = torch.cos(
        distances.clamp(-kernel_radius, kernel_radius)
        * (math.pi / (2.0 * kernel_radius))
    ).square()
    weights = sinc * window * (distances.abs() < kernel_radius)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(compute_dtype).eps
    )

    flat = waveform.to(compute_dtype).reshape(-1, source_length)
    padded = F.pad(flat, (kernel_radius, kernel_radius), mode='replicate')
    gathered = padded[:, indices.clamp(-kernel_radius, source_length + kernel_radius - 1) + kernel_radius]
    result = (gathered * weights.unsqueeze(0)).sum(dim=-1)
    return result.reshape(*waveform.shape[:-1], target_length).to(waveform.dtype)


class AudioProcessor(nn.Module):
    """Convert audio waveforms to LTX log-mel spectrograms locally."""

    def __init__(
        self,
        target_sample_rate: int,
        mel_bins: int,
        mel_hop_length: int,
        n_fft: int,
    ) -> None:
        super().__init__()
        self.target_sample_rate = int(target_sample_rate)
        self.mel_hop_length = int(mel_hop_length)
        self.n_fft = int(n_fft)
        self.register_buffer(
            '_window',
            torch.hann_window(self.n_fft, periodic=True),
            persistent=False,
        )
        self.register_buffer(
            '_mel_filterbank',
            _slaney_mel_filterbank(
                sample_rate=self.target_sample_rate,
                n_fft=self.n_fft,
                n_mels=int(mel_bins),
            ),
            persistent=False,
        )

    def resample_audio(self, audio: Audio) -> Audio:
        """Resample audio without an external audio package."""

        if audio.sampling_rate == self.target_sample_rate:
            return audio
        resampled = _resample_waveform(
            audio.waveform,
            int(audio.sampling_rate),
            self.target_sample_rate,
        )
        return Audio(waveform=resampled, sampling_rate=self.target_sample_rate)

    def waveform_to_mel(self, audio: Audio) -> torch.Tensor:
        """Convert waveform to log-mel tensor ``[batch, channels, time, mel]``."""

        waveform = self.resample_audio(audio).waveform
        compute_dtype = (
            waveform.dtype
            if waveform.dtype in (torch.float32, torch.float64)
            else torch.float32
        )
        source = waveform.to(compute_dtype)
        prefix_shape = source.shape[:-1]
        flat_source = source.reshape(-1, source.shape[-1])
        spectrum = torch.stft(
            flat_source,
            n_fft=self.n_fft,
            hop_length=self.mel_hop_length,
            win_length=self.n_fft,
            window=self._window.to(device=source.device, dtype=compute_dtype),
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()
        mel = torch.matmul(
            spectrum.transpose(-1, -2),
            self._mel_filterbank.to(device=source.device, dtype=compute_dtype),
        )
        mel = torch.log(mel.clamp_min(1e-5))
        mel = mel.reshape(*prefix_shape, mel.shape[-2], mel.shape[-1])
        return mel.to(waveform.dtype).contiguous()


class PerChannelStatistics(nn.Module):
    """Normalize and denormalize the audio latent representation by channel."""

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.register_buffer('std-of-means', torch.ones(latent_channels))
        self.register_buffer('mean-of-means', torch.zeros(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer('std-of-means').to(x)) + self.get_buffer(
            'mean-of-means'
        ).to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer('mean-of-means').to(x)) / self.get_buffer(
            'std-of-means'
        ).to(x)
