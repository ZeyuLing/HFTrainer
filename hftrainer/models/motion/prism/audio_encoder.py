"""Pluggable audio encoder wrapper for MCM-style audio conditioning.

Supports BEATs (default), HuBERT, Whisper encoder, and a mock mode
that returns random features for smoke tests.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hftrainer.registry import HF_MODELS


@HF_MODELS.register_module()
class AudioEncoderWrapper(nn.Module):
    """Pluggable wrapper around frozen audio feature extractors.

    Default: BEATs (768-dim per-frame features, ~50 fps).
    When ``pretrained=None``, operates in mock mode and returns random
    features — useful for smoke tests without downloading model weights.

    Args:
        encoder_type: One of ``'beats'``, ``'hubert'``, ``'whisper'``, ``'mock'``.
        pretrained: Path to the pretrained checkpoint. ``None`` enables mock mode.
        feature_dim: Output feature dimension per frame.
        target_sr: Target sample rate for the audio encoder.
        max_audio_seconds: Maximum audio duration in seconds. Longer waveforms
            are truncated.  BEATs uses O(n²) attention so long audio is both
            slow and memory-hungry.  Default 10 s.
    """

    def __init__(
        self,
        encoder_type: str = 'beats',
        pretrained: Optional[str] = None,
        feature_dim: int = 768,
        target_sr: int = 16000,
        max_audio_seconds: float = 10.0,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.feature_dim = feature_dim
        self.target_sr = target_sr
        self.max_audio_seconds = max_audio_seconds
        self._mock = pretrained is None or encoder_type == 'mock'

        if not self._mock:
            self._encoder = self._load_encoder(encoder_type, pretrained)
            # Keep a persistent fp32 copy on CPU to restore after any
            # framework-level dtype casting (e.g. FSDP mixed precision).
            self._encoder_fp32_state = {
                k: v.clone().float().cpu()
                for k, v in self._encoder.state_dict().items()
            }
        else:
            # Dummy parameter so the module is recognised as nn.Module
            self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _load_encoder(self, encoder_type: str, pretrained: str) -> nn.Module:
        if encoder_type == 'beats':
            return self._load_beats(pretrained)
        raise ValueError(
            f"Unsupported audio encoder type: {encoder_type}. "
            "Supported: 'beats', 'mock'."
        )

    @staticmethod
    def _load_beats(checkpoint_path: str) -> nn.Module:
        """Load BEATs encoder from a Microsoft checkpoint."""
        try:
            from hftrainer.models.motion.prism.third_party.beats import (
                BEATs, BEATsConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "BEATs vendored source not found. Check "
                "hftrainer/models/motion/prism/third_party/beats/"
            ) from exc

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        cfg = BEATsConfig(ckpt['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(ckpt['model'])
        model.eval()
        return model

    def _ensure_encoder_fp32(self) -> None:
        """Restore encoder parameters to fp32 if they were cast by FSDP/AMP.

        FSDP ``MixedPrecision`` and Accelerate's ``mixed_precision="bf16"``
        can silently cast *all* module parameters to bf16 — including frozen
        sub-modules that were never wrapped by FSDP.  BEATs attention
        (relative position bias, GRU, softmax) crashes under bf16 on V100
        (``CUDA driver error: invalid argument``).  This method detects the
        cast and restores fp32 from the snapshot taken at init time.
        """
        sample_param = next(self._encoder.parameters(), None)
        if sample_param is not None and sample_param.dtype != torch.float32:
            device = sample_param.device
            self._encoder.load_state_dict(
                {k: v.to(device) for k, v in self._encoder_fp32_state.items()},
                strict=True,
            )

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        sr: int = 16000,
    ) -> torch.Tensor:
        """Encode audio waveform into per-frame features.

        Args:
            waveform: ``[B, T_samples]`` raw audio waveform.
            sr: Sample rate of the input waveform.

        Returns:
            ``[B, N_frames, feature_dim]`` audio feature tensor.
        """
        if self._mock:
            batch_size = waveform.shape[0]
            n_samples = waveform.shape[-1]
            # Approximate 50 fps output for 16 kHz input
            n_frames = max(1, n_samples // (sr // 50)) if n_samples > 0 else 1
            return torch.randn(
                batch_size, n_frames, self.feature_dim,
                device=waveform.device, dtype=waveform.dtype,
            )

        # Truncate to max duration to avoid BEATs O(n²) attention OOM
        max_samples = int(self.max_audio_seconds * sr)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]

        if self.encoder_type == 'beats':
            # BEATs MUST run in fp32 — its attention kernels crash under
            # bf16/fp16 on V100 ("CUDA driver error: invalid argument").
            # Three layers of protection:
            #   1. Restore fp32 params if FSDP/AMP silently cast them
            #   2. Disable autocast context
            #   3. Explicit .float() on input waveform
            self._ensure_encoder_fp32()
            device_type = waveform.device.type
            with torch.autocast(device_type, enabled=False):
                features, _ = self._encoder.extract_features(waveform.float())
            return features

        raise RuntimeError(f"Forward not implemented for encoder_type={self.encoder_type}")
