"""Typed tuple/key-like outputs for the repository-local MiniMax-H3 stack."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


class ModelOutput:
    """Minimal ``BaseOutput``-compatible surface used by HFTrainer.

    Fields whose value is ``None`` remain addressable by name, while tuple
    iteration follows the usual model-output convention and omits them.
    """

    def _items(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (field.name, getattr(self, field.name))
            for field in fields(self)
            if getattr(self, field.name) is not None
        )

    def __iter__(self):
        return (value for _, value in self._items())

    def __getitem__(self, key: int | slice | str):
        if isinstance(key, str):
            return getattr(self, key)
        return self.to_tuple()[key]

    def __len__(self) -> int:
        return len(self._items())

    def keys(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._items())

    def values(self) -> tuple[Any, ...]:
        return self.to_tuple()

    def items(self) -> tuple[tuple[str, Any], ...]:
        return self._items()

    def to_tuple(self) -> tuple[Any, ...]:
        return tuple(value for _, value in self._items())


@dataclass
class MiniMaxH3TransformerOutput(ModelOutput):
    """Joint video/audio velocity predictions from MiniMax-H3."""

    sample: torch.Tensor
    audio_sample: torch.Tensor | None = None


@dataclass
class MiniMaxH3SchedulerOutput(ModelOutput):
    """Sample produced by one MiniMax-H3 rectified-flow Euler step."""

    prev_sample: torch.Tensor


@dataclass
class AutoencoderKLOutput(ModelOutput):
    """Posterior returned by the visual autoencoder encoder."""

    latent_dist: Any


@dataclass
class DecoderOutput(ModelOutput):
    """Decoded video or waveform sample."""

    sample: torch.Tensor


@dataclass
class MiniMaxH3AudioEncoderOutput(ModelOutput):
    """Posterior returned by the audio autoencoder encoder."""

    latent_dist: Any


__all__ = [
    "AutoencoderKLOutput",
    "DecoderOutput",
    "MiniMaxH3AudioEncoderOutput",
    "MiniMaxH3SchedulerOutput",
    "MiniMaxH3TransformerOutput",
    "ModelOutput",
]
