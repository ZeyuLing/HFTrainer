"""Typed tuple-like outputs used by local diffusion components."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TextEncoderOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor | None = None

    def __getitem__(self, index: int) -> torch.Tensor:
        return (self.last_hidden_state, self.pooler_output)[index]


@dataclass
class UNet2DConditionOutput:
    sample: torch.Tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        return (self.sample,)[index]


@dataclass
class DecoderOutput:
    sample: torch.Tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        return (self.sample,)[index]


@dataclass
class AutoencoderKLOutput:
    latent_dist: object

    def __getitem__(self, index: int):
        return (self.latent_dist,)[index]


@dataclass
class SchedulerOutput:
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor | None = None

    def __getitem__(self, index: int) -> torch.Tensor:
        return (self.prev_sample, self.pred_original_sample)[index]
