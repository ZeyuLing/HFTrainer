from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hftrainer.registry import MODELS


class _SingleCodebookQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0 / math.sqrt(max(dim, 1)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        flat_x = x.reshape(-1, self.dim)
        distances = (
            flat_x.pow(2).sum(dim=-1, keepdim=True)
            - 2 * flat_x @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=-1).unsqueeze(0)
        )
        indices = distances.argmin(dim=-1)
        return indices.view(*x.shape[:-1])

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook(indices)


@MODELS.register_module(force=True)
class FSQuantizer(nn.Module):
    """Minimal FSQ-compatible quantizer API for PRISM/VerMo VQ-VAEs.

    The original projects use an FSQ implementation exposed through a simple
    ``codebook_size`` / ``forward`` / ``dequantize`` interface. For HF-Trainer
    integration we only need that external contract, so this wrapper provides
    a single-codebook quantizer whose size is derived from ``levels``.
    """

    def __init__(
        self,
        dim: int,
        levels: Sequence[int] | Iterable[int],
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        levels = list(levels)
        if not levels:
            raise ValueError("FSQuantizer.levels must be a non-empty sequence.")
        self.dim = dim
        self.levels = levels
        self.codebook_size = int(math.prod(levels))
        self.commitment_weight = commitment_weight
        self.num_quantizers = 1
        self.layers = nn.ModuleList(
            [_SingleCodebookQuantizer(dim=dim, codebook_size=self.codebook_size)]
        )

    def _quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_bnd = x.transpose(1, 2).contiguous()
        indices = self.layers[0].encode(x_bnd)
        quantized = self.layers[0].dequantize(indices)
        return quantized, indices

    def forward(self, x: torch.Tensor):
        quantized, indices = self._quantize(x)
        x_bnd = x.transpose(1, 2).contiguous()
        commit_loss = F.mse_loss(quantized.detach(), x_bnd) * self.commitment_weight
        quantized = x_bnd + (quantized - x_bnd).detach()
        quantized = quantized.transpose(1, 2).contiguous()
        return quantized, indices, commit_loss, None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, indices = self._quantize(x)
        return indices

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = self.layers[0].dequantize(indices)
        if indices.ndim == 1:
            return quantized
        if indices.ndim == 2:
            return quantized.permute(0, 2, 1).contiguous()
        if indices.ndim == 3:
            return quantized.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Unsupported index shape for dequantize(): {tuple(indices.shape)}")

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return self.dequantize(indices)
