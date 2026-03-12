"""Tensor helpers for motion components."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def tensor_to_array(x: Any) -> Any:
    """Convert nested tensor structures to NumPy arrays."""

    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        if getattr(x, 'is_meta', False):
            raise ValueError('Cannot convert a meta tensor to NumPy.')
        if x.layout != torch.strided:
            x = x.to_dense()
        if getattr(x, 'is_quantized', False):
            x = x.dequantize()
        x = x.detach().cpu()
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        return x.contiguous().numpy()
    if isinstance(x, Mapping):
        return {k: tensor_to_array(v) for k, v in x.items()}
    if isinstance(x, list):
        return [tensor_to_array(v) for v in x]
    if isinstance(x, tuple):
        return tuple(tensor_to_array(v) for v in x)
    return np.asarray(x)
