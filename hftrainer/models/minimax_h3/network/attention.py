"""Optional kernel-only attention matching the released H3 inference runtime."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _flash_attention_3():
    try:
        from kernels import get_kernel

        return get_kernel(
            "kernels-community/sgl-flash-attn3",
            revision="15c17db0bf9ce6599db795fa02a8f27467c92860",
        ).flash_attn_varlen_func
    except (ImportError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "flash_attention_3 requires the kernels package and a compatible "
            "CUDA/PyTorch kernel build (validated with PyTorch 2.11, CUDA 13.0). "
            "Choose sdpa explicitly for the portable backend."
        ) from exc


def flash_attention_3(query, key, value, *, causal=False):
    """B,H,T,D tensors; supports GQA and unequal cached key lengths."""
    if query.device.type != "cuda":
        raise ValueError("flash_attention_3 requires CUDA tensors")
    batch, heads, length, dim = query.shape
    key_length = key.shape[2]
    cu_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32) * length
    cu_k = torch.arange(batch + 1, device=query.device, dtype=torch.int32) * key_length
    def flatten(x):
        return x.transpose(1, 2).reshape(-1, x.shape[1], x.shape[3]).contiguous()
    output = _flash_attention_3()(
        flatten(query), flatten(key), flatten(value), cu_q, cu_k,
        length, key_length, causal=causal,
    )
    return output.reshape(batch, length, heads, dim).transpose(1, 2)


def set_attention_backend(model, backend):
    """Configure repository-owned attention modules without replacing forwards."""
    if backend not in {"sdpa", "flash_attention_3"}:
        raise ValueError("attention_backend must be sdpa or flash_attention_3")
    for module in model.modules():
        if hasattr(module, "attention_backend"):
            module.attention_backend = backend
