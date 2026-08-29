"""Runtime capability checks for the pinned LTX-2.5 source revision."""

from __future__ import annotations

from typing import Any


def require_ltx_torch_capabilities(feature: str, torch_module: Any | None = None) -> None:
    """Fail early when PyTorch cannot import the pinned LTX implementation.

    The reviewed LTX source decorates its DiffVAE attention implementation with
    ``torch.compiler.nested_compile_region`` at module import time. PyTorch
    2.7.x satisfies the upstream package metadata but does not expose that API,
    so without this guard users only see a deep, unactionable ``AttributeError``.
    """

    if torch_module is None:
        import torch as torch_module

    compiler = getattr(torch_module, 'compiler', None)
    capability = getattr(compiler, 'nested_compile_region', None)
    if callable(capability):
        return

    version = getattr(torch_module, '__version__', 'unknown')
    raise RuntimeError(
        f"{feature} cannot use the pinned LTX-2.5 source with PyTorch {version}: "
        "torch.compiler.nested_compile_region is missing. PyTorch 2.7.x is "
        "therefore not compatible with this source revision. Prepare a dedicated "
        "runtime with the pinned official Lightricks/LTX-2 checkout and `uv sync`, "
        "or install a matching PyTorch >=2.8 build for your CUDA platform before "
        "installing the HFTrainer LTX extra."
    )


__all__ = ['require_ltx_torch_capabilities']
