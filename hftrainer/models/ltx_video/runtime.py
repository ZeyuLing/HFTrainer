"""Runtime capability checks for the pinned LTX-2.5 source revision."""

from __future__ import annotations

from typing import Any


def nested_compile_region(function, torch_module: Any | None = None):
    """Use PyTorch's nested compiler boundary when available.

    Importing the repository-local network remains possible on older PyTorch
    versions for configuration and artifact inspection. Full LTX execution is
    still guarded by :func:`require_ltx_torch_capabilities` and requires the
    supported runtime declared by the LTX extras.
    """

    if torch_module is None:
        import torch as torch_module
    compiler = getattr(torch_module, 'compiler', None)
    capability = getattr(compiler, 'nested_compile_region', None)
    return capability(function) if callable(capability) else function


def require_ltx_torch_capabilities(feature: str, torch_module: Any | None = None) -> None:
    """Fail early when PyTorch cannot execute the pinned local implementation.

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
        "therefore not compatible with this source revision. Install a matching "
        "PyTorch >=2.8 build for your CUDA platform; no external source checkout "
        "or separately installed LTX package is used."
    )


__all__ = ['nested_compile_region', 'require_ltx_torch_capabilities']
