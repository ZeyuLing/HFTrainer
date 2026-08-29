# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Lazy loaders for optional LTX training integrations.

The numerical training path must remain importable when experiment tracking and
remote artifact publishing are disabled.  Keep imports for those integrations
behind the feature gates that use them, and report an actionable error if a user
opts in without installing the corresponding support package.
"""

from __future__ import annotations

from types import ModuleType


class OptionalLTXDependencyError(RuntimeError):
    """Raised when an explicitly enabled LTX integration is not installed."""


def _missing_dependency(
    *,
    package: str,
    feature: str,
) -> OptionalLTXDependencyError:
    return OptionalLTXDependencyError(
        f"{feature} requires the optional support package {package!r}. "
        "Install HFTrainer's integration extras with "
        "`pip install 'hftrainer[ltx-video-integrations]'`, or disable the "
        "corresponding integration in the LTX training config."
    )


def require_wandb(*, feature: str = "LTX Weights & Biases logging") -> ModuleType:
    """Import W&B only after ``wandb.enabled`` selects the integration."""

    try:
        import wandb
    except ModuleNotFoundError as exc:
        if exc.name == "wandb" or (exc.name or "").startswith("wandb."):
            raise _missing_dependency(
                package="wandb",
                feature=feature,
            ) from exc
        raise
    return wandb


def require_huggingface_hub(
    *,
    feature: str = "LTX Hugging Face Hub publishing",
) -> tuple[ModuleType, ModuleType]:
    """Import Hub client modules only after ``hub.push_to_hub`` is enabled."""

    try:
        import huggingface_hub
        from huggingface_hub import utils as hub_utils
    except ModuleNotFoundError as exc:
        if exc.name == "huggingface_hub" or (exc.name or "").startswith(
            "huggingface_hub."
        ):
            raise _missing_dependency(
                package="huggingface-hub",
                feature=feature,
            ) from exc
        raise
    return huggingface_hub, hub_utils


def require_imageio(
    *,
    feature: str = "LTX Hub sample GIF conversion",
) -> ModuleType:
    """Import ImageIO only when Hub samples actually need GIF conversion."""

    try:
        import imageio
    except ModuleNotFoundError as exc:
        if exc.name == "imageio" or (exc.name or "").startswith("imageio."):
            raise _missing_dependency(
                package="imageio",
                feature=feature,
            ) from exc
        raise
    return imageio


__all__ = [
    "OptionalLTXDependencyError",
    "require_huggingface_hub",
    "require_imageio",
    "require_wandb",
]
