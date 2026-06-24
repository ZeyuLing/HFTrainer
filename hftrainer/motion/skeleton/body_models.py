"""Light SMPL / SMPL-X / SMPL-H body-model loaders and asset resolution.

This module is a skeleton-oriented facade over the public body-model package at
``hftrainer.motion.body_models``. New code that constructs the differentiable
LBS modules directly may import from either public path; legacy
``hftrainer.motion.body_models`` imports are compatibility
wrappers only.

Asset resolution prefers the in-repo ``checkpoints/`` symlinks over ``ref_repo``
so that library code never hard-codes a ``ref_repo`` path.
"""

from __future__ import annotations

import os
from typing import List, Optional


def resolve_smpl_model_dir(override: Optional[str] = None) -> str:
    """Resolve a directory that contains SMPL/SMPL-X/SMPL-H model files.

    Resolution order:
      1. ``override`` argument, if given and existing.
      2. ``$HFTRAINER_SMPL_MODEL_DIR`` env var.
      3. ``checkpoints/smpl_models`` (preferred in-repo symlink).
      4. ``ref_repo/MDM/body_models_nochumpy`` (avoids legacy chumpy pickles).
      5. ``ref_repo/MDM/body_models`` (last resort).

    Returns the first existing path; raises ``FileNotFoundError`` if none exist.
    """
    candidates: List[str] = []
    if override:
        candidates.append(override)
    env = os.environ.get("HFTRAINER_SMPL_MODEL_DIR")
    if env:
        candidates.append(env)
    candidates += [
        "checkpoints/smpl_models",
        "ref_repo/MDM/body_models_nochumpy",
        "ref_repo/MDM/body_models",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "No SMPL model directory found. Tried: "
        + ", ".join(c for c in candidates if c)
        + ". Set $HFTRAINER_SMPL_MODEL_DIR or pass override=."
    )


# Lazy re-export of the differentiable LBS modules so that importing this module
# does not require the (heavy) ``smplx`` dependency unless a model is actually
# constructed.
def __getattr__(name: str):  # PEP 562 module-level lazy attribute
    if name in {"SmplLite", "SmplxLite", "SmplxLiteJ24", "SmplxLiteV437Coco17"}:
        from hftrainer.motion.body_models import smplx_lite

        return getattr(smplx_lite, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "resolve_smpl_model_dir",
    "SmplLite",
    "SmplxLite",
    "SmplxLiteJ24",
    "SmplxLiteV437Coco17",
]
