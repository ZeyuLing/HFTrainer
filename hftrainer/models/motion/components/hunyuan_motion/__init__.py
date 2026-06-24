"""Shared HunyuanMotion components used by T2M/M2M/UMO bundles.

This package is the stable import surface for HunyuanMotion backbone, text,
geometry, body-model, and loss utilities. Compatibility modules under
``hymotion_m2m.network`` still exist for older code paths.
"""

from hftrainer.models.motion.hymotion_m2m.network.hymotion_dit import HunyuanMotionDiT
from hftrainer.models.motion.hymotion_m2m.network.hymotion_mmdit import HunyuanMotionMMDiT
from hftrainer.registry import HF_MODELS

if not HF_MODELS.get("HunyuanMotionMMDiT"):
    HF_MODELS.register_module(
        name="HunyuanMotionMMDiT", module=HunyuanMotionMMDiT, force=True,
    )

if not HF_MODELS.get("HunyuanMotionDiT"):
    HF_MODELS.register_module(
        name="HunyuanMotionDiT", module=HunyuanMotionDiT, force=True,
    )

__all__ = ["HunyuanMotionMMDiT", "HunyuanMotionDiT"]
