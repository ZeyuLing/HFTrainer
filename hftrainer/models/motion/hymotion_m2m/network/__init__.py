"""HyMotion-M2M private network implementation.

This package contains the HunyuanMotionMMDiT transformer and all supporting
modules (attention, encoders, loss, SMPL, text encoder, etc.) that are used
exclusively by the HyMotion-M2M task.
"""

from hftrainer.models.motion.hymotion_m2m.network.hymotion_mmdit import (
    HunyuanMotionMMDiT,
)
from hftrainer.models.motion.hymotion_m2m.network.hymotion_dit import (
    HunyuanMotionDiT,
)
from hftrainer.registry import HF_MODELS

if not HF_MODELS.get('HunyuanMotionMMDiT'):
    HF_MODELS.register_module(
        name='HunyuanMotionMMDiT', module=HunyuanMotionMMDiT, force=True,
    )

if not HF_MODELS.get('HunyuanMotionDiT'):
    HF_MODELS.register_module(
        name='HunyuanMotionDiT', module=HunyuanMotionDiT, force=True,
    )

__all__ = ['HunyuanMotionMMDiT', 'HunyuanMotionDiT']
