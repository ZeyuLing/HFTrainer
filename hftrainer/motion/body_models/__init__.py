"""Public body-model modules for motion-domain code."""

from hftrainer.motion.body_models.smplx_lite import (
    SmplLite,
    SmplxLite,
    SmplxLiteJ24,
    SmplxLiteV437Coco17,
)
from hftrainer.motion.skeleton.body_models import resolve_smpl_model_dir

__all__ = [
    "resolve_smpl_model_dir",
    "SmplLite",
    "SmplxLite",
    "SmplxLiteJ24",
    "SmplxLiteV437Coco17",
]
