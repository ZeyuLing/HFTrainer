"""Body model and classifier imports for quality checkers.

All imports come from hftrainer. No fallback — import errors surface immediately.
"""

from __future__ import annotations

from hftrainer.motion.body_models.smplx_lite import (
    SmplxLiteJ24,
    batch_rigid_transform_v2,
)

from hftrainer.evaluation.quality_check_rules.rotation_classifier import (
    JointRotationClassifier,
)

__all__ = [
    "SmplxLiteJ24",
    "batch_rigid_transform_v2",
    "JointRotationClassifier",
]
