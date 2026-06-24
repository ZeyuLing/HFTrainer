"""Vendored fairmotion / AIST++ motion feature extractors for Music-to-Dance.

Kept self-contained (no ``third_party`` import at eval time) so the M2D scorer is
reproducible from the ``hftrainer`` package alone. Both extractors are bit-for-bit
ports of the code used by the FineDance / Bailando / LODGE evaluations:

* :func:`extract_kinetic_features` -> 66-dim (3 per joint x 22 SMPL body joints)
* :func:`extract_manual_features`  -> 32-dim geometric/contact predicates
"""
from hftrainer.evaluation.motion.m2d_features.kinetic import (
    extract_kinetic_features,
)
from hftrainer.evaluation.motion.m2d_features.manual import (
    extract_manual_features,
)

__all__ = ["extract_kinetic_features", "extract_manual_features"]
