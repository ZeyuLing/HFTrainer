#!/usr/bin/env python3
"""Backward-compat shim. Canonical implementation now lives in hftrainer.

Use ``hftrainer.models.motion.components.utils.humanml_repr`` directly in new
code. This module re-exports the SMPL-H FK joint recovery for existing scripts.
"""
from __future__ import annotations

from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: F401
    recover_272_to_smplh_joints as decode_272_to_smplh_joints,
    recover_local_rotations_and_root,
    fk_smplh_joints,
    _rotation_6d_to_matrix,
    _accumulate_rotations,
)
