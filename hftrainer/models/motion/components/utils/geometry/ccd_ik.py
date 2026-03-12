"""Placeholder IK helper for optional post-hoc refinement."""

from __future__ import annotations

import torch


def ik(local_mat: torch.Tensor, target_pos, target_rot, target_ind, chain, parents):
    """
    Keep the public hook expected by SMPLPoseProcessor.

    HF-Trainer motion integration does not depend on the post-hoc IK refinement
    path for training or default inference, so the migrated implementation
    currently returns the original local transforms unchanged.
    """
    return local_mat
