"""Lightweight smoothing stub used by SMPLPoseProcessor."""

from __future__ import annotations

from typing import Optional

import numpy as np

from hftrainer.registry import MODELS


@MODELS.register_module(name=['SmoothNetFilter', 'smoothnet'], force=True)
class SmoothNetFilter:
    """
    Optional smoothing hook.

    The original VersatileMotion repo uses a pretrained SmoothNet checkpoint for
    post-processing. HF-Trainer keeps the hook surface so configs remain
    compatible, but defaults to identity behavior unless the user explicitly
    replaces this class with a stronger implementation.
    """

    def __init__(
        self,
        window_size: int,
        output_size: int,
        checkpoint: Optional[str] = None,
        hidden_size: int = 512,
        res_hidden_size: int = 512,
        num_blocks: int = 5,
        device: str = 'cpu',
    ):
        self.window_size = window_size
        self.output_size = output_size
        self.checkpoint = checkpoint
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks = num_blocks
        self.device = device

    def __call__(self, x):
        return np.asarray(x)
