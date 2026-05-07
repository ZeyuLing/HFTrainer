"""Synthetic dataset for MotionCLIP smoke tests.

Returns (motion, caption, num_frames) tuples with random motions and
template captions. Use only for pipeline / API verification.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from hftrainer.registry import DATASETS


_SMOKE_CAPTIONS = [
    "a person walks forward",
    "a person runs in place",
    "someone is jumping",
    "a person is dancing",
    "a person sits down",
    "a person waves hello",
    "the person performs a kick",
    "a person stands still",
]


@DATASETS.register_module()
class MotionCLIPSyntheticDataset(Dataset):
    """Synthetic dataset that yields (motion, caption, num_frames)."""

    def __init__(
        self,
        num_samples: int = 100,
        max_frame: int = 64,
        motion_dim: int = 135,
        captions: List[str] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_frame = max_frame
        self.motion_dim = motion_dim
        self.captions = list(captions) if captions else list(_SMOKE_CAPTIONS)
        self._rng = random.Random(seed)

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        keys = batch[0].keys()
        for key in keys:
            vals = [item[key] for item in batch]
            if isinstance(vals[0], torch.Tensor):
                result[key] = torch.stack(vals, dim=0)
            else:
                result[key] = vals
        return result

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        L = self.max_frame
        D = self.motion_dim
        motion = torch.randn(L, D)
        caption = self.captions[idx % len(self.captions)]
        num_frames = L
        return {
            'motion': motion,
            'caption': caption,
            'num_frames': num_frames,
        }
