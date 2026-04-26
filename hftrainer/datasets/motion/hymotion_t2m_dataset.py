"""Synthetic HyMotion-T2M dataset for smoke testing.

Generates random motion data with proper shapes for text-to-motion training.
Unlike M2M, there is no src_motion/src_mask — only motion + tgt_length.
For real training, use MotionhubMultiTaskMultiAgentDataset with appropriate transforms.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from hftrainer.registry import DATASETS


@DATASETS.register_module()
class HyMotionT2MSyntheticDataset(Dataset):
    """Synthetic dataset that generates random motion for T2M smoke tests.

    Each sample returns:
      - motion: (max_frame, motion_dim) motion sequence
      - tgt_length: int — number of valid frames
    """

    def __init__(
        self,
        num_samples: int = 100,
        max_frame: int = 64,
        motion_dim: int = 135,
        **kwargs,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_frame = max_frame
        self.motion_dim = motion_dim

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate that stacks tensors and keeps scalars as lists."""
        result = {}
        keys = batch[0].keys()
        for key in keys:
            vals = [item[key] for item in batch]
            if isinstance(vals[0], torch.Tensor):
                result[key] = torch.stack(vals, dim=0)
            elif isinstance(vals[0], (int, float)):
                result[key] = vals
            else:
                result[key] = vals
        return result

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        L = self.max_frame
        D = self.motion_dim

        motion = torch.randn(L, D)

        return {
            'motion': motion,
            'tgt_length': L,
        }
