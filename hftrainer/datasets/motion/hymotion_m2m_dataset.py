"""Synthetic HyMotion-M2M dataset for smoke testing.

Generates random src/tgt motion pairs with proper shapes and masks.
For real training, use the full MultiTaskM2MDataset with annotation files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from hftrainer.registry import DATASETS


@DATASETS.register_module()
class HyMotionM2MSyntheticDataset(Dataset):
    """Synthetic dataset that generates random motion pairs for M2M smoke tests.

    Each sample returns:
      - src_motion: (max_frame, motion_dim) source motion
      - tgt_motion: (max_frame, motion_dim) target motion
      - src_mask:   (max_frame, motion_dim) mask (1=needs generation, 0=keep)
      - tgt_length: int
      - src_length: int
    """

    def __init__(
        self,
        num_samples: int = 100,
        max_frame: int = 64,
        motion_dim: int = 135,
        mask_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_frame = max_frame
        self.motion_dim = motion_dim
        self.mask_ratio = mask_ratio

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
                result[key] = vals  # keep as list
            else:
                result[key] = vals
        return result

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        L = self.max_frame
        D = self.motion_dim

        # Random motions
        src_motion = torch.randn(L, D)
        tgt_motion = torch.randn(L, D)

        # Build mask: first half is kept (0), second half needs generation (1)
        split = int(L * (1 - self.mask_ratio))
        src_mask = torch.zeros(L, D)
        src_mask[split:] = 1.0

        return {
            'src_motion': src_motion,
            'tgt_motion': tgt_motion,
            'src_mask': src_mask,
            'tgt_length': L,
            'src_length': L,
        }


def hymotion_m2m_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple collate for the M2M dataset."""
    result = {}
    keys = batch[0].keys()
    for key in keys:
        vals = [item[key] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], (int, float)):
            result[key] = vals  # keep as list
        else:
            result[key] = vals
    return result
