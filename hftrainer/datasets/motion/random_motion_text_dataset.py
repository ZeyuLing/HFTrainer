"""Toy motion-text dataset for PRISM startup tests."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset

from hftrainer.registry import DATASETS


@DATASETS.register_module()
class RandomMotionTextDataset(Dataset):
    """Return fixed-shape random motion samples with captions."""

    def __init__(
        self,
        num_samples: int = 8,
        num_frames: int = 33,
        num_joints: int = 22,
        rot_dim: int = 6,
        captions=None,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.rot_dim = rot_dim
        self.captions = captions or ['a person walks forward', 'a person waves']
        generator = torch.Generator().manual_seed(seed)
        self.motion = torch.randn(
            num_samples,
            num_frames,
            num_joints * rot_dim + 6,
            generator=generator,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict:
        return {
            'motion': self.motion[idx].clone(),
            'num_frames': torch.tensor(self.num_frames, dtype=torch.long),
            'caption': self.captions[idx % len(self.captions)],
        }

