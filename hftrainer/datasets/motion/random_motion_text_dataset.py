"""Toy motion-text dataset for PRISM startup tests."""

from __future__ import annotations

from typing import Dict

import torch

from hftrainer.datasets.base_dataset import PipelineDataset
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class RandomMotionTextDataset(PipelineDataset):
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
        super().__init__(pipeline=None, serialize_data=False)

    def load_data_list(self):
        return [{'sample_id': idx} for idx in range(self.num_samples)]

    def get_data_info(self, idx) -> Dict:
        return {
            'motion': self.motion[idx].clone(),
            'num_frames': torch.tensor(self.num_frames, dtype=torch.long),
            'caption': self.captions[idx % len(self.captions)],
        }
