"""Toy dataset for VerMo startup tests and examples."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from hftrainer.models.motion.vermo_task_utils import ABBR_TASK_MAPPING
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class VermoToyDataset(Dataset):
    """Emit small synthetic VerMo samples for core text-motion tasks."""

    def __init__(
        self,
        tasks: List[str] = None,
        num_samples: int = 8,
        num_frames: int = 17,
        num_joints: int = 22,
        rot_dim: int = 6,
        seed: int = 0,
    ):
        self.tasks = tasks or ['t2m', 'm2t']
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.rot_dim = rot_dim
        generator = torch.Generator().manual_seed(seed)
        self.motion = torch.randn(
            num_samples,
            num_frames,
            num_joints * rot_dim + 6,
            generator=generator,
        )
        self.captions = [
            'a person raises both hands',
            'a person takes two steps forward',
            'a person turns left',
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict:
        task_abbr = self.tasks[idx % len(self.tasks)]
        sample = {
            'task': ABBR_TASK_MAPPING[task_abbr],
            'caption': self.captions[idx % len(self.captions)],
            'motion': self.motion[idx].clone(),
            'duration': float(self.num_frames) / 30.0,
            'num_person': 1,
            'num_frames': torch.tensor(self.num_frames, dtype=torch.long),
        }
        return sample

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        output = {'task': [item['task'] for item in batch]}
        for key in ('caption',):
            output[key] = [item[key] for item in batch]
        for key in ('duration', 'num_person'):
            output[key] = [item[key] for item in batch]
        for key in ('motion', 'num_frames'):
            output[key] = torch.stack([item[key] for item in batch], dim=0)
        return output
