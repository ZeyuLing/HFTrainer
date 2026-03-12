"""Toy dataset for VerMo startup tests and examples."""

from __future__ import annotations

from typing import Dict, List

import torch

from hftrainer.datasets.base_dataset import PipelineDataset
from hftrainer.models.vermo.task_utils import ABBR_TASK_MAPPING
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class VermoToyDataset(PipelineDataset):
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
        super().__init__(pipeline=None, serialize_data=False)

    def load_data_list(self):
        return [{'sample_id': idx} for idx in range(self.num_samples)]

    def get_data_info(self, idx) -> Dict:
        task_abbr = self.tasks[idx % len(self.tasks)]
        return {
            'task': ABBR_TASK_MAPPING[task_abbr],
            'caption': self.captions[idx % len(self.captions)],
            'motion': self.motion[idx].clone(),
            'duration': float(self.num_frames) / 30.0,
            'num_person': 1,
            'num_frames': torch.tensor(self.num_frames, dtype=torch.long),
        }

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
