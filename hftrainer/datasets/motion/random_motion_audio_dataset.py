"""Toy motion-audio dataset for MCM smoke tests.

Returns random motion samples with pre-computed fake audio features,
bypassing the real audio encoder entirely.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from hftrainer.datasets.base_dataset import PipelineDataset
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class RandomMotionAudioDataset(PipelineDataset):
    """Return fixed-shape random motion samples with captions and audio features."""

    def __init__(
        self,
        num_samples: int = 8,
        num_frames: int = 17,
        num_joints: int = 22,
        rot_dim: int = 6,
        audio_feature_dim: int = 768,
        audio_num_frames: int = 10,
        captions: Optional[list] = None,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.rot_dim = rot_dim
        self.audio_feature_dim = audio_feature_dim
        self.audio_num_frames = audio_num_frames
        self.captions = captions or [
            'a person dances to music',
            'a person gestures while talking',
        ]

        generator = torch.Generator().manual_seed(seed)
        self.motion = torch.randn(
            num_samples,
            num_frames,
            num_joints * rot_dim + 6,
            generator=generator,
        )
        self.audio_features = torch.randn(
            num_samples,
            audio_num_frames,
            audio_feature_dim,
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
            'audio_features': self.audio_features[idx].clone(),
        }
