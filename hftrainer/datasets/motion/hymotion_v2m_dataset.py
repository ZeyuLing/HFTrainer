"""Synthetic dataset for HyMotion-V2M smoke / sanity runs.

Produces random pre-extracted SAM-3D feature streams (T, feature_dim) together
with identity camera extrinsics, matching the input contract of
``HyMotionV2MPipeline.infer_from_feature`` (and the vendored
``MotionGenerationV2M.generate``).  This lets the smoke test exercise the full
feature-to-motion path without any real video / SAM-3D-Body dependency.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset

from hftrainer.registry import DATASETS


@DATASETS.register_module()
class HyMotionV2MSyntheticDataset(Dataset):
    """Random feature/camera streams for HyMotion-V2M feature-to-motion.

    Args:
        num_samples: number of synthetic samples.
        num_frames: temporal length T of each feature stream.
        feature_dim: SAM-3D context token dim (1024 for the released model).
        seed: base RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 4,
        num_frames: int = 40,
        feature_dim: int = 1024,
        seed: int = 0,
    ):
        self.num_samples = int(num_samples)
        self.num_frames = int(num_frames)
        self.feature_dim = int(feature_dim)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed + idx)
        feature = torch.randn(self.num_frames, self.feature_dim, generator=g)
        camera_RT = torch.eye(4).unsqueeze(0).repeat(self.num_frames, 1, 1)
        return {
            "feature": feature,
            "camera_RT": camera_RT,
            "length": self.num_frames,
            "index": idx,
        }
