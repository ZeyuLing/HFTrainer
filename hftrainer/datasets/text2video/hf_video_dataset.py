"""Simple video dataset for text-to-video training."""

import os
import json
from typing import Dict, Any, Optional, List

import torch
import numpy as np

from hftrainer.datasets.text2video.base_text2video_dataset import BaseText2VideoDataset
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class HFVideoDataset(BaseText2VideoDataset):
    """
    Simple video dataset that loads frames from a folder with metadata.jsonl.

    For demo/smoke test: can generate synthetic video tensors if no actual videos exist.

    Config example:
        dataset=dict(
            type='HFVideoDataset',
            data_root='data/text2video/demo',
            num_frames=16,
            height=64,
            width=64,
            synthetic=True,  # use synthetic data for smoke test
        )
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_frames: int = 16,
        height: int = 64,
        width: int = 64,
        max_samples: Optional[int] = None,
        synthetic: bool = False,
    ):
        self.data_root = data_root
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.synthetic = synthetic

        self.samples = []  # list of (video_path_or_None, caption)

        if synthetic:
            self._create_synthetic_samples(max_samples or 8)
        else:
            self._load_from_folder(data_root)
            if max_samples is not None:
                self.samples = self.samples[:max_samples]

    def _create_synthetic_samples(self, n: int):
        """Create synthetic video samples (random tensors with captions)."""
        captions = [
            'a cat walking on grass',
            'ocean waves at sunset',
            'a bird flying in the sky',
            'rain falling on leaves',
            'a person dancing',
            'fire burning brightly',
            'clouds moving across sky',
            'a river flowing through forest',
        ]
        for i in range(n):
            self.samples.append((None, captions[i % len(captions)]))

    def _load_from_folder(self, root: str):
        """Load from folder with metadata.jsonl."""
        meta_path = os.path.join(root, 'metadata.jsonl')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    video_path = os.path.join(root, item.get('video', ''))
                    caption = item.get('text', item.get('caption', ''))
                    self.samples.append((video_path, caption))
        else:
            # No metadata: use synthetic data
            self._create_synthetic_samples(8)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path, caption = self.samples[idx]

        if video_path is None or not os.path.exists(video_path) or self.synthetic:
            # Generate synthetic video (random noise + color tint)
            color = torch.rand(3, 1, 1, 1)  # random color per sample
            video = torch.randn(3, self.num_frames, self.height, self.width) * 0.1
            video = video + (color - 0.5) * 2  # shift to [-1, 1] range
            video = video.clamp(-1, 1)
        else:
            video = self._load_video(video_path)

        return {'video': video, 'text': caption}

    def _load_video(self, path: str) -> torch.Tensor:
        """Load video from file. Returns Tensor[C, T, H, W] in [-1, 1]."""
        try:
            import torchvision.io as io
            frames, _, _ = io.read_video(path, output_format='TCHW', pts_unit='sec')
            # Sample frames uniformly
            T = frames.shape[0]
            indices = torch.linspace(0, T - 1, self.num_frames).long()
            frames = frames[indices]  # [num_frames, C, H, W]

            # Resize
            import torchvision.transforms.functional as TF
            frames = torch.stack([
                TF.resize(f, [self.height, self.width]) for f in frames
            ])  # [T, C, H, W]

            # Rearrange and normalize: [C, T, H, W], [-1, 1]
            video = frames.float().permute(1, 0, 2, 3) / 127.5 - 1.0
            return video.clamp(-1, 1)

        except Exception:
            # Fallback to synthetic
            color = torch.rand(3, 1, 1, 1)
            video = torch.randn(3, self.num_frames, self.height, self.width) * 0.1
            video = video + (color - 0.5) * 2
            return video.clamp(-1, 1)
