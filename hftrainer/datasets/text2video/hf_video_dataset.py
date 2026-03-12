"""Simple video dataset for text-to-video training."""

import os
import json
from typing import Optional

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
        pipeline=None,
        serialize_data: bool = False,
    ):
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.synthetic = synthetic
        self.max_samples = max_samples
        super().__init__(pipeline=pipeline, serialize_data=serialize_data)

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
        records = []
        for i in range(n):
            records.append({
                'video_path': None,
                'text': captions[i % len(captions)],
                'synthetic': True,
                'num_frames': self.num_frames,
                'height': self.height,
                'width': self.width,
            })
        return records

    def _load_from_folder(self, root: str):
        """Load from folder with metadata.jsonl."""
        records = []
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
                    records.append({
                        'video_path': video_path,
                        'text': caption,
                        'synthetic': False,
                        'num_frames': self.num_frames,
                        'height': self.height,
                        'width': self.width,
                    })
        else:
            # No metadata: use synthetic data
            records = self._create_synthetic_samples(8)
        return records

    def build_default_pipeline(self):
        return [dict(type='LoadVideo')]

    def load_data_list(self):
        if self.synthetic:
            records = self._create_synthetic_samples(self.max_samples or 8)
        else:
            records = self._load_from_folder(self.data_root)
            if self.max_samples is not None:
                records = records[:self.max_samples]
        return records
