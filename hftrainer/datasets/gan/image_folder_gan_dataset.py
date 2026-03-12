"""Image-folder dataset for GAN training."""

import os
from typing import Dict, List, Optional

import torch

from hftrainer.datasets.base_dataset import PipelineDataset
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class ImageFolderGANDataset(PipelineDataset):
    """Generic image-folder dataset for GAN training."""

    EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    def __init__(
        self,
        data_root: str,
        image_size: int = 64,
        random_horizontal_flip: bool = True,
        max_samples: Optional[int] = None,
        pipeline=None,
        serialize_data: bool = False,
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.random_horizontal_flip = random_horizontal_flip
        self.max_samples = max_samples
        super().__init__(pipeline=pipeline, serialize_data=serialize_data)

    def _scan_images(self, root: str) -> List[str]:
        samples = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in sorted(filenames):
                if filename.lower().endswith(self.EXTENSIONS):
                    samples.append(os.path.join(dirpath, filename))
        return sorted(samples)

    def build_default_pipeline(self):
        pipeline = [
            dict(type='LoadImage'),
            dict(type='ResizeImage', size=(self.image_size, self.image_size)),
        ]
        if self.random_horizontal_flip:
            pipeline.append(dict(type='RandomHorizontalFlipImage'))
        pipeline.extend([
            dict(type='HFTrainerImageToTensor', image_key='image', output_key='real_data'),
            dict(
                type='NormalizeTensor',
                key='real_data',
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            dict(type='PackMetaKeys', meta_keys=('img_path', 'sample_idx')),
        ])
        return pipeline

    def load_data_list(self):
        samples = self._scan_images(self.data_root)
        if self.max_samples is not None:
            samples = samples[:self.max_samples]
        return [{'img_path': path} for path in samples]

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            'real_data': torch.stack([item['real_data'] for item in batch]),
            'metas': [item.get('metas', {}) for item in batch],
        }
