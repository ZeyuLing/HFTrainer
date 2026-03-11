"""Image-folder dataset for GAN training."""

import os
import random
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from hftrainer.registry import DATASETS
from hftrainer.utils.image import pil_to_tensor, resize_image


@DATASETS.register_module()
class ImageFolderGANDataset(Dataset):
    """
    Generic image-folder dataset for GAN training.

    Supports:
      - recursive folder scanning
      - flat folders
      - class-subfolder layouts
    """

    EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    def __init__(
        self,
        data_root: str,
        image_size: int = 64,
        random_horizontal_flip: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.random_horizontal_flip = random_horizontal_flip
        self.samples = self._scan_images(data_root)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _scan_images(self, root: str) -> List[str]:
        samples = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in sorted(filenames):
                if filename.lower().endswith(self.EXTENSIONS):
                    samples.append(os.path.join(dirpath, filename))
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.samples[idx]
        image = Image.open(path).convert('RGB')
        image = resize_image(image, (self.image_size, self.image_size))
        if self.random_horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        tensor = pil_to_tensor(image)
        tensor = (tensor - 0.5) / 0.5
        return {
            'real_data': tensor,
            'metas': {'path': path, 'idx': idx},
        }

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            'real_data': torch.stack([item['real_data'] for item in batch]),
            'metas': [item.get('metas', {}) for item in batch],
        }
