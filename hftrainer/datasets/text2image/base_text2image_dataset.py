"""Base text-to-image dataset interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import Dataset


class BaseText2ImageDataset(Dataset, ABC):
    """
    Abstract base class for text-to-image datasets.

    __getitem__ must return:
        {
            'pixel_values': Tensor[3, H, W],  # normalized image in [-1, 1]
            'text': str,                       # caption / prompt
        }
    """

    def __init__(self, image_size: int = 512, transform=None):
        self.image_size = image_size
        self.transform = transform
        self._build_default_transform()

    def _build_default_transform(self):
        if self.transform is None:
            try:
                from torchvision import transforms
                self.transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),  # [-1, 1]
                ])
            except ImportError:
                self.transform = None

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return dict with 'pixel_values' and 'text'."""

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        texts = [item['text'] for item in batch]
        return {'pixel_values': pixel_values, 'text': texts}
