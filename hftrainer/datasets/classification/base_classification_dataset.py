"""Base classification dataset interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import Dataset


class BaseClassificationDataset(Dataset, ABC):
    """
    Abstract base class for classification datasets.

    __getitem__ must return:
        {
            'pixel_values': Tensor[3, H, W],  # normalized image tensor
            'labels': int,                     # class index
            'metas': dict (optional)           # image path, class name, etc.
        }
    """

    def __init__(
        self,
        image_size: int = 224,
        transform=None,
    ):
        self.image_size = image_size
        self.transform = transform
        self._build_default_transform()

    def _build_default_transform(self):
        """Build default ImageNet transform if no custom transform given."""
        if self.transform is None:
            try:
                from torchvision import transforms
                self.transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
            except ImportError:
                self.transform = None

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return dict with 'pixel_values', 'labels', optional 'metas'."""

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default collate function."""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        metas = [item.get('metas', {}) for item in batch]
        return {'pixel_values': pixel_values, 'labels': labels, 'metas': metas}
