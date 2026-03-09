"""Base text-to-video dataset interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset


class BaseText2VideoDataset(Dataset, ABC):
    """
    Abstract base class for text-to-video datasets.

    __getitem__ must return:
        {
            'video': Tensor[C, T, H, W] or Tensor[T, C, H, W],  # in [-1, 1]
            'text': str,                                           # caption
        }
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        videos = torch.stack([item['video'] for item in batch])
        texts = [item['text'] for item in batch]
        return {'video': videos, 'text': texts}
