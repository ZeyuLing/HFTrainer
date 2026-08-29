"""Base text-to-video dataset with MMEngine-style pipelines."""

from abc import ABC
from typing import Callable, Dict, Any, List, Optional, Sequence, Union

import torch

from hftrainer.datasets.base_dataset import PipelineDataset


class BaseText2VideoDataset(PipelineDataset, ABC):
    """
    Abstract base class for text-to-video datasets.

    Subclasses should emit raw ``video_path`` / metadata and ``text``. Concrete
    decoding is delegated to transforms.
    """

    def __init__(
        self,
        pipeline: Optional[Sequence[Union[dict, Callable]]] = None,
        serialize_data: bool = False,
    ):
        super().__init__(pipeline=pipeline, serialize_data=serialize_data)

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        videos = torch.stack([item['video'] for item in batch])
        texts = [item['text'] for item in batch]
        return {'video': videos, 'text': texts}
