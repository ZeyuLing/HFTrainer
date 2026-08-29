"""Base text-to-image dataset with MMEngine-style pipelines."""

from abc import ABC
from typing import Callable, Dict, Any, List, Optional, Sequence, Union

import torch

from hftrainer.datasets.base_dataset import PipelineDataset


class BaseText2ImageDataset(PipelineDataset, ABC):
    """
    Abstract base class for text-to-image datasets.

    Subclasses should emit raw ``img_path`` / ``image`` and ``text`` fields.
    Concrete image preprocessing is delegated to transforms.
    """

    def __init__(
        self,
        image_size: int = 512,
        random_horizontal_flip: bool = True,
        pipeline: Optional[Sequence[Union[dict, Callable]]] = None,
        serialize_data: bool = False,
    ):
        self.image_size = image_size
        self.random_horizontal_flip = random_horizontal_flip
        super().__init__(pipeline=pipeline, serialize_data=serialize_data)

    def build_default_pipeline(self):
        pipeline = [
            dict(type='LoadImage'),
            dict(type='ResizeImage', size=(self.image_size, self.image_size)),
        ]
        if self.random_horizontal_flip:
            pipeline.append(dict(type='RandomHorizontalFlipImage'))
        pipeline.extend([
            dict(type='HFTrainerImageToTensor', image_key='image', output_key='pixel_values'),
            dict(
                type='NormalizeTensor',
                key='pixel_values',
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])
        return pipeline

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        texts = [item['text'] for item in batch]
        return {'pixel_values': pixel_values, 'text': texts}
