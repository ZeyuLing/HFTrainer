"""Base classification dataset with MMEngine-style pipelines."""

from abc import ABC
from typing import Dict, Any, List, Optional, Sequence, Union, Callable

import torch

from hftrainer.datasets.base_dataset import PipelineDataset
from hftrainer.utils.image import IMAGENET_MEAN, IMAGENET_STD


class BaseClassificationDataset(PipelineDataset, ABC):
    """
    Abstract base class for classification datasets.

    Subclasses should implement ``load_data_list()`` (and optionally
    ``get_data_info()``) to emit raw fields such as ``img_path`` / ``image``
    and ``label``. Concrete preprocessing is delegated to transforms.
    """

    def __init__(
        self,
        image_size: int = 224,
        pipeline: Optional[Sequence[Union[dict, Callable]]] = None,
        serialize_data: bool = False,
    ):
        self.image_size = image_size
        super().__init__(pipeline=pipeline, serialize_data=serialize_data)

    def build_default_pipeline(self):
        return [
            dict(type='LoadImage'),
            dict(type='ResizeImage', size=(self.image_size, self.image_size)),
            dict(type='HFTrainerImageToTensor', image_key='image', output_key='pixel_values'),
            dict(
                type='NormalizeTensor',
                key='pixel_values',
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
            dict(type='RenameKeys', mapping={'label': 'labels'}),
            dict(
                type='PackMetaKeys',
                meta_keys=('img_path', 'class_name', 'sample_idx'),
            ),
        ]

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default collate function."""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        metas = [item.get('metas', {}) for item in batch]
        return {'pixel_values': pixel_values, 'labels': labels, 'metas': metas}
