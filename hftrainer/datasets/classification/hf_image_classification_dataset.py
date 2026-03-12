"""HuggingFace image classification dataset wrapper."""

import os
import itertools
from typing import Optional

from hftrainer.datasets.classification.base_classification_dataset import BaseClassificationDataset
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class HFImageClassificationDataset(BaseClassificationDataset):
    """
    Wraps a HuggingFace dataset for image classification.

    Supports any HF dataset with 'image' and 'label' columns,
    including local ImageFolder datasets.

    Config example:
        dataset=dict(
            type='HFImageClassificationDataset',
            dataset_name_or_path='data/classification/demo',
            split='train',
            image_column='image',
            label_column='label',
            image_size=224,
        )
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = 'train',
        image_column: str = 'image',
        label_column: str = 'label',
        image_size: int = 224,
        pipeline=None,
        max_samples: Optional[int] = None,
        streaming: bool = False,
        serialize_data: bool = False,
    ):
        self.dataset_name_or_path = dataset_name_or_path
        self.split = split
        self.image_column = image_column
        self.label_column = label_column
        self.max_samples = max_samples
        self.streaming = streaming

        # Load dataset
        from datasets import load_dataset
        if os.path.isdir(dataset_name_or_path):
            # Local dataset
            try:
                self.hf_dataset = load_dataset(
                    'imagefolder',
                    data_dir=dataset_name_or_path,
                    split=split,
                )
            except Exception:
                self.hf_dataset = load_dataset(
                    dataset_name_or_path,
                    split=split,
                )
        else:
            if streaming:
                if max_samples is None:
                    raise ValueError(
                        "streaming=True requires max_samples so the dataset remains indexable."
                    )
                stream_ds = load_dataset(
                    dataset_name_or_path,
                    split=split,
                    streaming=True,
                )
                self.hf_dataset = list(itertools.islice(stream_ds, max_samples))
            else:
                self.hf_dataset = load_dataset(
                    dataset_name_or_path,
                    split=split,
                    streaming=False,
                )

        if max_samples is not None and hasattr(self.hf_dataset, 'select'):
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))

        # Build label2id mapping if available
        self.label2id = {}
        self.id2label = {}
        features = getattr(self.hf_dataset, 'features', None)
        label_feature = features.get(label_column, None) if features is not None else None
        if hasattr(label_feature, 'names'):
            names = label_feature.names
            self.id2label = {i: n for i, n in enumerate(names)}
            self.label2id = {n: i for i, n in enumerate(names)}
        super().__init__(
            image_size=image_size,
            pipeline=pipeline,
            serialize_data=serialize_data,
        )

    def load_data_list(self):
        return [{'hf_index': idx} for idx in range(len(self.hf_dataset))]

    def get_data_info(self, idx: int):
        data_info = super().get_data_info(idx)
        item = self.hf_dataset[data_info['hf_index']]
        label = item[self.label_column]
        if isinstance(label, str):
            label = self.label2id.get(label, 0)
        return {
            'image': item[self.image_column],
            'label': int(label),
            'class_name': self.id2label.get(int(label), str(label)),
            'sample_idx': data_info['sample_idx'],
        }
