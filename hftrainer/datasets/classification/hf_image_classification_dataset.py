"""HuggingFace image classification dataset wrapper."""

import os
from typing import Dict, Any, Optional, List

import torch
from PIL import Image

from hftrainer.datasets.classification.base_classification_dataset import BaseClassificationDataset
from hftrainer.registry import DATASETS
from hftrainer.utils.image import IMAGENET_MEAN, IMAGENET_STD, normalize_image, pil_to_tensor, resize_image


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
        transform=None,
        max_samples: Optional[int] = None,
        streaming: bool = False,
    ):
        super().__init__(image_size=image_size, transform=transform)

        self.image_column = image_column
        self.label_column = label_column

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
            self.hf_dataset = load_dataset(
                dataset_name_or_path,
                split=split,
                streaming=streaming,
            )

        if max_samples is not None:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))

        # Build label2id mapping if available
        self.label2id = {}
        self.id2label = {}
        if hasattr(self.hf_dataset.features.get(label_column, None), 'names'):
            names = self.hf_dataset.features[label_column].names
            self.id2label = {i: n for i, n in enumerate(names)}
            self.label2id = {n: i for i, n in enumerate(names)}

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.hf_dataset[idx]

        # Get image
        image = item[self.image_column]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transform
        if self.transform is not None:
            pixel_values = self.transform(image)
        else:
            image = resize_image(image, (self.image_size, self.image_size))
            pixel_values = normalize_image(pil_to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)

        # Get label
        label = item[self.label_column]
        if isinstance(label, str):
            label = self.label2id.get(label, 0)

        return {
            'pixel_values': pixel_values,
            'labels': int(label),
            'metas': {'idx': idx},
        }
