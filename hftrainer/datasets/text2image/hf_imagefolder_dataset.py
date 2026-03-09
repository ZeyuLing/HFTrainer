"""HF ImageFolder dataset for text-to-image."""

import os
import json
from typing import Dict, Any, Optional

import torch
from PIL import Image

from hftrainer.datasets.text2image.base_text2image_dataset import BaseText2ImageDataset
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class HFImageFolderDataset(BaseText2ImageDataset):
    """
    Simple image+caption dataset that reads from a folder with metadata.jsonl.

    Expected structure:
        data_root/
            images/
                001.jpg
                002.jpg
                ...
            metadata.jsonl   # {"image": "images/001.jpg", "text": "a cat"}

    Config example:
        dataset=dict(
            type='HFImageFolderDataset',
            data_root='data/text2image/demo',
            image_size=512,
        )
    """

    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: int = 512,
        transform=None,
        max_samples: Optional[int] = None,
        image_column: str = 'image',
        text_column: str = 'text',
        use_hf_datasets: bool = False,
    ):
        super().__init__(image_size=image_size, transform=transform)
        self.data_root = data_root
        self.image_column = image_column
        self.text_column = text_column

        self.samples = []  # list of (image_path, caption)

        if use_hf_datasets:
            self._load_from_hf_datasets(data_root, split)
        else:
            self._load_from_folder(data_root)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_from_folder(self, root: str):
        """Load from folder with metadata.jsonl."""
        meta_path = os.path.join(root, 'metadata.jsonl')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    img_path = os.path.join(root, item[self.image_column])
                    caption = item.get(self.text_column, item.get('caption', ''))
                    self.samples.append((img_path, caption))
        else:
            # No metadata: load all images with empty captions
            img_dir = os.path.join(root, 'images') if os.path.isdir(os.path.join(root, 'images')) else root
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith(self.IMAGE_EXTENSIONS):
                    self.samples.append((os.path.join(img_dir, fname), ''))

    def _load_from_hf_datasets(self, dataset_name_or_path: str, split: str):
        """Load using HuggingFace datasets library."""
        from datasets import load_dataset
        if os.path.isdir(dataset_name_or_path):
            ds = load_dataset('imagefolder', data_dir=dataset_name_or_path, split=split)
        else:
            ds = load_dataset(dataset_name_or_path, split=split)
        for item in ds:
            img = item[self.image_column]
            caption = item.get(self.text_column, item.get('caption', ''))
            self.samples.append((img, caption))  # img may be PIL Image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_or_path, caption = self.samples[idx]

        if isinstance(img_or_path, str):
            image = Image.open(img_or_path).convert('RGB')
        elif isinstance(img_or_path, Image.Image):
            image = img_or_path.convert('RGB')
        else:
            import numpy as np
            image = Image.fromarray(img_or_path).convert('RGB')

        if self.transform is not None:
            pixel_values = self.transform(image)
        else:
            import torchvision.transforms.functional as TF
            image = TF.resize(image, [self.image_size, self.image_size])
            pixel_values = TF.to_tensor(image)
            pixel_values = (pixel_values - 0.5) / 0.5  # [-1, 1]

        return {'pixel_values': pixel_values, 'text': caption}
