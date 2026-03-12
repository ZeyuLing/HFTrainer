"""HF ImageFolder dataset for text-to-image."""

import os
import json
from typing import Optional

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
        pipeline=None,
        max_samples: Optional[int] = None,
        image_column: str = 'image',
        text_column: str = 'text',
        use_hf_datasets: bool = False,
        random_horizontal_flip: bool = True,
        serialize_data: bool = False,
    ):
        self.data_root = data_root
        self.split = split
        self.image_column = image_column
        self.text_column = text_column
        self.max_samples = max_samples
        self.use_hf_datasets = use_hf_datasets
        self._hf_dataset = None

        super().__init__(
            image_size=image_size,
            random_horizontal_flip=random_horizontal_flip,
            pipeline=pipeline,
            serialize_data=serialize_data,
        )

    def _load_from_folder(self, root: str):
        """Load from folder with metadata.jsonl."""
        records = []
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
                    records.append({'img_path': img_path, 'text': caption})
        else:
            # No metadata: load all images with empty captions
            img_dir = os.path.join(root, 'images') if os.path.isdir(os.path.join(root, 'images')) else root
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith(self.IMAGE_EXTENSIONS):
                    records.append({'img_path': os.path.join(img_dir, fname), 'text': ''})
        return records

    def _load_from_hf_datasets(self, dataset_name_or_path: str, split: str):
        """Load using HuggingFace datasets library."""
        from datasets import load_dataset
        if os.path.isdir(dataset_name_or_path):
            self._hf_dataset = load_dataset('imagefolder', data_dir=dataset_name_or_path, split=split)
        else:
            self._hf_dataset = load_dataset(dataset_name_or_path, split=split)
        return [{'hf_index': idx} for idx in range(len(self._hf_dataset))]

    def load_data_list(self):
        if self.use_hf_datasets:
            records = self._load_from_hf_datasets(self.data_root, self.split)
        else:
            records = self._load_from_folder(self.data_root)
        if self.max_samples is not None:
            records = records[:self.max_samples]
            if self._hf_dataset is not None and hasattr(self._hf_dataset, 'select'):
                self._hf_dataset = self._hf_dataset.select(range(len(records)))
        return records

    def get_data_info(self, idx: int):
        data_info = super().get_data_info(idx)
        if self._hf_dataset is not None and 'hf_index' in data_info:
            item = self._hf_dataset[data_info['hf_index']]
            return {
                'image': item[self.image_column],
                'text': item.get(self.text_column, item.get('caption', '')),
                'sample_idx': data_info['sample_idx'],
            }
        data_info['sample_idx'] = data_info.get('sample_idx', idx)
        return data_info
