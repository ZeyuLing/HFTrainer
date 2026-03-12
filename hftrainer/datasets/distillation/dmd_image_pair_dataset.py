"""Dataset for DMD training with optional precomputed regression pairs."""

import json
import os
from typing import List, Optional

import torch

from hftrainer.datasets.text2image.base_text2image_dataset import BaseText2ImageDataset
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class DMDImagePairDataset(BaseText2ImageDataset):
    """Text-image dataset with optional precomputed DMD regression pairs."""

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
        regression_noise_column: str = 'regression_noise',
        regression_target_column: str = 'regression_target_latents',
        regression_text_column: str = 'regression_text',
        random_horizontal_flip: bool = True,
        serialize_data: bool = False,
    ):
        self.data_root = data_root
        self.split = split
        self.max_samples = max_samples
        self.image_column = image_column
        self.text_column = text_column
        self.use_hf_datasets = use_hf_datasets
        self.regression_noise_column = regression_noise_column
        self.regression_target_column = regression_target_column
        self.regression_text_column = regression_text_column
        self._hf_dataset = None
        super().__init__(
            image_size=image_size,
            random_horizontal_flip=random_horizontal_flip,
            pipeline=pipeline,
            serialize_data=serialize_data,
        )

    def build_default_pipeline(self):
        pipeline = super().build_default_pipeline()
        pipeline.extend([
            dict(
                type='LoadOptionalTorchTensor',
                file_key='regression_noise_path',
                output_key='regression_noise',
            ),
            dict(
                type='LoadOptionalTorchTensor',
                file_key='regression_target_latents_path',
                output_key='regression_target_latents',
            ),
        ])
        return pipeline

    def _load_from_folder(self, root: str):
        records = []
        meta_path = os.path.join(root, 'metadata.jsonl')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    records.append({
                        'img_path': os.path.join(root, item[self.image_column]),
                        'text': item.get(self.text_column, item.get('caption', '')),
                        'regression_noise_path': item.get(self.regression_noise_column),
                        'regression_target_latents_path': item.get(self.regression_target_column),
                        'regression_text': item.get(self.regression_text_column),
                        'data_root': self.data_root,
                    })
        else:
            img_dir = os.path.join(root, 'images') if os.path.isdir(os.path.join(root, 'images')) else root
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith(self.IMAGE_EXTENSIONS):
                    records.append({
                        'img_path': os.path.join(img_dir, fname),
                        'text': '',
                        'regression_noise_path': None,
                        'regression_target_latents_path': None,
                        'regression_text': None,
                        'data_root': self.data_root,
                    })
        return records

    def _load_from_hf_datasets(self, dataset_name_or_path: str, split: str):
        from datasets import load_dataset

        if os.path.isdir(dataset_name_or_path):
            self._hf_dataset = load_dataset('imagefolder', data_dir=dataset_name_or_path, split=split)
        else:
            self._hf_dataset = load_dataset(dataset_name_or_path, split=split)
        return [{'hf_index': idx, 'data_root': self.data_root} for idx in range(len(self._hf_dataset))]

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
                'regression_text': item.get(
                    self.regression_text_column,
                    item.get(self.text_column, item.get('caption', '')),
                ),
                'regression_noise_path': None,
                'regression_target_latents_path': None,
                'data_root': self.data_root,
                'sample_idx': data_info['sample_idx'],
            }
        data_info['sample_idx'] = data_info.get('sample_idx', idx)
        data_info['regression_text'] = data_info.get('regression_text') or data_info['text']
        return data_info

    @classmethod
    def collate_fn(cls, batch: List[dict]) -> dict:
        collated = BaseText2ImageDataset.collate_fn(batch)

        regression_text = [item.get('regression_text', item['text']) for item in batch]
        regression_noise = [
            item.get('regression_noise') for item in batch
            if item.get('regression_noise') is not None
        ]
        regression_target_latents = [
            item.get('regression_target_latents') for item in batch
            if item.get('regression_target_latents') is not None
        ]

        collated['regression_text'] = regression_text
        collated['regression_noise'] = (
            torch.stack(regression_noise) if len(regression_noise) == len(batch) else None
        )
        collated['regression_target_latents'] = (
            torch.stack(regression_target_latents)
            if len(regression_target_latents) == len(batch) else None
        )
        return collated
