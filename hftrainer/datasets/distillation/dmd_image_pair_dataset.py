"""Dataset for DMD training with optional precomputed regression pairs."""

import json
import os
from typing import Any, Dict, List, Optional

import torch

from hftrainer.datasets.text2image.hf_imagefolder_dataset import HFImageFolderDataset
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class DMDImagePairDataset(HFImageFolderDataset):
    """
    Text-image dataset with optional precomputed DMD regression pairs.

    Metadata entries may additionally provide:
      - regression_noise
      - regression_target_latents
      - regression_text

    Each tensor field should point to a `.pt` file relative to `data_root`.
    """

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
        regression_noise_column: str = 'regression_noise',
        regression_target_column: str = 'regression_target_latents',
        regression_text_column: str = 'regression_text',
    ):
        self.regression_noise_column = regression_noise_column
        self.regression_target_column = regression_target_column
        self.regression_text_column = regression_text_column
        self.sample_records: List[Dict[str, Any]] = []
        super().__init__(
            data_root=data_root,
            split=split,
            image_size=image_size,
            transform=transform,
            max_samples=max_samples,
            image_column=image_column,
            text_column=text_column,
            use_hf_datasets=use_hf_datasets,
        )

    def _load_from_folder(self, root: str):
        self.samples = []
        self.sample_records = []
        meta_path = os.path.join(root, 'metadata.jsonl')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    record = {
                        'image': os.path.join(root, item[self.image_column]),
                        'text': item.get(self.text_column, item.get('caption', '')),
                        'regression_noise': item.get(self.regression_noise_column),
                        'regression_target_latents': item.get(self.regression_target_column),
                        'regression_text': item.get(self.regression_text_column),
                    }
                    self.sample_records.append(record)
                    self.samples.append((record['image'], record['text']))
        else:
            super()._load_from_folder(root)
            self.sample_records = [
                {
                    'image': image,
                    'text': text,
                    'regression_noise': None,
                    'regression_target_latents': None,
                    'regression_text': None,
                }
                for image, text in self.samples
            ]

    def _load_from_hf_datasets(self, dataset_name_or_path: str, split: str):
        super()._load_from_hf_datasets(dataset_name_or_path, split)
        self.sample_records = [
            {
                'image': image,
                'text': text,
                'regression_noise': None,
                'regression_target_latents': None,
                'regression_text': None,
            }
            for image, text in self.samples
        ]

    def _load_optional_tensor(self, maybe_path: Optional[str]):
        if not maybe_path:
            return None
        full_path = maybe_path
        if not os.path.isabs(full_path):
            full_path = os.path.join(self.data_root, full_path)
        if not os.path.exists(full_path):
            return None
        return torch.load(full_path, map_location='cpu')

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        record = self.sample_records[idx]
        item['regression_text'] = record.get('regression_text') or item['text']
        item['regression_noise'] = self._load_optional_tensor(
            record.get('regression_noise')
        )
        item['regression_target_latents'] = self._load_optional_tensor(
            record.get('regression_target_latents')
        )
        return item

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = HFImageFolderDataset.collate_fn(batch)

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
