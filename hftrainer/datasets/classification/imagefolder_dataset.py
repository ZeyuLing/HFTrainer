"""Simple image folder dataset (no HF dependency)."""

import os
import glob
from typing import Dict, Any, Optional, List

import torch
from PIL import Image

from hftrainer.datasets.classification.base_classification_dataset import BaseClassificationDataset
from hftrainer.registry import DATASETS


@DATASETS.register_module()
class ImageFolderDataset(BaseClassificationDataset):
    """
    Simple ImageFolder dataset. Expects:
        data_root/
            class_a/img1.jpg
            class_b/img2.jpg
            ...
    or a flat directory with metadata.jsonl / labels.txt.
    """

    EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: int = 224,
        transform=None,
        max_samples: Optional[int] = None,
        label_file: Optional[str] = None,
    ):
        super().__init__(image_size=image_size, transform=transform)
        self.data_root = data_root

        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        if label_file and os.path.exists(label_file):
            self._load_from_label_file(label_file)
        elif os.path.isdir(data_root):
            self._load_from_folder(data_root)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_from_folder(self, root: str):
        """Load from ImageFolder structure: root/class_name/image.jpg"""
        class_dirs = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        if class_dirs:
            self.classes = class_dirs
            self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}
            for class_name in class_dirs:
                class_dir = os.path.join(root, class_name)
                for fname in sorted(os.listdir(class_dir)):
                    if fname.lower().endswith(self.EXTENSIONS):
                        self.samples.append((
                            os.path.join(class_dir, fname),
                            self.class_to_idx[class_name]
                        ))
        else:
            # Flat directory: try metadata.jsonl
            meta_path = os.path.join(root, 'metadata.jsonl')
            labels_path = os.path.join(root, 'labels.txt')

            if os.path.exists(meta_path):
                import json
                labels_set = set()
                items = []
                with open(meta_path) as f:
                    for line in f:
                        item = json.loads(line.strip())
                        img_path = os.path.join(root, item['image'])
                        label = item.get('label', item.get('class', 0))
                        labels_set.add(str(label))
                        items.append((img_path, str(label)))

                self.classes = sorted(labels_set)
                self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
                self.samples = [
                    (p, self.class_to_idx.get(str(l), int(l) if str(l).isdigit() else 0))
                    for p, l in items
                ]
            else:
                # Just load all images with label 0
                for fname in sorted(os.listdir(root)):
                    if fname.lower().endswith(self.EXTENSIONS):
                        self.samples.append((os.path.join(root, fname), 0))
                self.classes = ['unknown']

    def _load_from_label_file(self, label_file: str):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = os.path.join(self.data_root, parts[0])
                    label = int(parts[1])
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            pixel_values = self.transform(image)
        else:
            import torchvision.transforms.functional as TF
            pixel_values = TF.to_tensor(TF.resize(image, [self.image_size, self.image_size]))

        return {
            'pixel_values': pixel_values,
            'labels': label,
            'metas': {'path': img_path, 'idx': idx},
        }
