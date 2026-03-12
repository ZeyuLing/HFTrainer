"""Simple image folder dataset (no HF dependency)."""

import os
from typing import Optional

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
        pipeline=None,
        max_samples: Optional[int] = None,
        label_file: Optional[str] = None,
        serialize_data: bool = False,
    ):
        self.data_root = data_root
        self.split = split
        self.max_samples = max_samples
        self.label_file = label_file

        self.classes = []
        self.class_to_idx = {}
        super().__init__(
            image_size=image_size,
            pipeline=pipeline,
            serialize_data=serialize_data,
        )

    def _load_from_folder(self, root: str):
        """Load from ImageFolder structure: root/class_name/image.jpg"""
        records = []
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
                        records.append({
                            'img_path': os.path.join(class_dir, fname),
                            'label': self.class_to_idx[class_name],
                            'class_name': class_name,
                        })
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
                records = [
                    {
                        'img_path': p,
                        'label': self.class_to_idx.get(str(l), int(l) if str(l).isdigit() else 0),
                        'class_name': str(l),
                    }
                    for p, l in items
                ]
            else:
                # Just load all images with label 0
                for fname in sorted(os.listdir(root)):
                    if fname.lower().endswith(self.EXTENSIONS):
                        records.append({
                            'img_path': os.path.join(root, fname),
                            'label': 0,
                            'class_name': 'unknown',
                        })
                self.classes = ['unknown']
                self.class_to_idx = {'unknown': 0}
        return records

    def _load_from_label_file(self, label_file: str):
        records = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = os.path.join(self.data_root, parts[0])
                    label = int(parts[1])
                    records.append({
                        'img_path': img_path,
                        'label': label,
                        'class_name': str(label),
                    })
        return records

    def load_data_list(self):
        if self.label_file and os.path.exists(self.label_file):
            records = self._load_from_label_file(self.label_file)
        elif os.path.isdir(self.data_root):
            records = self._load_from_folder(self.data_root)
        else:
            records = []
        if self.max_samples is not None:
            records = records[:self.max_samples]
        return records
