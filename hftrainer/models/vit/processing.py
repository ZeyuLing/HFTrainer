"""Repository-local image processor for ViT artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch

from hftrainer.utils.image import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    normalize_image,
    pil_to_tensor,
    resize_image,
)


class ViTImageProcessor:
    """Resize and normalize PIL images without a model framework dependency."""

    def __init__(
        self,
        size: int = 224,
        image_mean: Sequence[float] = IMAGENET_MEAN,
        image_std: Sequence[float] = IMAGENET_STD,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        rescale_factor: float = 1.0 / 255.0,
    ):
        self.size = int(size)
        self.image_mean = tuple(float(value) for value in image_mean)
        self.image_std = tuple(float(value) for value in image_std)
        self.do_resize = bool(do_resize)
        self.do_rescale = bool(do_rescale)
        self.do_normalize = bool(do_normalize)
        self.rescale_factor = float(rescale_factor)

    @classmethod
    def from_pretrained(cls, directory: str | Path, default_size: int = 224):
        path = Path(directory) / 'preprocessor_config.json'
        if not path.is_file():
            return cls(size=default_size)
        with path.open('r', encoding='utf-8') as handle:
            config = json.load(handle)
        raw_size = config.get('size', default_size)
        if isinstance(raw_size, dict):
            raw_size = raw_size.get('height', raw_size.get('shortest_edge', default_size))
        return cls(
            size=raw_size,
            image_mean=config.get('image_mean', IMAGENET_MEAN),
            image_std=config.get('image_std', IMAGENET_STD),
            do_resize=config.get('do_resize', True),
            do_rescale=config.get('do_rescale', True),
            do_normalize=config.get('do_normalize', True),
            rescale_factor=config.get('rescale_factor', 1.0 / 255.0),
        )

    def __call__(self, images: Iterable, return_tensors: str = 'pt'):
        if return_tensors != 'pt':
            raise ValueError("ViTImageProcessor only supports return_tensors='pt'.")
        tensors = []
        for image in images:
            if self.do_resize:
                image = resize_image(image, (self.size, self.size))
            tensor = pil_to_tensor(image)
            if self.do_rescale and self.rescale_factor != 1.0 / 255.0:
                tensor = tensor * (self.rescale_factor * 255.0)
            if self.do_normalize:
                tensor = normalize_image(tensor, self.image_mean, self.image_std)
            tensors.append(tensor)
        if not tensors:
            raise ValueError('At least one image is required.')
        return {'pixel_values': torch.stack(tensors)}

    def save_pretrained(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        config = {
            'size': {'height': self.size, 'width': self.size},
            'image_mean': list(self.image_mean),
            'image_std': list(self.image_std),
            'do_resize': self.do_resize,
            'do_rescale': self.do_rescale,
            'do_normalize': self.do_normalize,
            'rescale_factor': self.rescale_factor,
        }
        with (output / 'preprocessor_config.json').open('w', encoding='utf-8') as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
