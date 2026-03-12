"""Image loading and preprocessing transforms."""

from __future__ import annotations

import random
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image

from hftrainer.registry import TRANSFORMS
from hftrainer.utils.image import normalize_image, pil_to_tensor, resize_image


@TRANSFORMS.register_module()
class LoadImage:
    """Load a PIL image from a path or normalize an existing image object."""

    def __init__(
        self,
        image_key: str = 'image',
        path_key: str = 'img_path',
        to_rgb: bool = True,
    ):
        self.image_key = image_key
        self.path_key = path_key
        self.to_rgb = to_rgb

    def __call__(self, results: dict) -> dict:
        image = results.get(self.image_key)
        if image is None and self.path_key in results:
            image = results[self.path_key]

        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            results[self.image_key] = image
            return results

        if not isinstance(image, Image.Image):
            raise TypeError(f'Unsupported image value type: {type(image)!r}')
        if self.to_rgb and image.mode != 'RGB':
            image = image.convert('RGB')
        results[self.image_key] = image
        return results


@TRANSFORMS.register_module()
class ResizeImage:
    """Resize a PIL image or CHW tensor."""

    def __init__(self, size: Iterable[int], image_key: str = 'image'):
        self.size = tuple(size)
        self.image_key = image_key

    def __call__(self, results: dict) -> dict:
        image = results[self.image_key]
        if isinstance(image, Image.Image):
            results[self.image_key] = resize_image(image, self.size)
            return results
        if isinstance(image, torch.Tensor):
            if image.ndim != 3:
                raise ValueError(f'Expected CHW tensor, got shape {tuple(image.shape)}')
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=self.size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            results[self.image_key] = image
            return results
        raise TypeError(f'Unsupported image type for ResizeImage: {type(image)!r}')


@TRANSFORMS.register_module()
class RandomHorizontalFlipImage:
    """Random horizontal flip for images."""

    def __init__(self, prob: float = 0.5, image_key: str = 'image'):
        self.prob = prob
        self.image_key = image_key

    def __call__(self, results: dict) -> dict:
        if random.random() >= self.prob:
            return results
        image = results[self.image_key]
        if isinstance(image, Image.Image):
            results[self.image_key] = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif isinstance(image, torch.Tensor):
            results[self.image_key] = torch.flip(image, dims=(-1,))
        else:
            raise TypeError(f'Unsupported image type for RandomHorizontalFlipImage: {type(image)!r}')
        return results


@TRANSFORMS.register_module()
class HFTrainerImageToTensor:
    """Convert a PIL image to CHW float tensor in [0, 1]."""

    def __init__(
        self,
        image_key: str = 'image',
        output_key: str = 'pixel_values',
        pop_input: bool = True,
    ):
        self.image_key = image_key
        self.output_key = output_key
        self.pop_input = pop_input

    def __call__(self, results: dict) -> dict:
        image = results.pop(self.image_key) if self.pop_input else results[self.image_key]
        if isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.max().item() > 1:
                tensor = tensor / 255.0
        else:
            tensor = pil_to_tensor(image)
        results[self.output_key] = tensor
        return results


@TRANSFORMS.register_module()
class NormalizeTensor:
    """Normalize a tensor key with mean/std."""

    def __init__(
        self,
        key: str,
        mean,
        std,
    ):
        self.key = key
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, results: dict) -> dict:
        tensor = results[self.key].float()
        results[self.key] = normalize_image(tensor, self.mean, self.std)
        return results
