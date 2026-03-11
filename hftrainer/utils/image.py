"""Image helpers that keep basic data paths usable without torchvision."""

from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_image(image: Image.Image, size: Sequence[int]) -> Image.Image:
    """Resize a PIL image to (height, width)."""
    height, width = int(size[0]), int(size[1])
    return image.resize((width, height), Image.BILINEAR)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float tensor in [0, 1]."""
    array = np.array(image, copy=True)
    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor


def normalize_image(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    """Normalize a CHW image tensor."""
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def denormalize_image(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    """Undo image normalization on a CHW tensor."""
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return tensor * std_tensor + mean_tensor


def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    """Save a CHW tensor in [0, 1] to disk via PIL."""
    tensor = tensor.detach().cpu().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    if array.shape[2] == 1:
        array = array[:, :, 0]
    Image.fromarray(array).save(path)
