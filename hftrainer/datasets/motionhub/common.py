from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, List

import torch

from hftrainer.registry import TRANSFORMS


hm3d_pattern = re.compile(
    r"^[^#]+#"
    r"(?:[\w\-']+\/[A-Za-z]+(?:\s[\w\-']+\/[A-Za-z]+)*)"
    r"#(?:-?\d+\.\d+|nan)"
    r"#(?:-?\d+\.\d+|nan)$",
    re.IGNORECASE,
)


def read_json(path: str) -> Any:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def read_txt(path: str) -> str:
    return Path(path).read_text(encoding='utf-8')


def convert_to_tensor(value: Any):
    if isinstance(value, torch.Tensor):
        return value
    if value is None:
        return None
    return torch.as_tensor(value)


class Compose:
    """Minimal transform compose backed by HFTrainer's transform registry."""

    def __init__(self, transforms: Iterable[Any] | None):
        self.transforms: List[Any] = []
        for transform in transforms or []:
            if isinstance(transform, dict):
                self.transforms.append(TRANSFORMS.build(transform))
            else:
                self.transforms.append(transform)

    def __call__(self, data: dict) -> dict:
        for transform in self.transforms:
            data = transform(data)
            if data is None:
                return None
        return data
