"""Self-contained StyleGAN2 artifact I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

FORMAT = 'hftrainer-stylegan2-v1'
CONFIG_NAME = 'stylegan2_config.json'
SAFE_WEIGHTS_NAME = 'model.safetensors'
TORCH_WEIGHTS_NAME = 'model.pt'


def save_artifact(path: str | Path, config: Mapping, state_dict: Mapping, *, safe: bool) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    payload = {'format': FORMAT, 'model': dict(config)}
    (root / CONFIG_NAME).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tensors = {name: value.detach().cpu().contiguous() for name, value in state_dict.items()}
    if safe:
        from safetensors.torch import save_file

        save_file(tensors, str(root / SAFE_WEIGHTS_NAME))
    else:
        torch.save(tensors, root / TORCH_WEIGHTS_NAME)
    return root


def load_artifact(path: str | Path) -> tuple[dict, dict[str, torch.Tensor]]:
    root = Path(path)
    config_path = root / CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing StyleGAN2 artifact config: {config_path}')
    payload = json.loads(config_path.read_text(encoding='utf-8'))
    if payload.get('format') != FORMAT:
        raise ValueError(f'Unsupported StyleGAN2 artifact format: {payload.get("format")!r}')
    safe_path = root / SAFE_WEIGHTS_NAME
    torch_path = root / TORCH_WEIGHTS_NAME
    if safe_path.is_file():
        from safetensors.torch import load_file

        weights = load_file(str(safe_path), device='cpu')
    elif torch_path.is_file():
        weights = torch.load(torch_path, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f'Missing {SAFE_WEIGHTS_NAME} or {TORCH_WEIGHTS_NAME} in {root}')
    return dict(payload['model']), weights


__all__ = ['CONFIG_NAME', 'FORMAT', 'load_artifact', 'save_artifact']
