"""Checkpoint I/O for repository-local causal language models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import torch


def _torch_load(path: Path) -> Dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        value = torch.load(path, map_location='cpu')
    if isinstance(value, dict) and isinstance(value.get('state_dict'), dict):
        value = value['state_dict']
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    ):
        raise ValueError(f'{path} does not contain a tensor state dictionary.')
    return value


def _safe_load(path: Path) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(f'{path.name} requires the declared safetensors dependency.') from exc
    return load_file(str(path), device='cpu')


def _load_files(files: Iterable[Path]) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for path in files:
        shard = _safe_load(path) if path.suffix == '.safetensors' else _torch_load(path)
        overlap = set(state).intersection(shard)
        if overlap:
            raise ValueError(f'Duplicate checkpoint keys in {path.name}: {sorted(overlap)[:8]}')
        state.update(shard)
    return state


def load_state_dict(directory: str | Path) -> Dict[str, torch.Tensor]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(
            f'Checkpoint directory not found: {root}. Only local artifacts are supported.'
        )
    for index_name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index_path = root / index_name
        if index_path.is_file():
            with index_path.open('r', encoding='utf-8') as handle:
                index = json.load(handle)
            weight_map = index.get('weight_map')
            if not isinstance(weight_map, dict):
                raise ValueError(f'{index_path} has no valid weight_map.')
            paths = [root / name for name in sorted(set(weight_map.values()))]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f'Missing checkpoint shards: {missing}')
            return _load_files(paths)
    for name in ('model.safetensors', 'pytorch_model.bin'):
        path = root / name
        if path.is_file():
            return _load_files([path])
    raise FileNotFoundError(
        f'No model.safetensors or pytorch_model.bin checkpoint found under {root}.'
    )


def save_state_dict(
    state_dict: Dict[str, torch.Tensor],
    directory: str | Path,
    safe_serialization: bool = True,
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    # Cloning also breaks shared storage between tied embeddings, which the
    # safe format intentionally rejects as ambiguous.
    tensors = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in state_dict.items()
    }
    if safe_serialization:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError('Safe serialization requires the declared safetensors dependency.') from exc
        output = root / 'model.safetensors'
        save_file(tensors, str(output))
    else:
        output = root / 'pytorch_model.bin'
        torch.save(tensors, output)
    return output
