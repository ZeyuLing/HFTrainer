"""Checkpoint utilities."""

import os
import glob
import re
from typing import Optional


def find_latest_checkpoint(work_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in work_dir.
    Looks for directories named 'checkpoint-{step}' or 'checkpoint-epoch{epoch}'.
    Returns the path with the highest step/epoch number, or None if not found.
    """
    if not os.path.isdir(work_dir):
        return None

    # Look for checkpoint directories
    pattern = os.path.join(work_dir, 'checkpoint-*')
    candidates = glob.glob(pattern)

    if not candidates:
        return None

    def extract_step(path):
        basename = os.path.basename(path)
        # Try 'checkpoint-{number}'
        m = re.match(r'checkpoint-(\d+)$', basename)
        if m:
            return int(m.group(1))
        # Try 'checkpoint-epoch{number}'
        m = re.match(r'checkpoint-epoch(\d+)$', basename)
        if m:
            return int(m.group(1))
        return -1

    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        return None

    latest = max(candidates, key=extract_step)
    if extract_step(latest) < 0:
        return None
    return latest


def load_checkpoint(path: str, map_location='cpu') -> dict:
    """Load a checkpoint file (safetensors or pytorch)."""
    import torch

    if os.path.isfile(path):
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            return load_file(path, device=map_location)
        else:
            return torch.load(path, map_location=map_location)
    elif os.path.isdir(path):
        # Try safetensors first
        st_path = os.path.join(path, 'model.safetensors')
        if os.path.exists(st_path):
            from safetensors.torch import load_file
            return load_file(st_path, device=map_location)
        pt_path = os.path.join(path, 'model.pt')
        if os.path.exists(pt_path):
            import torch
            return torch.load(pt_path, map_location=map_location)
        pt_path = os.path.join(path, 'pytorch_model.bin')
        if os.path.exists(pt_path):
            import torch
            return torch.load(pt_path, map_location=map_location)
    raise FileNotFoundError(f"No checkpoint found at: {path}")


def save_checkpoint(state_dict: dict, path: str, use_safetensors: bool = True):
    """Save a state dict to path."""
    import torch
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    if use_safetensors and path.endswith('.safetensors'):
        from safetensors.torch import save_file
        # safetensors requires flat dict with tensor values only
        flat = {}
        for k, v in state_dict.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
            else:
                flat[k] = v
        save_file(flat, path)
    else:
        torch.save(state_dict, path)
