"""Inspect non-transformer state in PRISM checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hftrainer.utils.checkpoint_utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def tensor_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    data = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "norm": float(data.norm()),
        "mean": float(data.mean()) if data.numel() else 0.0,
        "std": float(data.std()) if data.numel() > 1 else 0.0,
        "abs_max": float(data.abs().max()) if data.numel() else 0.0,
        "has_nan": bool(torch.isnan(data).any()),
        "has_inf": bool(torch.isinf(data).any()),
    }


def inspect(path: str) -> Dict[str, Any]:
    state = load_checkpoint(path, map_location="cpu")
    row: Dict[str, Any] = {
        "path": path,
        "top_keys": list(state.keys())[:20],
        "has_bundle_params": "__bundle_params__" in state,
        "bundle_params": {},
        "module_keys": {},
    }
    bundle_params = state.get("__bundle_params__")
    if isinstance(bundle_params, dict):
        for key, value in bundle_params.items():
            row["bundle_params"][key] = tensor_summary(value) if torch.is_tensor(value) else repr(type(value))
    for key, value in state.items():
        if isinstance(value, dict):
            row["module_keys"][key] = len(value)
    return row


def main():
    args = parse_args()
    result = [inspect(path) for path in args.checkpoints]
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
