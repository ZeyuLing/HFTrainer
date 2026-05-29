"""Save PRISM weights immediately after pre-FSDP load, before prepare()."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import hftrainer  # noqa: F401
from accelerate import Accelerator
from mmengine.config import Config

from hftrainer.runner.accelerate_runner import AccelerateRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-frozen-modules",
        action="store_true",
        help="Also save frozen nn.Module states such as the PRISM VAE.",
    )
    return parser.parse_args()


def clone_state_to_cpu(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().clone()
        elif isinstance(value, dict):
            out[key] = clone_state_to_cpu(value)
        else:
            out[key] = value
    return out


def tensor_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for module_name, module_state in state.items():
        if not isinstance(module_state, dict):
            continue
        tensors = [value for value in module_state.values() if isinstance(value, torch.Tensor)]
        if not tensors:
            continue
        summary[module_name] = {
            "num_tensors": len(tensors),
            "numel": int(sum(t.numel() for t in tensors)),
            "first_keys": [
                key
                for key, value in list(module_state.items())[:8]
                if isinstance(value, torch.Tensor)
            ],
        }
    return summary


def add_frozen_modules(bundle, state: Dict[str, Any]) -> None:
    for name in getattr(bundle, "_frozen_modules", []):
        module = getattr(bundle, name, None)
        if isinstance(module, torch.nn.Module):
            state[name] = clone_state_to_cpu(module.state_dict())


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    work_dir = getattr(cfg, "work_dir", "work_dirs/default")

    accelerator = Accelerator(mixed_precision="no")
    bundle = AccelerateRunner._build_bundle(cfg.model, accelerator)
    loaded, meta = AccelerateRunner._pre_prepare_load(
        bundle,
        cfg,
        work_dir,
        accelerator,
    )

    state = clone_state_to_cpu(bundle.state_dict_to_save())
    if args.include_frozen_modules:
        add_frozen_modules(bundle, state)
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model_pre_prepare.pt")

    if accelerator.is_main_process:
        torch.save(state, model_path)
        result = {
            "config": args.config,
            "work_dir": work_dir,
            "loaded": bool(loaded),
            "meta": meta,
            "model_path": model_path,
            "summary": tensor_summary(state),
        }
        summary_path = os.path.join(args.output_dir, "pre_prepare_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
