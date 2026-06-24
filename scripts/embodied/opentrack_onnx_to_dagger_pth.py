#!/usr/bin/env python3
"""Convert an OpenTrack DAgger MLP ONNX checkpoint into a PyTorch state_dict."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out-pth", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out_pth.exists() and not args.force:
        print(args.out_pth)
        return

    model = onnx.load(args.onnx)
    state_dict = {}
    for init in model.graph.initializer:
        if init.name.startswith("net.") and (init.name.endswith(".weight") or init.name.endswith(".bias")):
            state_dict[init.name] = torch.from_numpy(numpy_helper.to_array(init).copy())
    if not state_dict:
        raise SystemExit(f"No MLP weights found in {args.onnx}")

    args.out_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.out_pth)
    print(args.out_pth)
    print(f"weights={len(state_dict)}")


if __name__ == "__main__":
    main()
