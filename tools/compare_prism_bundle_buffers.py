"""Compare PRISM bundle-level buffers across checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hftrainer.utils.checkpoint_utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--keys", default="latents_mean,latents_std")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def get_tensor(state, key: str):
    if key in state:
        return state[key]
    bundle = state.get("__bundle_params__")
    if isinstance(bundle, dict) and key in bundle:
        return bundle[key]
    raise KeyError(key)


def main():
    args = parse_args()
    ref = load_checkpoint(args.reference, map_location="cpu")
    cand = load_checkpoint(args.candidate, map_location="cpu")
    result = {}
    for key in [item.strip() for item in args.keys.split(",") if item.strip()]:
        x = get_tensor(ref, key).detach().float().cpu()
        y = get_tensor(cand, key).detach().float().cpu()
        diff = y - x
        result[key] = {
            "shape": list(x.shape),
            "ref_norm": float(x.norm()),
            "candidate_norm": float(y.norm()),
            "delta_norm": float(diff.norm()),
            "max_abs_delta": float(diff.abs().max()),
            "ref_values": x.flatten().tolist(),
            "candidate_values": y.flatten().tolist(),
        }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
