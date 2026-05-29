"""Compare PRISM checkpoint parameter deltas against a reference checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hftrainer.utils.checkpoint_utils import load_checkpoint


WATCH_KEYS = (
    "patch_embedding.weight",
    "condition_embedder.time_embedder.linear_1.weight",
    "condition_embedder.text_embedder.linear_1.weight",
    "blocks.0.attn1.to_q.weight",
    "blocks.0.attn2.to_q.weight",
    "blocks.0.ffn.net.0.proj.weight",
    "blocks.15.attn1.to_q.weight",
    "blocks.31.attn1.to_q.weight",
    "head.modulation",
    "head.head.weight",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--module", default="transformer")
    parser.add_argument("--output", default=None)
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def module_state(checkpoint: str, module: str) -> Dict[str, torch.Tensor]:
    state = load_checkpoint(checkpoint, map_location="cpu")
    if module in state and isinstance(state[module], dict):
        state = state[module]
    else:
        prefix = module + "."
        state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
    return {
        key: value.detach().float().cpu()
        for key, value in state.items()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }


def iter_common(ref: Dict[str, torch.Tensor], cand: Dict[str, torch.Tensor]) -> Iterable[str]:
    for key in ref:
        if key in cand and ref[key].shape == cand[key].shape:
            yield key


def main():
    args = parse_args()
    ref = module_state(args.reference, args.module)
    cand = module_state(args.candidate, args.module)

    total_delta_sq = 0.0
    total_ref_sq = 0.0
    per_key = []
    for key in iter_common(ref, cand):
        delta = cand[key] - ref[key]
        delta_norm = float(delta.pow(2).sum().sqrt())
        ref_norm = float(ref[key].pow(2).sum().sqrt())
        rel = delta_norm / (ref_norm + 1e-12)
        total_delta_sq += delta_norm * delta_norm
        total_ref_sq += ref_norm * ref_norm
        per_key.append({
            "key": key,
            "shape": list(ref[key].shape),
            "delta_norm": delta_norm,
            "ref_norm": ref_norm,
            "relative_delta": rel,
            "candidate_abs_max": float(cand[key].abs().max()),
        })

    per_key.sort(key=lambda row: row["delta_norm"], reverse=True)
    watch = {
        key: next((row for row in per_key if row["key"] == key), None)
        for key in WATCH_KEYS
    }
    result = {
        "reference": args.reference,
        "candidate": args.candidate,
        "module": args.module,
        "num_reference_tensors": len(ref),
        "num_candidate_tensors": len(cand),
        "num_common_tensors": len(per_key),
        "total_delta_norm": total_delta_sq ** 0.5,
        "total_reference_norm": total_ref_sq ** 0.5,
        "total_relative_delta": (total_delta_sq ** 0.5) / ((total_ref_sq ** 0.5) + 1e-12),
        "watch": watch,
        "top_delta": per_key[: args.topk],
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
