#!/usr/bin/env python3
"""Evaluate native InterHuman-262 2P packs with hftrainer's InterCLIP evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hftrainer.evaluation.evaluators.interhuman_262 import InterHuman262Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, help="Ground-truth native-262 pack.")
    parser.add_argument("--pred", action="append", default=[], help="name=path native-262 pack.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--ckpt-path", default=None, help="Optional InterCLIP checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retrieval-batch-size", type=int, default=32)
    parser.add_argument("--retrieval-reps", type=int, default=20)
    parser.add_argument("--diversity-times", type=int, default=300)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_paths = {}
    for item in args.pred:
        if "=" not in item:
            raise SystemExit(f"--pred must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        pred_paths[name] = path

    evaluator = InterHuman262Evaluator(
        ckpt_path=args.ckpt_path,
        device=args.device,
        batch_size=args.batch_size,
        retrieval_batch_size=args.retrieval_batch_size,
        retrieval_repeats=args.retrieval_reps,
        diversity_times=args.diversity_times,
        max_len=args.max_len,
    )
    results = evaluator.evaluate_npz(args.gt, pred_paths, seed=args.seed)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, row in results.items():
        print(
            f"{name}: R@3={row['rp_top3']:.4f} MM-D={row['mm_dist']:.4f} "
            f"Div={row['diversity']:.4f} FID={row['fid']:.4f}",
            flush=True,
        )
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
