#!/usr/bin/env python3
"""Evaluate HumanML3D motion-to-text caption predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hftrainer.evaluation.evaluators import HumanMLM2TEvaluator  # noqa: E402

DEFAULT_PRED_DIR = REPO / "outputs/evaluation/m2t/humanml3d_official_test/hml263/motiongpt"
DEFAULT_DATA_ROOT = REPO / "ref_repo/CondMDI/dataset/HumanML3D"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--bert-score", action="store_true")
    parser.add_argument("--bert-model-type", default=None)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_file = Path(args.out_file) if args.out_file else pred_dir / "metrics_humanml3d_m2t.json"
    ev = HumanMLM2TEvaluator(
        data_root=args.data_root,
        device=args.device,
        chunk_size=args.chunk_size,
        n_repeats=args.n_repeats,
        seed=args.seed,
        compute_bert_score=args.bert_score,
        bert_model_type=args.bert_model_type,
    )
    metrics = ev.evaluate_dir(
        pred_dir,
        include_matching=not args.text_only,
        batch_size=args.batch_size,
        max_samples=args.max_samples or None,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {out_file}")


if __name__ == "__main__":
    main()
