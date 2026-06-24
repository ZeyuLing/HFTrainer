#!/usr/bin/env python3
"""Score per-pair MotionStreamer-272 predictions (from ``ms_t2m_h3d272.py``)
with the hftrainer-native ``MotionStreamer272Evaluator`` and compare against the
MotionStreamer paper HumanML3D row.

Predictions are keyed by the deterministic pair index produced by
``MotionStreamer272Evaluator.load_test_pairs()``; this script reconstructs the
same pairs and aligns each ``<idx:06d>.npy`` with its caption + GT, then calls
``evaluate(...)`` directly (no per-name file reuse).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
UNIT_LENGTH = 4
MIN_MOTION_LENGTH = 60

# MotionStreamer paper (HumanML3D, Table) reference row.
PAPER = {"fid": 0.792, "r_top1": 0.491, "r_top3": 0.788, "diversity": 9.220, "mm_dist": 2.992}
PAPER_GT = {"r_top1": 0.706, "diversity": 27.36, "mm_dist": 15.01}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True, help="dir of <idx:06d>.npy per-pair preds")
    p.add_argument("--n_repeats", type=int, default=20)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator

    ev = MotionStreamer272Evaluator(device="cuda")
    pairs = ev.load_test_pairs()
    pred_dir = Path(args.pred_dir)

    captions, real_motions, pred_motions, lengths, pred_lengths = [], [], [], [], []
    skipped = 0
    for idx, (name, caption, gt, ml) in enumerate(pairs):
        pf = pred_dir / f"{idx:06d}.npy"
        if not pf.exists():
            skipped += 1
            continue
        pred = np.load(pf)
        pred_ml = (len(pred) // UNIT_LENGTH) * UNIT_LENGTH
        if pred_ml < MIN_MOTION_LENGTH:
            skipped += 1
            continue
        pred = pred[:pred_ml]
        captions.append(caption)
        real_motions.append(gt)
        pred_motions.append(pred)
        # Released protocol: encode GT at its full m_length, prediction at its own
        # generated length — a short AR sample must NOT truncate the GT reference.
        lengths.append(ml)
        pred_lengths.append(min(ml, len(pred)))

    print(f"[eval] scored pairs={len(captions)} skipped={skipped}", flush=True)
    res = ev.evaluate(
        captions, real_motions, pred_motions, lengths,
        pred_lengths=pred_lengths, n_repeats=args.n_repeats,
    )

    def f(x):
        return f"{x:.4f}" if isinstance(x, float) else x

    print("\n=== MotionStreamer-272 (hftrainer repro) ===")
    print(f"  n = {len(captions)}")
    print(f"  GT(real)  R-Prec={res['r_precision_real']}  Div={f(res['diversity_real'])}  "
          f"MM={f(res['matching_score_real'])}")
    print(f"            (paper GT: R@1 {PAPER_GT['r_top1']}, Div {PAPER_GT['diversity']}, "
          f"MM {PAPER_GT['mm_dist']})")
    print(f"  pred      FID={f(res['fid'])}  R-Prec={res['r_precision_pred']}  "
          f"Div={f(res['diversity_pred'])}  MM={f(res['matching_score_pred'])}")
    print(f"            (paper MS: FID {PAPER['fid']}, R@1 {PAPER['r_top1']}, "
          f"R@3 {PAPER['r_top3']}, Div {PAPER['diversity']}, MM {PAPER['mm_dist']})")

    if args.out_json:
        res["n"] = len(captions)
        res["skipped"] = skipped
        json.dump(res, open(args.out_json, "w"), indent=2, default=str)
        print(f"  -> {args.out_json}")


if __name__ == "__main__":
    main()
