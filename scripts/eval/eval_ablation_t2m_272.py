#!/usr/bin/env python3
"""Eval MS272 T2M metrics for the loss-ablation arms at one epoch, loading the
MotionStreamer272Evaluator once and scoring each arm's pred272 dir with the
name-based matcher. Prints a comparison table and dumps summary.json.

Usage:
  python3 scripts/eval/eval_ablation_t2m_272.py --epoch 5 --n-repeats 20 \
      --root outputs/tmp/20260628_ablation_t2m_eval \
      --arms a0_full a1_velocity_only a2_no_smoothness a3_no_aux_geom
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--arms", nargs="+",
                   default=["a0_full", "a1_velocity_only",
                            "a2_no_smoothness", "a3_no_aux_geom"])
    p.add_argument("--n-repeats", type=int, default=20)
    args = p.parse_args()

    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator
    ev = MotionStreamer272Evaluator(device="cuda")

    root = Path(args.root)
    gt = ev.evaluate_dir("", n_repeats=args.n_repeats, gt_only=True)
    print(f"[GT] R-Prec={gt['r_precision_real']} Div={gt['diversity_real']:.3f} "
          f"MM={gt['matching_score_real']:.3f}", flush=True)

    hdr = f"{'arm':>20} {'n':>4} {'R1':>7} {'R2':>7} {'R3':>7} {'FID':>8} {'MM':>7} {'Div':>7}"
    print(hdr, flush=True)
    rows = []
    for arm in args.arms:
        pdir = root / arm / f"epoch_{args.epoch}" / "pred272"
        if not pdir.exists():
            print(f"{arm:>20}  MISSING {pdir}", flush=True)
            continue
        res = ev.evaluate_dir(str(pdir), n_repeats=args.n_repeats)
        rp = res["r_precision_pred"]
        skipped = res.get("skipped_no_pred", 0)
        scored = (500 - skipped) if isinstance(skipped, int) else "?"
        row = {
            "arm": arm, "scored": scored,
            "R1": float(rp[0]), "R2": float(rp[1]), "R3": float(rp[2]),
            "FID": float(res["fid"]),
            "MM": float(res["matching_score_pred"]),
            "Div": float(res["diversity_pred"]),
        }
        rows.append(row)
        print(f"{arm:>20} {scored:>4} {row['R1']:>7.4f} {row['R2']:>7.4f} "
              f"{row['R3']:>7.4f} {row['FID']:>8.4f} {row['MM']:>7.4f} "
              f"{row['Div']:>7.4f}", flush=True)

    out = root / f"metrics_ms272_epoch{args.epoch}.json"
    json.dump({"epoch": args.epoch,
               "gt": {k: (v if not hasattr(v, "tolist") else v.tolist())
                      for k, v in gt.items() if k in
                      ("r_precision_real", "diversity_real", "matching_score_real")},
               "rows": rows}, open(out, "w"), indent=2, default=str)
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
