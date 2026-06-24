#!/usr/bin/env python3
"""Per-epoch MotionStreamer-272 trend eval using the CORRECT name-based matcher.

Uses ``MotionStreamer272Evaluator.evaluate_dir``, which pairs each test-split
``name`` with ``<pred_dir>/<name>.npy`` (the real HumanML3D sample id that
gen_ours_m2m_272.py writes). This avoids the eval_ms_h3d272.py bug where preds
were matched by enumerate index {idx:06d} against a SHUFFLED test.txt, mis-
pairing nearly every caption/GT and inflating FID to ~34.

Usage:
  python3 scripts/eval/eval_ms272_trend.py --epochs 5 6 7 8 10 14 18 21 \
      --trend-dir outputs/tmp/20260623_t2m_epoch_trend --n-repeats 20
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, nargs="+", required=True)
    p.add_argument("--trend-dir", required=True)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator
    ev = MotionStreamer272Evaluator(device="cuda")

    trend = Path(args.trend_dir)
    rows = []
    # GT-only reference (real-vs-real) once, for context.
    gt = ev.evaluate_dir("", n_repeats=args.n_repeats, gt_only=True)
    print(f"[GT] R-Prec={gt['r_precision_real']} Div={gt['diversity_real']:.3f} "
          f"MM={gt['matching_score_real']:.3f}", flush=True)

    header = f"{'epoch':>6} {'n':>4} {'R1':>7} {'R2':>7} {'R3':>7} {'FID':>8} {'MM':>7} {'Div':>7}"
    print(header, flush=True)
    for E in args.epochs:
        pdir = trend / f"epoch_{E}" / "pred272"
        if not pdir.exists():
            print(f"{E:>6}  MISSING {pdir}", flush=True)
            continue
        res = ev.evaluate_dir(str(pdir), n_repeats=args.n_repeats)
        rp = res["r_precision_pred"]  # [R1, R2, R3]
        n = res.get("num_samples", res.get("n", "?"))
        skipped = res.get("skipped_no_pred", "?")
        row = {
            "epoch": E,
            "scored": (500 - skipped) if isinstance(skipped, int) else None,
            "skipped": skipped,
            "R1": float(rp[0]), "R2": float(rp[1]), "R3": float(rp[2]),
            "FID": float(res["fid"]),
            "MM": float(res["matching_score_pred"]),
            "Div": float(res["diversity_pred"]),
        }
        rows.append(row)
        nn = row["scored"] if row["scored"] is not None else "?"
        print(f"{E:>6} {nn:>4} {row['R1']:>7.4f} {row['R2']:>7.4f} {row['R3']:>7.4f} "
              f"{row['FID']:>8.4f} {row['MM']:>7.4f} {row['Div']:>7.4f}", flush=True)

    out_json = args.out_json or str(trend / "metrics_ms272" / "summary.json")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"gt": {k: (v if not hasattr(v, "tolist") else v.tolist())
                       for k, v in gt.items() if k in
                       ("r_precision_real", "diversity_real", "matching_score_real")},
               "rows": rows}, open(out_json, "w"), indent=2, default=str)
    print(f"[done] -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
