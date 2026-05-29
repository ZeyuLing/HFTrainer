#!/usr/bin/env python3
"""Aggregate PRISM overfit positions NPZ into overall metrics.

Beyond the stored full-body MPJPE (which includes accumulated global root
trajectory drift from abs_rel translation rollout), this also computes a
*root-aligned* MPJPE (per-frame pelvis subtracted) to isolate local pose error
from global trajectory drift -- key for judging whether the overfit memorised
the body articulation even when the global path drifts.

Usage: aggregate_prism_overfit_positions.py <positions_dir>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def root_aligned_mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    # pred/gt: (T, 22, 3). Subtract pelvis (joint 0) per frame, then MPJPE.
    pred_a = pred - pred[:, 0:1, :]
    gt_a = gt - gt[:, 0:1, :]
    return float(np.linalg.norm(pred_a - gt_a, axis=-1).mean() * 1000.0)


def full_mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0)


def main():
    pos_dir = sys.argv[1]
    files = sorted(f for f in os.listdir(pos_dir) if f.endswith(".npz"))
    rows = []
    for f in files:
        d = np.load(os.path.join(pos_dir, f), allow_pickle=True)
        pred = np.asarray(d["pred_positions"], dtype=np.float64)
        gt = np.asarray(d["gt_positions"], dtype=np.float64)
        m = d["metrics"].item() if "metrics" in d else {}
        rows.append({
            "key": os.path.splitext(f)[0],
            "T": int(d["num_frames"]) if "num_frames" in d else pred.shape[0],
            "mpjpe_full_mm": full_mpjpe_mm(pred, gt),
            "mpjpe_rootalign_mm": root_aligned_mpjpe_mm(pred, gt),
            "mpjre_deg": float(m.get("mpjre_deg", float("nan"))),
            "stored_mpjpe_mm": float(m.get("mpjpe_mm", float("nan"))),
            "transl_l2": float(m.get("transl_l2", float("nan"))),
        })

    if not rows:
        print("No NPZ found in", pos_dir)
        return

    def col(name):
        return np.array([r[name] for r in rows], dtype=np.float64)

    full = col("mpjpe_full_mm")
    rootal = col("mpjpe_rootalign_mm")
    mpjre = col("mpjre_deg")

    def stats(x):
        x = x[np.isfinite(x)]
        return dict(mean=float(x.mean()), median=float(np.median(x)),
                    p90=float(np.percentile(x, 90)), max=float(x.max()),
                    min=float(x.min()))

    stored = col("stored_mpjpe_mm") if any("stored_mpjpe_mm" in r for r in rows) else full
    summary = {
        "num_samples": len(rows),
        "mpjpe_full_mm": stats(full),
        "mpjpe_stored_mm": stats(stored),
        "mpjpe_rootalign_mm": stats(rootal),
        "mpjre_deg": stats(mpjre),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    order = np.argsort(-full)
    print("\n=== Worst 8 by full MPJPE ===")
    print(f"{'key':<28} {'T':>4} {'full':>9} {'rootAl':>9} {'mpjre':>7}")
    for i in order[:8]:
        r = rows[i]
        print(f"{r['key']:<28} {r['T']:>4} {r['mpjpe_full_mm']:>9.1f} "
              f"{r['mpjpe_rootalign_mm']:>9.1f} {r['mpjre_deg']:>7.2f}")
    print("\n=== Best 5 by full MPJPE ===")
    for i in order[::-1][:5]:
        r = rows[i]
        print(f"{r['key']:<28} {r['T']:>4} {r['mpjpe_full_mm']:>9.1f} "
              f"{r['mpjpe_rootalign_mm']:>9.1f} {r['mpjre_deg']:>7.2f}")

    out = os.path.join(os.path.dirname(pos_dir.rstrip("/")),
                       os.path.basename(pos_dir.rstrip("/")) + "_aggregate.json")
    with open(out, "w") as fp:
        json.dump({"summary": summary, "per_sample": rows}, fp, indent=2)
    print("\nSaved aggregate ->", out)


if __name__ == "__main__":
    main()
