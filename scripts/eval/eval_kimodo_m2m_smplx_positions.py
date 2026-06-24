#!/usr/bin/env python3
"""Position metrics (UMO 272-ric protocol) for KIMODO-SMPLX M2M per-sid NPZ.

Generic counterpart of ``aggregate_mib_umo.py`` (whose ``main`` is hard-wired to
the legacy ``MODELS``/``E2_both_1f`` directory layout). Here we simply walk a flat
directory of ``<sid>.npz`` produced by ``gen_kimodo_m2m_smplx.py`` (keys
``motion_135`` pred, ``gt_motion_135``, ``src_mask``) and report mean MPJPE /
[P]-MPJPE / MPJPE(gen) / jitter / foot-skating in the heading-removed
HumanML3D-272 root-relative joint space (cm), matching UMO.

Usage:
  python3 scripts/eval/eval_kimodo_m2m_smplx_positions.py \
      --pred-dir outputs/.../inbetween/preds_npz \
      --out      outputs/.../inbetween/positions.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _REPO_ROOT)

# Reuse the exact per-file metric used by the paper's inbetween position numbers.
from aggregate_mib_umo import _eval_npz  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.pred_dir, "*.npz")))
    files = [f for f in files if not os.path.basename(f).startswith("_summary")]
    if args.max_samples:
        files = files[: args.max_samples]
    if not files:
        raise SystemExit(f"no npz under {args.pred_dir}")

    accum = {"mpjpe_full": [], "mpjpe_gen": [], "p_mpjpe": [],
             "jitter": [], "foot_skating": []}
    warned = False
    for i, f in enumerate(files):
        try:
            r = _eval_npz(f)
        except Exception as e:  # noqa: BLE001
            if not warned:
                print(f"[warn] _eval_npz failed on {f}: {type(e).__name__}: {e}",
                      flush=True)
                warned = True
            r = None
        if r:
            for k in accum:
                accum[k].append(r[k])
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} kept={len(accum['mpjpe_full'])} "
                  f"MPJPE={np.mean(accum['mpjpe_full'])*100:.2f}cm "
                  f"[P]={np.mean(accum['p_mpjpe'])*100:.3f}cm", flush=True)

    n = len(accum["mpjpe_full"])
    if n == 0:
        raise SystemExit("no valid npz evaluated")
    res = {
        "pred_dir": args.pred_dir,
        "n_samples": n,
        "mpjpe_full_cm": float(np.mean(accum["mpjpe_full"]) * 100),
        "mpjpe_gen_cm": float(np.mean(accum["mpjpe_gen"]) * 100),
        "p_mpjpe_cm": float(np.mean(accum["p_mpjpe"]) * 100),
        "jitter": float(np.mean(accum["jitter"])),
        "foot_skating": float(np.mean(accum["foot_skating"])),
    }
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n=== POSITION METRICS (UMO 272-ric, cm) ===")
    print(f" N={n}")
    print(f" MPJPE      = {res['mpjpe_full_cm']:.2f} cm")
    print(f" MPJPE(gen) = {res['mpjpe_gen_cm']:.2f} cm")
    print(f" [P]-MPJPE  = {res['p_mpjpe_cm']:.3f} cm")
    print(f" Jitter     = {res['jitter']:.1f}")
    print(f" Foot       = {res['foot_skating']:.3f}")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
