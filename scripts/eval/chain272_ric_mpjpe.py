#!/usr/bin/env python3
"""272-ric-space MPJPE / [P]-MPJPE between a pred 272-chain dir and a GT 272-chain
dir (matched by clip id), for baselines whose output is stored as ``motion_272``
(``<id>.npz`` with key ``motion_272``) — e.g. ``_condmdi_chain_272`` vs
``_gtchain_272`` under ``output/evaluation/mib_h3d_full/``.

MPJPE is the HumanML3D-272 root-relative joint space ([8:74], heading-removed,
xz-centered), identical to ``aggregate_mib_umo.py`` / ``paper_npz_ric_mpjpe.py``.
``[P]-MPJPE`` is restricted to the preserved frames; for the MIB (first+last
frame) protocol that is frame 0 and the last valid frame (``--preserve mib``).
Also reports a 272-ric jitter proxy.

Usage:
    python3 scripts/eval/chain272_ric_mpjpe.py \
        --pred-dir output/evaluation/mib_h3d_full/_condmdi_chain_272 \
        --gt-dir   output/evaluation/mib_h3d_full/_gtchain_272 \
        --preserve mib --tag condmdi_mib --out-json docs/temp/condmdi_mib_ric.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

_NJ = 22


def _ric(m272: np.ndarray) -> np.ndarray:
    T = m272.shape[0]
    return m272[:, 8:8 + 3 * _NJ].reshape(T, _NJ, 3)


def _jitter(pos: np.ndarray, fps: int = 30) -> float:
    if pos.shape[0] < 4:
        return 0.0
    acc = np.diff(pos, n=3, axis=0) * (fps ** 3)
    return float(np.linalg.norm(acc, axis=-1).mean())


def _load272(fp: str) -> np.ndarray | None:
    d = np.load(fp, allow_pickle=True)
    if "motion_272" not in d:
        return None
    return np.asarray(d["motion_272"], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--preserve", choices=["mib", "none"], default="mib",
                    help="mib = first+last frame preserved (MIB protocol)")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--tag", default="pred")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.npz")))
    if args.max_samples:
        pred_files = pred_files[: args.max_samples]

    acc = {"mpjpe_full": [], "p_mpjpe": [], "mpjpe_gen": [], "jitter": []}
    n_miss = 0
    for pf in pred_files:
        cid = os.path.basename(pf)[:-4]
        gf = os.path.join(args.gt_dir, cid + ".npz")
        if not os.path.exists(gf):
            n_miss += 1
            continue
        mp = _load272(pf)
        mg = _load272(gf)
        if mp is None or mg is None:
            n_miss += 1
            continue
        T = min(len(mp), len(mg))
        if T < 2:
            continue
        Pp, Gg = _ric(mp[:T]), _ric(mg[:T])
        err = np.linalg.norm(Pp - Gg, axis=-1)  # (T,22) m
        pres = np.zeros(T, dtype=bool)
        if args.preserve == "mib":
            pres[0] = True
            pres[T - 1] = True
        gen = ~pres
        acc["mpjpe_full"].append(float(err.mean()))
        acc["mpjpe_gen"].append(float(err[gen].mean()) if gen.any() else float(err.mean()))
        acc["p_mpjpe"].append(float(err[pres].mean()) if pres.any() else 0.0)
        acc["jitter"].append(_jitter(Pp, 30))

    if not acc["mpjpe_full"]:
        print(f"[{args.tag}] no matched pairs (pred={len(pred_files)}, miss={n_miss})")
        return
    out = {"tag": args.tag, "pred_dir": args.pred_dir, "gt_dir": args.gt_dir,
           "preserve": args.preserve, "n": len(acc["mpjpe_full"]), "n_miss": n_miss}
    for k, v in acc.items():
        out[k + "_mean"] = float(np.mean(v))
        out[k + "_std"] = float(np.std(v))
    print(f"\n=== {args.tag} (272-ric, preserve={args.preserve}) ===")
    print(f" n={out['n']} miss={n_miss}  MPJPE_full={out['mpjpe_full_mean']*100:.2f}cm  "
          f"MPJPE_gen={out['mpjpe_gen_mean']*100:.2f}cm  "
          f"[P]-MPJPE={out['p_mpjpe_mean']*100:.3f}cm  jitter={out['jitter_mean']:.1f}")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"-> {args.out_json}")


if __name__ == "__main__":
    main()
