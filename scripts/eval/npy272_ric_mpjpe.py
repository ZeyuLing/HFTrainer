#!/usr/bin/env python3
"""272-ric-space MPJPE/[P]-MPJPE between two dirs of per-clip ``{id}.npy`` (T,272)
arrays (e.g. the output of ``convert_motion135_to_h3d272.py``), matched by id.

MPJPE is the HumanML3D-272 root-relative joint space ([8:8+66], 22 joints),
identical to ``chain272_ric_mpjpe.py`` but reading raw ``.npy`` 272 arrays
instead of ``.npz`` with a ``motion_272`` key.

For the MIB protocol (--preserve mib) the first and last frame are the preserved
endpoints; [P]-MPJPE is restricted to them and MPJPE_gen to the rest.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

_NJ = 22


def _ric(m: np.ndarray) -> np.ndarray:
    T = len(m)
    return m[:, 8:8 + 3 * _NJ].reshape(T, _NJ, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--preserve", choices=["mib", "none"], default="mib")
    ap.add_argument("--tag", default="pred")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    fu, pp, miss = [], [], 0
    for f in sorted(glob.glob(os.path.join(args.pred_dir, "*.npy"))):
        cid = os.path.basename(f)[:-4]
        g = os.path.join(args.gt_dir, cid + ".npy")
        if not os.path.exists(g):
            miss += 1
            continue
        a = np.load(f).astype("float32")
        b = np.load(g).astype("float32")
        T = min(len(a), len(b))
        if T < 2:
            continue
        e = np.linalg.norm(_ric(a[:T]) - _ric(b[:T]), axis=-1)  # (T,22) metres
        pres = np.zeros(T, bool)
        if args.preserve == "mib":
            pres[0] = True
            pres[T - 1] = True
        gen = ~pres
        fu.append(float(e[gen].mean()) if gen.any() else float(e.mean()))
        pp.append(float(e[pres].mean()) if pres.any() else 0.0)

    out = {
        "tag": args.tag, "n": len(fu), "n_miss": miss,
        "mpjpe_gen_cm": float(np.mean(fu) * 100) if fu else None,
        "p_mpjpe_cm": float(np.mean(pp) * 100) if pp else None,
    }
    print(f"[{args.tag}] n={out['n']} miss={miss} "
          f"MPJPE_gen={out['mpjpe_gen_cm']}cm [P]={out['p_mpjpe_cm']}cm")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        json.dump(out, open(args.out_json, "w"), indent=2)
        print(f"-> {args.out_json}")


if __name__ == "__main__":
    main()
