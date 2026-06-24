#!/usr/bin/env python3
"""Step-1 check: does our hftrainer MDM 263 output match official MDM paper metrics?

The HumanML3D-263 ``[vald]`` R-Precision, Matching-Score and Diversity only need
the *generated* motions + their captions (no real GT distribution), so we can
score our ``mdm_263`` predictions directly against MDM's released reference:
    R-P T1/T2/T3 = 0.3195 / 0.4978 / 0.6110 ; MM-Dist = 5.5659 ; Div = 9.5595
(FID needs real 263 GT and is reported separately once GT is available.)

Each ``mdm_263/<id>.npy`` was generated from the *first* caption of test id
``<id>``, so we use ``caption_selection="first"``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hftrainer.evaluation.evaluators.humanml3d_263 import (
    HumanML263Evaluator,
    read_h3d_texts,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272"

MIN_LEN = 40
UNIT = 4


def build_pred_only_samples(pred_dir: Path, texts_dir: Path, ids, workers=32):
    """Build samples with motion_pred = mdm_263 (motion_gt set to the same array
    so the shared evaluate() loop runs; only the [vald] outputs are consumed)."""
    from concurrent.futures import ThreadPoolExecutor

    def _fetch(sid):
        pf = pred_dir / f"{sid}.npy"
        if not pf.exists():
            return None
        mp = np.load(pf)
        tl = [t for t in read_h3d_texts(texts_dir / f"{sid}.txt")
              if t["f_tag"] == 0.0 and t["to_tag"] == 0.0]
        return sid, mp, tl

    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_fetch, ids):
            if res is None:
                continue
            sid, mp, tl = res
            if mp.ndim != 2 or mp.shape[1] != 263 or len(mp) < MIN_LEN or not tl:
                continue
            t_eff = (len(mp) // UNIT) * UNIT
            if t_eff < MIN_LEN:
                continue
            out.append({"name": sid, "motion_pred": mp[:t_eff],
                        "motion_gt": mp[:t_eff], "text_list": tl, "length": t_eff})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--data_root", default=str(DEFAULT_DATA))
    p.add_argument("--ckpt_dir", default=None, help="263 evaluator dir (text_mot_match.tar + glove/ + meta/). Use /dev/shm cache to avoid CephFS.")
    p.add_argument("--n_repeats", type=int, default=20)
    p.add_argument("--caption_selection", default="first", choices=["first", "random"])
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    data_root = Path(args.data_root)
    ids = [x.strip() for x in (data_root / "split" / "test.txt").read_text().splitlines() if x.strip()]
    ev = HumanML263Evaluator(ckpt_dir=args.ckpt_dir)
    ev._ensure_loaded()
    samples = build_pred_only_samples(Path(args.pred_dir), data_root / "texts", ids)
    print(f"[+] samples = {len(samples)}", flush=True)

    res = ev.evaluate(samples, mode="pred", n_repeats=args.n_repeats,
                      caption_selection=args.caption_selection)

    rp = res["r_precision"]["mean"]
    out = {
        "n_samples": res["n_samples"],
        "n_repeats": res["n_repeats"],
        "caption_selection": args.caption_selection,
        "r_precision_vald": rp,
        "r_precision_vald_std": res["r_precision"]["std"],
        "matching_score_vald": res["matching_score"]["mean"],
        "diversity_vald": res["diversity"]["mean"],
        "note_fid_needs_gt": True,
        "official_mdm_ref": {
            "r_precision": [0.3195, 0.4978, 0.6110],
            "matching_score": 5.5659, "diversity": 9.5595, "fid": 0.5443,
        },
    }
    print(json.dumps(out, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"[+] wrote {args.out_json}")


if __name__ == "__main__":
    main()
