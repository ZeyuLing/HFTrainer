#!/usr/bin/env python3
"""Build m2m_eval_viewer NPZ cases for the TP2M (prefix-pose) task.

The viewer (`motion_annot_web/m2m_eval_viewer/app.py`) colors each frame/joint by
an ``src_mask`` (T,198; 0=condition/known -> green, 1=generate -> orange). For
prefix conditioning the first ``cond`` frames are the given prefix (condition)
and the rest are generated, so we set ``src_mask[:cond]=0`` and ``src_mask[cond:]=1``.

Each case NPZ holds: motion_135 (prediction), gt_motion_135 (GT, same canonical
id), src_mask (T,198), caption. Output layout matches what the viewer scans:

    <out-dir>/<model>/<task_setting>/npz/<id>.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
_TEXTS = os.path.join(
    _ROOT, "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/texts")


def _read_caption(cid: str) -> str:
    fp = os.path.join(_TEXTS, f"{cid}.txt")
    if not os.path.isfile(fp):
        return ""
    for ln in open(fp):
        ln = ln.strip()
        if not ln:
            continue
        # HumanML3D text lines: "caption#tokens#..." -> keep the caption part.
        return ln.split("#")[0].strip()
    return ""


def _load_m135(path: str):
    d = np.load(path, allow_pickle=True)
    if "motion_135" not in d:
        return None
    return np.asarray(d["motion_135"], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="repacked prep dir with motion_135")
    ap.add_argument("--gt-dir", required=True, help="GT prep dir (canonical id motion_135)")
    ap.add_argument("--cond", type=int, required=True, help="# prefix condition frames")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="PRISM_ours")
    ap.add_argument("--task-setting", default=None, help="default TP2M_cond{N}")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", default=None, help="comma list of canonical ids to force-include")
    args = ap.parse_args()

    task = args.task_setting or f"TP2M_cond{args.cond}"
    dst = os.path.join(args.out_dir, args.model, task, "npz")
    os.makedirs(dst, exist_ok=True)

    files = sorted(e.name for e in os.scandir(args.pred_dir) if e.name.endswith(".npz"))
    forced = set((args.ids or "").split(",")) - {""}
    forced_files = [f"{i}.npz" for i in forced if os.path.isfile(os.path.join(args.pred_dir, f"{i}.npz"))]
    rest = [f for f in files if f not in set(forced_files)]
    if args.limit and len(rest) > args.limit - len(forced_files):
        rng = np.random.RandomState(args.seed)
        k = max(0, args.limit - len(forced_files))
        rest = [rest[i] for i in sorted(rng.choice(len(rest), min(k, len(rest)), False))]
    chosen = forced_files + rest

    n_ok = 0
    for fn in chosen:
        cid = os.path.splitext(fn)[0]
        pred = _load_m135(os.path.join(args.pred_dir, fn))
        if pred is None or pred.shape[0] <= args.cond:
            continue
        gtp = os.path.join(args.gt_dir, fn)
        gt = _load_m135(gtp) if os.path.isfile(gtp) else None
        T = pred.shape[0]
        src_mask = np.ones((T, 198), np.float32)
        src_mask[: args.cond] = 0.0                 # prefix = condition (green)
        out = {
            "motion_135": pred,
            "src_mask": src_mask,
            "caption": _read_caption(cid),
            "task_key": task,
        }
        if gt is not None:
            out["gt_motion_135"] = gt
        np.savez(os.path.join(dst, fn), **out)
        n_ok += 1
    print(f"[viewer] {args.model}/{task}: wrote {n_ok} cases -> {dst}", flush=True)


if __name__ == "__main__":
    main()
