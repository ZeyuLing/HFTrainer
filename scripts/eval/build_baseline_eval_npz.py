#!/usr/bin/env python3
"""Package a baseline's per-clip SMPL ``motion_135`` (e.g. from CondMDI joints ->
hml263_to_smpl_ik) into the *self-contained* editing-eval NPZ schema used by the
paper metric scripts, so the baseline becomes directly comparable to ``\\ours``:

    <out-dir>/<id>.npz  with
        motion_135     (T,135)  baseline prediction, resampled to GT length
        gt_motion_135  (T,135)  native GT (from eval_h3d_editing source_npz)
        src_mask       (T,135)  protocol mask (1=generate, 0=observe), IDENTICAL
                                to hftrainer build_inbetween_mask used by \\ours
        caption        str      caption_en (for R-precision)
        source_id      str

Then feed <out-dir> to:
    scripts/eval/paper_npz_ric_mpjpe.py   -> MPJPE_gen / [P]-MPJPE  (272-ric)
    scripts/eval/eval_editing_272_fid.py  -> FID / R@k / Div        (FK-matched)

Pred is resampled (linear) to the GT frame count so the protocol mask's observed
window (esp. post20's tail) stays frame-aligned with GT.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)

from hftrainer.evaluation.motion.m2m_eval_tasks import build_inbetween_mask  # noqa: E402

PROTO_KW = {
    "pre20":  dict(keep_start_frac=0.20, keep_end=0),
    "post20": dict(keep_start=0, keep_end_frac=0.20),
    "mid60":  dict(keep_start_frac=0.20, keep_end_frac=0.20),
    "both_1f": dict(keep_start=1, keep_end=1),
}


def _resample_linear(m: np.ndarray, target_T: int) -> np.ndarray:
    """Linear time-resample (T,D) -> (target_T,D)."""
    T = m.shape[0]
    if T == target_T:
        return m.astype(np.float32)
    if T == 1:
        return np.repeat(m, target_T, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, target_T)
    out = np.empty((target_T, m.shape[1]), dtype=np.float32)
    for d in range(m.shape[1]):
        out[:, d] = np.interp(dst, src, m[:, d])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik-dir", required=True,
                    help="dir of <id>.npz each with motion_135 (baseline IK output)")
    ap.add_argument("--eval-json",
                    default="data/eval/m2m_v2/eval_h3d_editing.json")
    ap.add_argument("--protocol", required=True, choices=list(PROTO_KW))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-condition", dest="keep_condition",
                    action="store_true", default=False,
                    help="hard-keep GT on observed/condition frames. ONLY for the "
                         "VIEWER (so condition shows clean GT, like \\ours). MUST be "
                         "OFF for metrics: the 272 rep integrates root angular "
                         "velocity (heading) cumulatively, so splicing a GT "
                         "condition trajectory onto a model's generated region "
                         "rotates every generated frame's heading-removed joints and "
                         "corrupts mpjpe_gen/FID. Metrics must use the model's own "
                         "coherent sequence.")
    args = ap.parse_args()

    os.chdir(REPO)
    meta = json.load(open(args.eval_json))["data_list"]
    by_id = {}
    for it in meta:
        sid = it.get("source_id") or os.path.splitext(
            os.path.basename(it["motion_path"]))[0]
        by_id[str(sid)] = it

    os.makedirs(args.out_dir, exist_ok=True)
    ik_files = sorted(glob.glob(os.path.join(args.ik_dir, "*.npz")))
    n_ok = n_skip = 0
    for f in ik_files:
        sid = os.path.splitext(os.path.basename(f))[0]
        it = by_id.get(sid)
        if it is None:
            n_skip += 1
            continue
        gtz = np.load(it["motion_path"], allow_pickle=True)
        gt135 = np.asarray(gtz["motion_135"], dtype=np.float32)
        T = gt135.shape[0]
        pred = np.asarray(np.load(f, allow_pickle=True)["motion_135"],
                          dtype=np.float32)
        pred = _resample_linear(pred, T)
        mask = build_inbetween_mask(T, D=135, **PROTO_KW[args.protocol]).astype(np.float32)
        if mask.shape[-1] != 135:
            mask = mask[:, :135]
        if args.keep_condition:
            # Observed/condition frames are GIVEN to the model (== GT). \\ours
            # stores them verbatim from GT; splice GT here so every baseline is
            # scored/visualised on the SAME premise (only generated frames differ).
            obs = mask.max(axis=-1) < 0.5          # True = observed/condition frame
            pred[obs] = gt135[obs]
        np.savez(
            os.path.join(args.out_dir, f"{sid}.npz"),
            motion_135=pred,
            gt_motion_135=gt135,
            src_mask=mask,
            caption=it.get("caption_en", ""),
            source_id=sid,
        )
        n_ok += 1
    print(f"[build] protocol={args.protocol} ok={n_ok} skip={n_skip} -> {args.out_dir}")


if __name__ == "__main__":
    main()
