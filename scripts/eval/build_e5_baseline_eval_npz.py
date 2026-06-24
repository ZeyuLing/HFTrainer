#!/usr/bin/env python3
"""Package a baseline's E5 (trajectory-following) prediction into the SAME
self-contained eval NPZ schema \\ours produced for Table 7, by pairing strictly
*by index* with the \\ours E5 NPZ dump.

Why index pairing (not source_id)?  Both the \\ours E5 dump and every baseline
generation (KIMODO ``run_kimodo_all_tasks.py``, etc.) enumerate the SAME shuffled
eval order and write ``{i:05d}.npz``.  We verified ours[i] and kimodo[i] share the
identical caption + frame count for every checked i, while ``load_eval_samples``
returns a *different* (raw) order -- so pairing by index is the only safe key.

For each index i we re-use \\ours's own ``gt_motion_135`` + trajectory ``src_mask``
+ ``caption`` (guaranteeing IDENTICAL GT, observed frames, and clip set as \\ours)
and only swap in the baseline prediction (linearly resampled to \\ours GT length).

Two input modes:
  --kimodo-raw-dir  : idx-keyed raw KIMODO npz (global_rot_mats SOMA-77 + translation);
                      FAITHFUL SOMA-rotation transfer -> motion_135 (no IK), matching
                      Table 5 ``run_keyframe_kimodo.sh`` ROT_MODE=1.
  --pred-dir        : idx-keyed npz each carrying ``motion_135`` (T,135) (e.g. a
                      baseline already retargeted to SMPL via hml263_to_smpl_ik).

Out NPZ schema (consumed by collect_ours_posthoc_metrics + paper_npz_ric_mpjpe +
eval_editing_272_fid):
    motion_135     (T,135) baseline pred, resampled to \\ours GT length
    gt_motion_135  (T,135) \\ours GT (identical to \\ours)
    src_mask       (T,198) \\ours trajectory mask (1=generate, 0=observe)
    caption        str
    source_id      str     (idx string; ours E5 dump carries no source_id)
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


def _resample_linear(m: np.ndarray, target_T: int) -> np.ndarray:
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


def _make_kimodo_rotation_retarget():
    """Return fn(raw_path)->motion_135 via faithful SOMA-30 rotation transfer."""
    sys.path.insert(0, os.path.join(REPO, "motion_annot_web", "score_m2m"))
    sys.path.insert(0, os.path.join(REPO, "motion_annot_web", "score_m2m", "soma_model"))
    from utils_soma import SOMASKEL30_IN_NVSKEL77  # noqa: E402
    from hftrainer.motion.retarget import SMPLSOMARetargeter  # noqa: E402

    S30 = np.asarray(SOMASKEL30_IN_NVSKEL77, dtype=np.int64)
    rot_rt = SMPLSOMARetargeter()

    def retarget(raw_path):
        z = np.load(raw_path, allow_pickle=True)
        gr77 = np.asarray(z["global_rot_mats"], dtype=np.float32)  # (T,77,3,3)
        if "translation" in z.files:
            transl = np.asarray(z["translation"], dtype=np.float32)
        else:
            pj = np.asarray(z["posed_joints"], dtype=np.float32)
            transl = pj[:, 0, :] if pj.ndim == 3 else pj[:, :3]
        gr30 = gr77[:, S30]  # (T,30,3,3)
        T = gr30.shape[0]
        src = np.zeros((T, 135), dtype=np.float32)
        src[:, :3] = transl
        res = rot_rt.soma_to_smpl_from_rotations(
            gr30, src, height_mode="source_root")
        m = res["motion_135"] if isinstance(res, dict) else res
        return np.asarray(m, dtype=np.float32)

    return retarget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-npz-dir", required=True,
                    help="\\ours E5 npz dir (idx-named, gt_motion_135+src_mask+caption)")
    ap.add_argument("--kimodo-raw-dir", default=None,
                    help="idx-named raw KIMODO npz (global_rot_mats); rotation transfer")
    ap.add_argument("--pred-dir", default=None,
                    help="idx-named npz with motion_135 (already SMPL-retargeted)")
    ap.add_argument("--pred-sid-dir", default=None,
                    help="source_id-named npz/npy with motion_135 (OmniControl/GMD/"
                         "MotionLab via IK). Paired to \\ours idx via --idx2sid.")
    ap.add_argument("--idx2sid",
                    default="output/evaluation/table7_traj/e5_idx2sid.json",
                    help="json {idx: source_id} for ours E5 order (eval_h3d_editing)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=100000)
    args = ap.parse_args()

    n_modes = sum(bool(x) for x in
                  (args.kimodo_raw_dir, args.pred_dir, args.pred_sid_dir))
    assert n_modes == 1, \
        "exactly one of --kimodo-raw-dir / --pred-dir / --pred-sid-dir is required"

    os.chdir(REPO)
    retarget = _make_kimodo_rotation_retarget() if args.kimodo_raw_dir else None
    pred_root = args.kimodo_raw_dir or args.pred_dir or args.pred_sid_dir
    idx2sid = None
    if args.pred_sid_dir:
        idx2sid = {int(k): v for k, v in json.load(open(args.idx2sid)).items()}

    ours = sorted(glob.glob(os.path.join(args.ours_npz_dir, "*.npz")))
    os.makedirs(args.out_dir, exist_ok=True)

    n_ok = n_miss = n_bad = 0
    for i, of in enumerate(ours):
        if n_ok >= args.max_samples:
            break
        idx = os.path.splitext(os.path.basename(of))[0]
        if idx2sid is not None:
            sid = idx2sid.get(int(idx))
            pf = None
            for ext in (".npz", ".npy"):
                cand = os.path.join(pred_root, f"{sid}{ext}") if sid else None
                if cand and os.path.exists(cand):
                    pf = cand
                    break
            if pf is None:
                n_miss += 1
                continue
        else:
            pf = os.path.join(pred_root, f"{idx}.npz")
            if not os.path.exists(pf):
                n_miss += 1
                continue
        z = np.load(of, allow_pickle=True)
        gt = np.asarray(z["gt_motion_135"], dtype=np.float32)
        T = gt.shape[0]
        try:
            if retarget is not None:
                pred = retarget(pf)
            elif pf.endswith(".npy"):
                pred = np.asarray(np.load(pf, allow_pickle=True), dtype=np.float32)
            else:
                pz = np.load(pf, allow_pickle=True)
                key = "motion_135" if "motion_135" in pz.files else pz.files[0]
                pred = np.asarray(pz[key], dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            if n_bad < 3:
                print(f"[build-e5] bad {pf}: {type(e).__name__}: {e}")
            n_bad += 1
            continue
        if pred.ndim != 2 or pred.shape[1] < 135:
            n_bad += 1
            continue
        pred = pred[:, :135]
        pred = _resample_linear(pred, T)
        np.savez(
            os.path.join(args.out_dir, f"{idx}.npz"),
            motion_135=pred,
            gt_motion_135=gt,
            src_mask=np.asarray(z["src_mask"], dtype=np.float32),
            caption=str(z["caption"]) if "caption" in z.files else "",
            source_id=idx,
        )
        n_ok += 1
    print(f"[build-e5] ok={n_ok} miss_pred={n_miss} bad={n_bad} "
          f"ours_npz={len(ours)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
