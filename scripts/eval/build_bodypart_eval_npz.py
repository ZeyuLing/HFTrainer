#!/usr/bin/env python3
"""Package a Table-6 ExpB baseline (CondMDI / OmniControl) into per-clip eval NPZ.

The baseline emits, after ``hml263_to_smpl_ik.py``, a dir of ``{source_id}.npz``
each carrying ``motion_135`` (SMPL prediction @30fps).  This builder pairs every
prediction with the SHARED ground-truth + caption used by \\ours E10 (so all
methods are scored on the identical clip set / GT) and emits the canonical eval
NPZ consumed by ``paper_npz_observed_pos_mpjpe.py`` + ``eval_editing_272_fid.py``:

    motion_135     (T,135) baseline pred, linearly resampled to GT length
    gt_motion_135  (T,135) shared GT (editing source_npz/{source_id}.npz)
    src_mask       (T,135) body-part ROTATION mask (1=generate, 0=observe);
                   reference only -- the position metric uses PART_JOINTS directly
    caption        str     (editing data_list caption_en)
    source_id      str

Out NPZ are ``{source_id}.npz`` (one per shared clip).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)
sys.path.insert(0, _REPO)

from bodypart_pos_common import (  # noqa: E402
    SOURCE_NPZ_DIR, build_part_level_mask, load_editing_index, shared_source_ids,
)

# map Table-6 part key -> build_part_level_mask keep_part arg
_KEEP = {
    "A_upper": "upper", "B_lower": "lower", "C_spine_only": "spine_only",
    "D_arms_only": "arms_only", "E_legs_only": "legs_only",
    "F_left_arm": "left_arm", "G_right_arm": "right_arm",
    "H_left_leg": "left_leg", "I_right_leg": "right_leg",
    "J_feet_only": "feet_only", "K_no_feet": "no_feet",
}


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


def _load_pred(ik_dir: str, sid: str) -> np.ndarray | None:
    for cand in (f"{sid}.npz", f"{sid}.npy"):
        p = os.path.join(ik_dir, cand)
        if os.path.exists(p):
            if p.endswith(".npz"):
                z = np.load(p, allow_pickle=True)
                key = "motion_135" if "motion_135" in z.files else z.files[0]
                return np.asarray(z[key], dtype=np.float32)
            return np.asarray(np.load(p), dtype=np.float32)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik-dir", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap on shared clip count (0=all available)")
    args = ap.parse_args()

    keep = _KEEP[args.part]
    dl = load_editing_index()
    cap_by_sid = {str(it["source_id"]): str(it.get("caption_en", "")) for it in dl}
    sids = shared_source_ids(args.max_samples or None)
    os.makedirs(args.out_dir, exist_ok=True)

    written = missing = 0
    for sid in sids:
        pred = _load_pred(args.ik_dir, sid)
        if pred is None:
            missing += 1
            continue
        gpath = os.path.join(SOURCE_NPZ_DIR, f"{sid}.npz")
        if not os.path.exists(gpath):
            missing += 1
            continue
        gt = np.asarray(np.load(gpath, allow_pickle=True)["motion_135"], dtype=np.float32)
        T = gt.shape[0]
        pred_r = _resample_linear(pred, T)
        if pred_r.shape[1] < 135:
            missing += 1
            continue
        pred_r = pred_r[:, :135]
        src_mask = build_part_level_mask(T, D=135, keep_part=keep).astype(np.float32)
        np.savez(
            os.path.join(args.out_dir, f"{sid}.npz"),
            motion_135=pred_r,
            gt_motion_135=gt,
            src_mask=src_mask,
            caption=cap_by_sid.get(sid, ""),
            source_id=sid,
        )
        written += 1
    print(f"[build_bodypart {args.part}] written={written} missing_pred={missing} "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
