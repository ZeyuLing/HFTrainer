#!/usr/bin/env python3
"""Package KIMODO E10 part-control retarget output into the self-contained
editing-eval NPZ schema (motion_135 / gt_motion_135 / src_mask / caption), so
the same paper metric scripts used for ``\\ours`` apply unchanged:

    scripts/eval/eval_editing_272_fid.py   -> FID  (FK-matched, 272 TMR)
    scripts/eval/paper_npz_ric_mpjpe.py    -> Foot / Jitter (272-ric space)
    hftrainer ... compute_rotation_ctrl_error -> Ctrl.Err (deg, observed joints)

GT and the part-level mask are derived EXACTLY as \\ours: GT comes from the same
``load_eval_samples(eval_e10_part_control.json)`` order that KIMODO consumed, and
``src_mask`` from ``build_part_level_mask(T, 135, keep_part=key)`` (0=keep/observe,
1=generate). KIMODO retarget output is source_id-keyed; we map each loaded sample
i -> source_id -> <smplx-dir>/<source_id>.npz, and write idx-keyed {i:05d}.npz so
the index ordering matches \\ours E10 dumps.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)

from hftrainer.evaluation.motion.m2m_eval_tasks import build_part_level_mask  # noqa: E402
from scripts.eval.eval_m2m_v2_all_tasks import load_eval_samples  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smplx-dir", required=True,
                    help="dir of <source_id>.npz (KIMODO rotation retarget motion_135)")
    ap.add_argument("--setting-key", required=True,
                    help="bare part key: upper/lower/spine_only/arms_only/legs_only/"
                         "left_arm/right_arm/left_leg/right_leg/feet_only/no_feet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-file",
                    default="data/eval/m2m_v2/eval_e10_part_control.json")
    ap.add_argument("--motion-data-dir",
                    default=os.path.join(REPO, "data", "hymotion_data"))
    ap.add_argument("--max-samples", type=int, default=5000)
    args = ap.parse_args()

    os.chdir(REPO)
    samples = load_eval_samples(
        args.data_file, args.motion_data_dir, args.max_samples,
        require_caption=False)
    os.makedirs(args.out_dir, exist_ok=True)

    n_ok = n_miss = 0
    for i, s in enumerate(samples):
        sid = os.path.splitext(os.path.basename(s["path"]))[0]
        raw = os.path.join(args.smplx_dir, f"{sid}.npz")
        if not os.path.exists(raw):
            n_miss += 1
            continue
        gt135 = np.asarray(s["motion"], dtype=np.float32)
        T = gt135.shape[0]
        pred = np.asarray(np.load(raw, allow_pickle=True)["motion_135"],
                          dtype=np.float32)
        if pred.shape[0] != T:
            pred = _resample_linear(pred, T)
        mask = np.asarray(
            build_part_level_mask(T=T, D=135, keep_part=args.setting_key),
            dtype=np.float32)
        if mask.shape[-1] != 135:
            mask = mask[:, :135]
        np.savez(
            os.path.join(args.out_dir, f"{i:05d}.npz"),
            motion_135=pred,
            gt_motion_135=gt135,
            src_mask=mask,
            caption=s.get("caption", ""),
            source_id=sid,
        )
        n_ok += 1
    print(f"[build-e10] key={args.setting_key} ok={n_ok} miss={n_miss} "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
