#!/usr/bin/env python3
"""Convert \\ours{} M2M edit outputs (135-dim) to MotionFix TMR 'pose' .npy files.

The MotionFix TMR Generated-to-Target evaluator (tmr_evaluator/motion2motion_retr.py
::collect_gen_samples) globs ``<dir>/*.npy``; each file is a pickled dict
``{'pose': [T, 135]}`` with layout ``[trans(3), global_orient_6d(6), body_pose_6d(126)]``
where the 6d uses MotionFix's convention (pytorch3d ``matrix_to_rotation_6d`` =
first two ROWS of the rotation matrix).

CRITICAL — 6d convention: \\ours{}'s ``motion_135`` rot6d uses the HyMotion-M2M
"row" layout (first two COLUMNS, per process_smplx_pose), which is NOT the same as
MotionFix's. We therefore go through rotation MATRICES (convention-free) and
re-encode with MotionFix's own ``transform_body_pose`` so the 6d matches exactly:
    ours 6d(row) --hftrainer.rotation_6d_to_matrix(row)--> R --MotionFix(rot->6d)--> pose
    GT aa(66)     --MotionFix(aa->6d)--------------------------------------------> pose

Generated NPZ are named ``<sample_idx:05d>.npz`` (datalist order) with a
``motion_135`` array; we map idx -> MotionFix keyid via the datalist so the output
files are ``<keyid>.npy`` and align with the TMR target lookup.

Usage:
    python3 scripts/eval/motionfix_135_to_tmr_pose.py \
        --gen-npz-dir <out>/smpl_caption_editfix_latest/E16_style_edit/npz \
        --datalist data/eval/m2m_v2/eval_motionfix_instruction.json \
        --out-dir <out>/tmr_pose
    # GT sanity (target self-retrieval upper bound; expect very high R@1):
    python3 scripts/eval/motionfix_135_to_tmr_pose.py \
        --datalist ... --gt-passthrough --out-dir <out>/tmr_pose_gt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

_REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
_MFIX = _REPO / "data/MotionFix/motionfix"
import sys
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_MFIX))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-npz-dir", default=None)
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-passthrough", action="store_true")
    ap.add_argument("--ours-convention", default="row", choices=["row", "column"],
                    help="rot6d layout of ours motion_135 (HyMotion-M2M = row).")
    args = ap.parse_args()

    # MotionFix-native 6d (pytorch3d, first two rows).
    from src.tools.transforms3d import transform_body_pose
    # hftrainer decoder for ours rot6d (-> rotation matrix).
    from hftrainer.motion.representation.rotation import rotation_6d_to_matrix

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = json.load(open(args.datalist))["data_list"]

    def gt_pose(motion_path):
        d = np.load(motion_path, allow_pickle=True)
        trans = torch.from_numpy(d["trans"][:, :3].astype(np.float32))
        aa = torch.from_numpy(d["poses"][:, :66].astype(np.float32))  # SMPL-22 aa
        sixd = transform_body_pose(aa, "aa->6d").reshape(aa.shape[0], -1)  # [T,132]
        return torch.cat([trans, sixd], dim=-1).numpy().astype(np.float32)

    def ours_pose(motion_135):
        m = torch.from_numpy(np.asarray(motion_135, dtype=np.float32))
        T = m.shape[0]
        trans = m[:, :3]
        rot6d = m[:, 3:135].reshape(T, 22, 6)
        R = rotation_6d_to_matrix(rot6d, convention=args.ours_convention)  # [T,22,3,3]
        sixd = transform_body_pose(R, "rot->6d").reshape(T, -1)  # MotionFix 6d [T,132]
        return torch.cat([trans, sixd], dim=-1).numpy().astype(np.float32)

    n_ok, n_miss = 0, 0
    for idx, it in enumerate(items):
        keyid = it.get("prompt_id") or it.get("annotation_id") or f"{idx:05d}"
        try:
            if args.gt_passthrough:
                pose = gt_pose(it["motion_path"])
            else:
                npz = Path(args.gen_npz_dir) / f"{idx:05d}.npz"
                if not npz.exists():
                    n_miss += 1
                    continue
                pose = ours_pose(np.load(npz, allow_pickle=True)["motion_135"])
        except Exception as e:  # noqa: BLE001
            if n_miss < 5:
                print(f"[miss] {keyid}: {type(e).__name__}: {e}")
            n_miss += 1
            continue
        np.save(out_dir / f"{keyid}.npy", {"pose": pose})
        n_ok += 1

    print(f"[done] wrote {n_ok} pose files ({n_miss} missing) -> {out_dir}")


if __name__ == "__main__":
    main()
