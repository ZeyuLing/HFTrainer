#!/usr/bin/env python3
"""Build a CondMDI-compatible ``dataset/HumanML3D`` directory from our 272-derived
recon HumanML3D-263 test set, so CondMDI can run the in-betweening protocol on the
EXACT same 4012-clip eval set (and same first/last endpoints) as our model.

Layout produced under ``ref_repo/CondMDI/dataset/HumanML3D``::

    new_joint_vecs/<id>.npy          relative-root 263 (= our recon, unchanged)
    new_joint_vecs_abs_3d/<id>.npy   absolute-root 263 (CondMDI input rep)
    texts/<id>.txt                   HumanML3D-style captions
    test.txt                         all source_ids
    train.txt / val.txt              copy of test (loader needs them to exist)
    Mean.npy / Std.npy               standard t2m stats (copied)
    Mean_abs_3d.npy / Std_abs_3d.npy abs stats (copied from HumanML3D_abs)

rel->abs follows CondMDI ``HumanML3D.motion_to_abs_data``: keep dims [3:], replace
dim0 with cumulative root rot angle and dims[1,2] with absolute root x,z.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
CONDMDI = REPO / "ref_repo/CondMDI"
sys.path.insert(0, str(CONDMDI))
from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos  # noqa: E402


def rel263_to_abs263(rel: np.ndarray) -> np.ndarray:
    """(T,263) unnormalized relative -> (T,263) unnormalized absolute root."""
    t = torch.from_numpy(np.asarray(rel, dtype=np.float32))[None]  # [1,T,263]
    r_rot_quat, r_pos, rot_ang = recover_root_rot_pos(
        t[None], abs_3d=False, return_rot_ang=True)  # mirror motion_to_abs_data
    abs_ = t[None].clone()
    abs_[..., 0] = rot_ang
    abs_[..., [1, 2]] = r_pos[..., [0, 2]]
    return abs_[0, 0].numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", default=str(REPO / "data/eval/m2m_v2/eval_h3d_editing.json"))
    ap.add_argument("--recon", default=str(REPO / "work_dirs/h3d263_eval/h3d263_test_recon_fk"))
    ap.add_argument("--texts", default=str(REPO / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/texts"))
    ap.add_argument("--out", default=str(CONDMDI / "dataset/HumanML3D"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = json.load(open(args.eval_json))["data_list"]
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out)
    (out / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    (out / "new_joint_vecs_abs_3d").mkdir(parents=True, exist_ok=True)
    (out / "texts").mkdir(parents=True, exist_ok=True)

    recon = Path(args.recon)
    texts = Path(args.texts)

    ok, bad, kept_ids = 0, 0, []
    for it in items:
        sid = it["source_id"]
        rp = recon / "new_joint_vecs" / f"{sid}.npy"
        tp = texts / f"{sid}.txt"
        if not rp.exists() or not tp.exists():
            bad += 1
            continue
        rel = np.load(str(rp)).astype(np.float32)
        if rel.ndim != 2 or rel.shape[1] != 263 or rel.shape[0] < 24:
            bad += 1
            continue
        try:
            abs_ = rel263_to_abs263(rel)
        except Exception as e:  # noqa: BLE001
            print(f"  [abs-fail] {sid}: {type(e).__name__}: {e}")
            bad += 1
            continue
        if not np.isfinite(abs_).all():
            bad += 1
            continue
        np.save(str(out / "new_joint_vecs" / f"{sid}.npy"), rel)
        np.save(str(out / "new_joint_vecs_abs_3d" / f"{sid}.npy"), abs_)
        shutil.copyfile(str(tp), str(out / "texts" / f"{sid}.txt"))
        kept_ids.append(sid)
        ok += 1
        if ok % 500 == 0:
            print(f"  built {ok} ...", flush=True)

    (out / "test.txt").write_text("\n".join(kept_ids) + "\n")
    (out / "train.txt").write_text("\n".join(kept_ids) + "\n")
    (out / "val.txt").write_text("\n".join(kept_ids[:64]) + "\n")

    # stats
    src_std_mean = CONDMDI / "dataset/t2m_mean.npy"
    src_std_std = CONDMDI / "dataset/t2m_std.npy"
    src_abs_mean = CONDMDI / "dataset/HumanML3D_abs/Mean_abs_3d.npy"
    src_abs_std = CONDMDI / "dataset/HumanML3D_abs/Std_abs_3d.npy"
    shutil.copyfile(src_std_mean, out / "Mean.npy")
    shutil.copyfile(src_std_std, out / "Std.npy")
    shutil.copyfile(src_abs_mean, out / "Mean_abs_3d.npy")
    shutil.copyfile(src_abs_std, out / "Std_abs_3d.npy")

    print(f"[+] DONE ok={ok} bad={bad} -> {out}")
    print(f"    test.txt = {len(kept_ids)} ids")


if __name__ == "__main__":
    main()
