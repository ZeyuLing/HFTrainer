#!/usr/bin/env python3
"""Unified edit-distance measurement for E16 style_edit npz outputs.

Two modes:
  (A) --npz-dir DIR : read generated npz (motion_135=ours, source_motion_135,
      gt_motion_135) and report d(ours,src), d(ours,gt), d(src,gt), %closer-to-gt,
      improvement = (d(src,gt)-d(ours,gt))/d(src,gt).
  (B) --datalist JSON : NO model. Load source/target via load_motion_135d
      (canonical, same as eval input pipeline) and report d(src,tgt) + length stats.
      Used to prove training pairs are non-degenerate (target != source).

All distances: per-pair RMSE over min(T) frames x 135 dims, then mean over pairs.
"""
import argparse
import glob
import json
import os
import sys
import numpy as np

ROOT = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, ROOT)


def rmse(a, b):
    T = min(a.shape[0], b.shape[0])
    if T == 0:
        return None
    return float(np.sqrt(np.mean((a[:T] - b[:T]) ** 2)))


def mode_npz(npz_dir):
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    d_os, d_og, d_sg, closer = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if not {"motion_135", "source_motion_135", "gt_motion_135"} <= set(d.files):
            continue
        o = d["motion_135"].astype(np.float32)
        s = d["source_motion_135"].astype(np.float32)
        g = d["gt_motion_135"].astype(np.float32)
        dos, dog, dsg = rmse(o, s), rmse(o, g), rmse(s, g)
        if None in (dos, dog, dsg):
            continue
        d_os.append(dos); d_og.append(dog); d_sg.append(dsg)
        closer.append(1.0 if dog < dos else 0.0)
    n = len(d_os)
    if n == 0:
        print(f"[npz] {npz_dir}: NO valid pairs")
        return
    d_os, d_og, d_sg = map(np.array, (d_os, d_og, d_sg))
    impr = (d_sg.mean() - d_og.mean()) / d_sg.mean() * 100
    print(f"[npz] {npz_dir}")
    print(f"  n={n}")
    print(f"  d(ours,src) = {d_os.mean():.4f}")
    print(f"  d(ours,gt)  = {d_og.mean():.4f}")
    print(f"  d(src,gt)   = {d_sg.mean():.4f}")
    print(f"  improvement = {impr:+.1f}%   (d_src_gt -> d_ours_gt)")
    print(f"  closer-to-gt = {np.mean(closer)*100:.1f}%")


def mode_datalist(datalist):
    from scripts.eval.eval_m2m_v2_all_tasks import load_motion_135d
    import torch
    bone = torch.load(os.path.join(ROOT, "data/hymotion_m2m_data/bone_offsets_22.pt"),
                       map_location="cpu").numpy()
    d = json.load(open(datalist))
    items = d["data_list"]
    dsg, lt, ls = [], [], []
    nbad = 0
    for x in items:
        s = load_motion_135d(x["source_motion_path"], bone_offsets=bone, canonical=True)
        t = load_motion_135d(x["motion_path"], bone_offsets=bone, canonical=True)
        if s is None or t is None:
            nbad += 1
            continue
        dd = rmse(s, t)
        if dd is None:
            nbad += 1
            continue
        dsg.append(dd); lt.append(t.shape[0]); ls.append(s.shape[0])
    dsg = np.array(dsg)
    print(f"[datalist] {datalist}")
    print(f"  n={len(dsg)} (bad={nbad})")
    print(f"  d(src,tgt)  = {dsg.mean():.4f}  (min={dsg.min():.3f} max={dsg.max():.3f})")
    print(f"  frac d(src,tgt)<0.1 (degenerate) = {(dsg<0.1).mean()*100:.1f}%")
    print(f"  len src mean={np.mean(ls):.1f}  tgt mean={np.mean(lt):.1f}  "
          f"len-diff mean={np.mean(np.abs(np.array(lt)-np.array(ls))):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir")
    ap.add_argument("--datalist")
    a = ap.parse_args()
    if a.npz_dir:
        mode_npz(a.npz_dir)
    if a.datalist:
        mode_datalist(a.datalist)
