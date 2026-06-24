#!/usr/bin/env python3
"""MBench ``Pose_Quality`` (NRDF naturalness distance) on our SMPL-22 motions.

MBench computes Pose_Quality as ``mean(NRDF.dist_pred) * 10`` over the SMPL body
pose (21 joints, no root). Their pipeline needs SMPLify ``.pt`` files, but our
``motion_135`` already stores the per-joint local rotations, so we feed the 21
body-joint rotations straight into the same pretrained NRDF model
(``ref_repo/ViMoGen/checkpoints/nrdf/...``). Lower = more natural.

Reuses the same manifest as ``compute_phys_h3d.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
_VIMOGEN = os.path.join(_ROOT, "ref_repo", "ViMoGen")
for p in (_ROOT, _VIMOGEN):
    if p not in sys.path:
        sys.path.insert(0, p)

from scipy.spatial.transform import Rotation as R  # noqa: E402
from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    recover_local_rotations_and_root,
)

_NRDF_DIR = os.path.join(
    _VIMOGEN, "checkpoints/nrdf/amass_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1")
_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _rot6d_to_rotmat_rowmajor(d6):
    # d6: (...,6) row-major (two columns). Returns (...,3,3).
    x = d6.reshape(*d6.shape[:-1], 3, 2)
    a1, a2 = x[..., 0], x[..., 1]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2p = a2 - (np.sum(b1 * a2, axis=-1, keepdims=True)) * b1
    b2 = a2p / (np.linalg.norm(a2p, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _body_axis_angle(path, mode):
    """Return (T,21,3) axis-angle for SMPL body joints 1..21, or None."""
    if mode == "m135":
        d = np.load(path, allow_pickle=True)
        if "motion_135" not in d:
            return None
        m = np.asarray(d["motion_135"], np.float32)
        rot6d = m[:, 3:135].reshape(-1, 22, 6)[:, 1:22]        # (T,21,6)
        rotmat = _rot6d_to_rotmat_rowmajor(rot6d)               # (T,21,3,3)
    else:
        if path.endswith(".npz"):
            d = np.load(path, allow_pickle=True)
            if "motion_272" not in d:
                return None
            m = np.asarray(d["motion_272"], np.float32)
        else:
            m = np.asarray(np.load(path), np.float32)
        if m.ndim != 2 or m.shape[1] != 272:
            return None
        rot, _ = recover_local_rotations_and_root(m)            # (T,22,3,3)
        rotmat = np.asarray(rot, np.float32)[:, 1:22]
    T = rotmat.shape[0]
    aa = R.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 21, 3)
    return aa.astype(np.float32)


def _list_files(src, mode, limit, seed):
    suffixes = [".npz"] if mode == "m135" else [".npy", ".npz"]
    files = sorted(
        e.path
        for e in os.scandir(src)
        if any(e.name.endswith(suffix) for suffix in suffixes)
    )
    if limit and len(files) > limit:
        rng = np.random.RandomState(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), limit, False))]
    return files


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--m135-dir")
    g.add_argument("--gt272-dir")
    g.add_argument("--manifest")
    ap.add_argument("--tag", default="method")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.manifest:
        methods = []
        for ln in open(args.manifest):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            tag, mode, d = ln.split("\t")
            methods.append((tag, mode, d))
    elif args.m135_dir:
        methods = [(args.tag, "m135", args.m135_dir)]
    else:
        methods = [(args.tag, "gt272", args.gt272_dir)]

    from mbench.third_party.NRDF import load_model, axis_angle_to_quaternion
    nrdf = load_model(_NRDF_DIR)
    print("[nrdf] loaded", flush=True)

    out = {}
    for tag, mode, d in methods:
        if not os.path.isdir(d):
            print(f"[skip] {tag}: missing {d}", flush=True)
            continue
        files = _list_files(d, mode, args.limit, args.seed)
        clip_scores = []
        for fp in files:
            aa = _body_axis_angle(fp, mode)
            if aa is None or aa.shape[0] < 1:
                continue
            t = torch.from_numpy(aa).to(_DEV)               # (T,21,3)
            q = axis_angle_to_quaternion(t)                  # (T,21,4)
            dists = []
            with torch.no_grad():
                for i in range(0, q.shape[0], args.batch):
                    dp = nrdf(q[i:i + args.batch], train=False)["dist_pred"]
                    dists.append(dp.flatten())
            clip_scores.append(float(torch.cat(dists).mean().item()) * 10.0)
        n = len(clip_scores)
        mean = float(np.mean(clip_scores)) if n else 0.0
        out[tag] = {"n": n, "PoseQuality": mean}
        print(f"[TABLE] {tag}  n={n}  PoseQuality={mean:.4f}", flush=True)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        json.dump(out, open(args.out_json, "w"), indent=1)
        print(f"[done] -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
