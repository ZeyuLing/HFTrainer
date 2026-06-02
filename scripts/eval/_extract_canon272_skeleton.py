#!/usr/bin/env python3
"""Extract the canonical SMPL-22 rest skeleton of the GT ``humanml3d_272`` set.

The GT 272 set is built from **SMPL-X** joints; FK'ing rotations on the SMPL-H
rest skeleton mis-places the upper body by ~210 mm (collar/neck/head/shoulder
rest positions differ), inflating the 272 FID. This script recovers the body
GT 272 was actually built with, directly from the data, by solving the
kinematic-tree offset that reproduces the stored joint positions::

    pos[j] = pos[parent] + Rg[parent] @ (rest[j] - rest[parent])
    => offset[j] = Rg[parent]^T @ (pos[j] - pos[parent])    (averaged over frames)

Output: ``scripts/eval/assets/bone_offsets_canon272.npy`` (22,3) parent-relative
offsets, consumed by :func:`motionstreamer_272_encoder.motion135_to_272`
(``skeleton="canon272"``, the default).

Usage::
    # GT 272 motion_data must be available (e.g. cached to /dev/shm):
    python3 scripts/eval/_extract_canon272_skeleton.py \
        --gt-dir /dev/shm/ms272_data/motion_data --n-clips 400
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from hftrainer.datasets.motion.representation.humanml_repr import (  # noqa: E402
    recover_local_rotations_and_root, recover_272_stored_positions,
    fk_smplh_joints, _load_smplh_rest, DEFAULT_PATHS,
)

NJ = 22


def _global_rot(rot, parents):
    T = rot.shape[0]
    Rg = np.zeros_like(rot)
    Rg[:, 0] = rot[:, 0]
    for j in range(1, NJ):
        Rg[:, j] = np.matmul(Rg[:, parents[j]], rot[:, j])
    return Rg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="/dev/shm/ms272_data/motion_data")
    ap.add_argument("--n-clips", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "bone_offsets_canon272.npy"))
    args = ap.parse_args()

    mp = DEFAULT_PATHS.resolve("smplh_model")
    _, parents = _load_smplh_rest(mp)
    ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(args.gt_dir, "*.npy")))
    ids = ids[:args.n_clips]

    acc = np.zeros((NJ, 3))
    cnt = 0
    for cid in ids:
        m = np.load(os.path.join(args.gt_dir, cid + ".npy"))
        sp = recover_272_stored_positions(m)
        rot, _ = recover_local_rotations_and_root(m)
        Rg = _global_rot(rot, parents)
        for j in range(1, NJ):
            p = parents[j]
            off = np.matmul(np.transpose(Rg[:, p], (0, 2, 1)),
                            (sp[:, j] - sp[:, p])[..., None]).squeeze(-1)
            acc[j] += off.sum(0)
        cnt += sp.shape[0]
    off_canon = (acc / cnt).astype(np.float64)
    off_canon[0] = 0.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, off_canon)
    print(f"[saved] {args.out}  (cnt={cnt} frames over {len(ids)} clips)")

    # quick validation: stored-vs-FK joint error, SMPL-H vs canonical
    def fk_off(rot, root, offs):
        T = rot.shape[0]
        Rg = np.zeros((T, NJ, 3, 3)); pos = np.zeros((T, NJ, 3))
        Rg[:, 0] = rot[:, 0]; pos[:, 0] = offs[0][None] + root
        for j in range(1, NJ):
            p = parents[j]; Rg[:, j] = np.matmul(Rg[:, p], rot[:, j])
            # off_canon is PARENT-RELATIVE (same convention as differentiable_fk
            # / motion135_to_fk) -> use offs[j] directly, NOT offs[j]-offs[p].
            pos[:, j] = pos[:, p] + np.matmul(Rg[:, p], offs[j][None, :, None]).squeeze(-1)
        return pos

    eh, ec = [], []
    for cid in ids[:80]:
        m = np.load(os.path.join(args.gt_dir, cid + ".npy"))
        sp = recover_272_stored_positions(m)
        rot, root = recover_local_rotations_and_root(m)
        eh.append(np.linalg.norm(sp - fk_smplh_joints(rot, root, mp), axis=-1).mean())
        ec.append(np.linalg.norm(sp - fk_off(rot, root, off_canon), axis=-1).mean())
    print(f"stored-vs-FK joint err: SMPL-H={np.mean(eh) * 1000:.1f}mm  "
          f"canon272={np.mean(ec) * 1000:.1f}mm")


if __name__ == "__main__":
    main()
