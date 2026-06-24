#!/usr/bin/env python3
"""Build MS272 hybrid diagnostics to isolate non-rotation vs local-rotation FID."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hftrainer.motion.representation.motion272 import motion135_to_272


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="NPZ dir with motion_135")
    ap.add_argument("--gt-dir", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data")
    ap.add_argument("--split", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--max-ids", type=int, default=0)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_root = Path(args.out_root)
    rot_only = out_root / "gt_nonrot_ours_rot"
    nonrot_only = out_root / "ours_nonrot_gt_rot"
    root_only = out_root / "gt_plus_ours_root_heading"
    posvel_only = out_root / "gt_plus_ours_pos_vel"
    for d in (rot_only, nonrot_only, root_only, posvel_only):
        d.mkdir(parents=True, exist_ok=True)

    ids = [x.strip() for x in Path(args.split).read_text().splitlines() if x.strip()]
    if args.max_ids > 0:
        ids = ids[: args.max_ids]

    done = 0
    skipped = 0
    for cid in tqdm(ids, ncols=90):
        pf = pred_dir / f"{cid}.npz"
        gf = gt_dir / f"{cid}.npy"
        if not (pf.exists() and gf.exists()):
            skipped += 1
            continue
        gt = np.asarray(np.load(gf), dtype=np.float32)
        z = np.load(pf, allow_pickle=True)
        pred = motion135_to_272(np.asarray(z["motion_135"], dtype=np.float32)).astype(np.float32)
        T = min(len(gt), len(pred))
        if T < 2:
            skipped += 1
            continue
        gt = gt[:T]
        pred = pred[:T]

        a = gt.copy()
        a[:, 140:272] = pred[:, 140:272]
        np.savez_compressed(rot_only / f"{cid}.npz", motion_272=a)

        b = gt.copy()
        b[:, 0:140] = pred[:, 0:140]
        np.savez_compressed(nonrot_only / f"{cid}.npz", motion_272=b)

        c = gt.copy()
        c[:, 0:8] = pred[:, 0:8]
        np.savez_compressed(root_only / f"{cid}.npz", motion_272=c)

        e = gt.copy()
        e[:, 8:140] = pred[:, 8:140]
        np.savez_compressed(posvel_only / f"{cid}.npz", motion_272=e)
        done += 1

    print(f"[done] wrote={done} skipped={skipped} out={out_root}")


if __name__ == "__main__":
    main()
