#!/usr/bin/env python3
"""Apply the official HY-Motion-1.0 inference smoothing to already-repacked
``t2m_viz/hymotion`` motion_135 clips, for *quick visual validation* of the
jitter fix without re-running the 26 GB model.

The official ``MotionFlowMatching`` decode applies quaternion-Gaussian SLERP
smoothing (sigma=1.0) to body rot6d and Savitzky-Golay (window=11, polyorder=5)
to the root translation. The repacked viewer clips were produced from the
*un-smoothed* generation, so they jitter. Here we smooth them at the rotation-
*matrix* level (convention-safe) and write a parallel ``hymotion_smooth`` dir so
the viewer can show raw-vs-smooth side by side.

NOTE: the authoritative fix lives in ``HyMotionT2MBundle.decode_motion_from_latent``
(applied at generation time); this script only smooths the repacked row-major
motion_135 for an immediate before/after on the website.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(REPO / "outputs/evaluation/t2m_viz/hymotion"))
    ap.add_argument("--out_dir", default=str(REPO / "outputs/evaluation/t2m_viz/hymotion_smooth"))
    ap.add_argument("--sigma", type=float, default=1.0)
    args = ap.parse_args()

    from hftrainer.motion.skeleton.fk import (
        rot6d_to_rotmat_row_major,
        rotmat_to_rot6d_row_major,
    )
    from hftrainer.models.motion.hymotion_t2m._smoothing import (
        matrix_to_quaternion,
        quaternion_to_matrix,
        smooth_rotation,
        smooth_with_savgol,
    )

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.npz"))
    print(f"[smooth] {len(files)} clips {in_dir} -> {out_dir} (sigma={args.sigma})", flush=True)
    for f in files:
        m135 = np.load(f)["motion_135"].astype(np.float32)  # (L,135) row-major
        L = m135.shape[0]
        transl = torch.from_numpy(m135[:, :3])
        rot6d_row = torch.from_numpy(m135[:, 3:135]).reshape(L, 22, 6)

        # row-major rot6d -> matrix -> quat -> Gaussian smooth -> matrix -> rot6d
        rotmat = rot6d_to_rotmat_row_major(rot6d_row)            # (L,22,3,3)
        quat = matrix_to_quaternion(rotmat).numpy()             # (L,22,4)
        quat_s = smooth_rotation(quat.copy(), sigma=args.sigma)  # (L,22,4)
        rotmat_s = quaternion_to_matrix(torch.from_numpy(quat_s))
        rot6d_row_s = rotmat_to_rot6d_row_major(rotmat_s).reshape(L, 132)

        transl_s = smooth_with_savgol(transl, window_length=11, polyorder=5)
        m135_s = torch.cat([transl_s, rot6d_row_s], dim=-1).float().numpy()
        np.savez_compressed(out_dir / f.name, motion_135=m135_s)
        print(f"[ok] {f.stem}: L={L}", flush=True)

    cap = in_dir / "captions.json"
    if cap.is_file():
        shutil.copy(cap, out_dir / "captions.json")
    print(f"[done] -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
