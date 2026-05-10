#!/usr/bin/env python3
"""Round-trip diagnostic for humanml3d_272 ↔ HumanML3D-263 ↔ humanml3d_272.

Pipeline per id (e.g. ``000021``):
    1. Load original ``humanml3d_272/motion_data/<id>.npy`` -> ``m272_orig`` (T_30, 272).
    2. Decode -> joints (T_30, 22, 3) global  [build_h3d263 path].
    3. Resample 30->20 fps via lerp -> joints (T_20, 22, 3).
    4. ``process_file`` -> ``m263`` (T_20, 263)  [build_h3d263 path].
    5. ``convert_momask263_to_h3d272.decode_263_to_pose`` ->
       (positions20, R_all20, _).
    6. Resample 20->30 fps -> (positions30, R_all30).
    7. ``encode_h3d272`` -> ``m272_recon`` (T_30, 272).

Compare ``m272_orig`` vs ``m272_recon`` block by block.

Usage::

    python3 tools/diag_h3d263_to_272_roundtrip.py \
        --src_h3d272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
        --ids 000021,000612,001000

Outputs per-block max-abs / mean-abs error and a few channel snippets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_h3d263_test_from_h3d272 import (  # noqa: E402
    decode_272_to_global_positions,
    linear_resample_positions,
)
from convert_momask263_to_h3d272 import (  # noqa: E402
    decode_263_to_pose,
    linear_resample_positions as lerp_pos_20to30,
    slerp_rotations,
    encode_h3d272,
)

# Initialise MoMask globals (process_file uses module-level constants).
import utils.motion_process as motion_process  # noqa: E402  (after sys.path setup is in build_h3d263)
from utils.motion_process import process_file  # noqa: E402
from utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain  # noqa: E402
from common.skeleton import Skeleton  # noqa: E402


def _setup_motion_process_globals(ref_first_frame_pos: np.ndarray):
    motion_process.l_idx1 = 5
    motion_process.l_idx2 = 8
    motion_process.fid_l = [7, 10]
    motion_process.fid_r = [8, 11]
    motion_process.face_joint_indx = [2, 1, 17, 16]
    motion_process.r_hip = 2
    motion_process.l_hip = 1
    motion_process.joints_num = 22
    motion_process.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    motion_process.kinematic_chain = t2m_kinematic_chain
    skel = Skeleton(motion_process.n_raw_offsets, motion_process.kinematic_chain, "cpu")
    motion_process.tgt_offsets = skel.get_offsets_joints(torch.from_numpy(ref_first_frame_pos).float())


_BLOCKS = [
    ("root_xz_vel  [0:2]",     0,   2),
    ("heading_d_6d [2:8]",     2,   8),
    ("joints_pos  [8:74]",     8,  74),
    ("joints_vel  [74:140]",  74, 140),
    ("joints_rot  [140:272]", 140, 272),
]


def block_summary(diff: np.ndarray, channels: tuple) -> dict:
    a, b = channels
    sl = diff[:, a:b]
    return {
        "max_abs": float(np.max(np.abs(sl))),
        "mean_abs": float(np.mean(np.abs(sl))),
        "rms": float(np.sqrt(np.mean(sl ** 2))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_h3d272", required=True)
    p.add_argument("--ids", default="000021,000612,001000,003000,005000")
    args = p.parse_args()

    src = Path(args.src_h3d272)
    ids = [s.strip() for s in args.ids.split(",") if s.strip()]

    # Build canonical reference skeleton from 000021 (mirrors build_h3d263 setup).
    ref = decode_272_to_global_positions(np.load(str(src / "motion_data" / "000021.npy")))
    ref20 = linear_resample_positions(ref, 30.0, 20.0)
    _setup_motion_process_globals(ref20[0])

    print(f"{'id':>10s} | {'block':25s} | {'max_abs':>10s} {'mean_abs':>10s} {'rms':>10s} | {'orig_std':>10s} {'recon_std':>10s}")
    print("-" * 110)

    for sid in ids:
        m_orig_file = src / "motion_data" / f"{sid}.npy"
        if not m_orig_file.exists():
            print(f"  [skip] {sid}: missing")
            continue
        m_orig = np.load(str(m_orig_file))  # (T_30, 272)
        if m_orig.shape[1] != 272 or len(m_orig) < 4:
            continue

        # Forward: 272 -> joints (30 fps) -> joints (20 fps) -> 263
        joints30 = decode_272_to_global_positions(m_orig)
        joints20 = linear_resample_positions(joints30, 30.0, 20.0)
        m263, _, _, _ = process_file(joints20, 0.002)  # (T_20, 263)

        # Backward: 263 -> (positions20, R_all20) -> 30 fps -> 272
        positions20, R_all20, _ = decode_263_to_pose(m263)
        positions30 = lerp_pos_20to30(positions20, 20.0, 30.0)
        R_all30 = slerp_rotations(R_all20, 20.0, 30.0)
        m_recon = encode_h3d272(positions30, R_all30)

        # Match length (lerp/slerp rounding may differ from m_orig length).
        T = min(len(m_orig), len(m_recon))
        diff = m_orig[:T] - m_recon[:T]

        for name, a, b in _BLOCKS:
            stats = block_summary(diff, (a, b))
            orig_std = float(np.std(m_orig[:T, a:b]))
            recon_std = float(np.std(m_recon[:T, a:b]))
            print(f"{sid:>10s} | {name:25s} | {stats['max_abs']:10.4f} {stats['mean_abs']:10.4f} "
                  f"{stats['rms']:10.4f} | {orig_std:10.4f} {recon_std:10.4f}")
        print("-" * 110)


if __name__ == "__main__":
    main()
