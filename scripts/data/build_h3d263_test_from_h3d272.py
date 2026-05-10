#!/usr/bin/env python3
"""Reconstruct the HumanML3D-263 test set from MotionStreamer's humanml3d_272.

MotionStreamer's ``humanml3d_272/motion_data/<id>.npy`` (30 fps) is the same
underlying motion as HumanML3D's ``new_joint_vecs/<id>.npy`` (20 fps), just in
a different representation and FPS. We can rebuild the HumanML3D-263 test
data needed by MoMask's native evaluator (Comp_v6_KLD005 / text_mot_match) by:

    1. Decode (T_30, 272) -> global joint positions (T_30, 22, 3) at 30 fps.
    2. Downsample to 20 fps via linear interpolation.
    3. Run HumanML3D's ``process_file`` to get (T_20, 263) features +
       (T_20, 22, 3) joints.
    4. Save to a HumanML3D-style directory layout:
            <out_root>/new_joint_vecs/<id>.npy  -- shape (T_20, 263)
            <out_root>/new_joints/<id>.npy      -- shape (T_20, 22, 3)
            <out_root>/Mean.npy / Std.npy        -- 263-dim, copied from
                                                    versatilemotion
            <out_root>/test.txt                  -- list of <id> with successful
                                                    reconstructions

Usage::

    python3 tools/build_h3d263_test_from_h3d272.py \
        --src_h3d272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
        --src_meanstd_263 \
            /apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/checkpoints/tm2t/t2m/Comp_v6_KLD005/meta \
        --out_root work_dirs/momask_eval/h3d263_test \
        --src_fps 30 --dst_fps 20

The resulting directory is then plugged into a MoMask-eval-friendly opt
file (``checkpoints/t2m/Comp_v6_KLD005/opt.txt``) by editing
``data_root`` to point at it (or by using a custom standalone eval script
that reads the directory directly).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add MoMask code so we can use process_file (== HumanML3D's algo).
REPO_ROOT = Path(__file__).resolve().parents[1]
MOMASK_ROOT = REPO_ROOT / "ref_repo" / "Momask" / "momask-codes"
sys.path.insert(0, str(MOMASK_ROOT))

import utils.motion_process as motion_process  # noqa: E402
from utils.motion_process import process_file  # noqa: E402
from utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain  # noqa: E402
from common.skeleton import Skeleton  # noqa: E402  -- ensure init


# ---------------------------------------------------------------------------
# 272 -> global joint positions (T, 22, 3)
# ---------------------------------------------------------------------------

def _rotation_6d_rowmajor_to_matrix(c6: np.ndarray) -> np.ndarray:
    """Row-major 6D = [R[0,:]; R[1,:]] -> 3x3 (Gram-Schmidt orthogonalise)."""
    a = c6[..., 0:3]
    b = c6[..., 3:6]
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    z = np.cross(a, b)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
    y = np.cross(z, a)
    R = np.stack([a, y, z], axis=-2)  # rows: a, y, z
    return R


def decode_272_to_global_positions(motion_272: np.ndarray) -> np.ndarray:
    """Decode (T, 272) HumanML3D-272 feature -> (T, 22, 3) GLOBAL joint positions.

    Mirrors / inverts the forward pipeline in ``representation_272.py``.

    Layout (for reference):
        [..., 0:2]    : root xz vel (no heading)
        [..., 2:8]    : heading delta rot 6D (row-major; identity at frame 0)
        [..., 8:74]   : 22 joint positions (no heading, root xz = 0)
        [..., 74:140] : 22 joint velocities (no heading) -- not used for decode
        [..., 140:272]: 22 joint local rotations 6D (no heading) -- not used here
    """
    T = motion_272.shape[0]
    assert motion_272.shape[1] == 272

    # 1) Recover global yaw trajectory by integrating heading_delta_rot (R_y delta).
    heading_delta_6d = motion_272[:, 2:8].astype(np.float64)
    R_delta = _rotation_6d_rowmajor_to_matrix(heading_delta_6d)
    delta_yaw = np.arctan2(R_delta[:, 0, 2], R_delta[:, 0, 0])  # delta angle of R_y
    global_heading = np.cumsum(delta_yaw)  # global_heading[0] = delta_yaw[0] (=0 by construction)
    # global_heading[t] = sum_{k <= t} delta_yaw[k]; this corresponds to MS's
    # `global_heading[t]` = heading-removal angle at frame t.
    # In MS's forward code:
    #     positions_no_heading[t] = R_y(global_heading[t]) @ position_data[t]
    # so to recover positions_global from positions_no_heading we need
    #     position_data[t] = R_y(-global_heading[t]) @ positions_no_heading[t]
    # (then add back the root xz origin).

    # 2) Recover the per-frame root xz translation.
    #    In MS forward:
    #       root_xz_velocity_no_heading[t] = R_y(global_heading[t-1]) @ (root_xz[t] - root_xz[t-1])
    #    So root_xz[t] = root_xz[t-1] + R_y(-global_heading[t-1]) @ root_xz_velocity[t]
    root_xz_vel = motion_272[:, 0:2].astype(np.float64)  # (T, 2)
    root_xz = np.zeros((T, 2), dtype=np.float64)
    if T > 1:
        for t in range(1, T):
            ang = -global_heading[t - 1]
            c, s = np.cos(ang), np.sin(ang)
            v = root_xz_vel[t]
            root_xz[t, 0] = root_xz[t - 1, 0] + c * v[0] + s * v[1]
            root_xz[t, 1] = root_xz[t - 1, 1] - s * v[0] + c * v[1]

    # 3) Recover positions in heading-canonical frame, then un-rotate per-frame.
    pos_no_heading = motion_272[:, 8:74].reshape(T, 22, 3).astype(np.float64)
    cos_g = np.cos(-global_heading)
    sin_g = np.sin(-global_heading)
    # R_y(-g) @ position_no_heading
    px = cos_g[:, None] * pos_no_heading[:, :, 0] + sin_g[:, None] * pos_no_heading[:, :, 2]
    py = pos_no_heading[:, :, 1]
    pz = -sin_g[:, None] * pos_no_heading[:, :, 0] + cos_g[:, None] * pos_no_heading[:, :, 2]
    positions_global = np.stack([px, py, pz], axis=-1)
    positions_global[:, :, 0] += root_xz[:, 0:1]
    positions_global[:, :, 2] += root_xz[:, 1:2]
    return positions_global


# ---------------------------------------------------------------------------
# Linear resampler
# ---------------------------------------------------------------------------

def linear_resample_positions(pos_src: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if abs(src_fps - dst_fps) < 1e-6:
        return pos_src.astype(np.float64)
    T_src, J, _ = pos_src.shape
    duration = (T_src - 1) / src_fps
    T_dst = int(round(duration * dst_fps)) + 1
    T_dst = max(2, T_dst)
    src_t = np.arange(T_src, dtype=np.float64) / src_fps
    dst_t = np.linspace(0.0, duration, T_dst)
    pos_dst = np.empty((T_dst, J, 3), dtype=np.float64)
    for j in range(J):
        for d in range(3):
            pos_dst[:, j, d] = np.interp(dst_t, src_t, pos_src[:, j, d])
    return pos_dst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_h3d272", required=True)
    p.add_argument("--src_meanstd_263", required=True,
                   help="Directory with HumanML3D-263 'mean.npy' & 'std.npy' "
                        "(e.g. versatilemotion .../Comp_v6_KLD005/meta/).")
    p.add_argument("--out_root", required=True)
    p.add_argument("--src_fps", type=float, default=30.0)
    p.add_argument("--dst_fps", type=float, default=20.0)
    p.add_argument("--feet_thre", type=float, default=0.002)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--split_name", default="test")
    args = p.parse_args()

    src = Path(args.src_h3d272)
    out = Path(args.out_root)
    (out / "new_joint_vecs").mkdir(parents=True, exist_ok=True)
    (out / "new_joints").mkdir(parents=True, exist_ok=True)

    # ----- 1. Inject MoMask's HumanML3D-263 globals (defined inside their
    #         __main__ block) into the motion_process module so process_file()
    #         works when imported.
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

    # ----- 2. Reference skeleton: decode 000021 (HumanML3D's canonical example)
    #         from humanml3d_272, downsample to dst_fps, take frame 0.
    print("[+] Building reference skeleton from 000021 ...")
    ref_m272 = np.load(str(src / "motion_data" / "000021.npy"))
    ref_pos = decode_272_to_global_positions(ref_m272)
    ref_pos_resampled = linear_resample_positions(ref_pos, args.src_fps, args.dst_fps)
    ref_first = torch.from_numpy(ref_pos_resampled[0]).float()
    tgt_skel = Skeleton(motion_process.n_raw_offsets, motion_process.kinematic_chain, "cpu")
    motion_process.tgt_offsets = tgt_skel.get_offsets_joints(ref_first)
    print(f"    tgt_offsets shape={tuple(motion_process.tgt_offsets.shape)}")

    split_file = src / "split" / f"{args.split_name}.txt"
    ids = [s.strip() for s in split_file.read_text().splitlines() if s.strip()]
    if args.max_samples:
        ids = ids[: args.max_samples]
    print(f"[+] {len(ids)} ids in {args.split_name}")

    out_ids = []
    for sid in tqdm(ids, ncols=80):
        m_file = src / "motion_data" / f"{sid}.npy"
        if not m_file.exists():
            continue
        try:
            m272 = np.load(str(m_file))
            if m272.ndim != 2 or m272.shape[1] != 272 or len(m272) < 4:
                continue
            pos_30 = decode_272_to_global_positions(m272)
            pos_20 = linear_resample_positions(pos_30, args.src_fps, args.dst_fps)
            data, joints, _, _ = process_file(pos_20, args.feet_thre)
            if data.shape[0] < 40:
                continue
            np.save(str(out / "new_joint_vecs" / f"{sid}.npy"), data.astype(np.float32))
            np.save(str(out / "new_joints" / f"{sid}.npy"), joints.astype(np.float32))
            out_ids.append(sid)
        except Exception as e:
            print(f"  [!] {sid}: {e}")

    # Copy mean/std for HumanML3D-263 normalization.
    src_ms = Path(args.src_meanstd_263)
    shutil.copy(str(src_ms / "mean.npy"), str(out / "Mean.npy"))
    shutil.copy(str(src_ms / "std.npy"), str(out / "Std.npy"))

    # Split file with successfully reconstructed ids.
    (out / f"{args.split_name}.txt").write_text("\n".join(out_ids) + "\n")

    print(f"[+] wrote {len(out_ids)} / {len(ids)} reconstructions to {out}")
    print(f"[+] mean/std @ {out / 'Mean.npy'} ({(out / 'Mean.npy').stat().st_size} B)")


if __name__ == "__main__":
    main()
