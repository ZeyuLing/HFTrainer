#!/usr/bin/env python3
"""Convert MoMask HumanML3D-263 (20 fps) outputs to HumanML3D-272 (30 fps).

This is a re-implementation that bridges the FPS gap between MoMask
(HumanML3D-263 @ 20 fps) and MotionStreamer's TMR-272 evaluator
(HumanML3D-272 @ 30 fps) so FID / R-Precision / MM-Dist / Diversity computed
on the 272 evaluator are not contaminated by the velocity scale mismatch.

Pipeline per sample:
    1. Decode 263 -> (a) joint positions (T20, 22, 3) global via recover_from_ric;
                     (b) root yaw angle yaw_t = cumsum(rot_velocity);
                     (c) non-root joint local rotation matrices R_local
                         (T20, 21, 3, 3) via cont6d_to_matrix on data[..., 67:193].
                     Root local rotation matrix = R_y(yaw_t) (HumanML3D 263
                     stores root rotation only via yaw).
    2. Upsample to 30 fps: T30 = round(T20 * 30 / 20):
         - positions_30: linear interpolation in time (per joint xyz).
         - rotations_30 (22 joints): slerp on quaternions via
           ``scipy.spatial.transform.Slerp``.
    3. Run a faithful copy of MotionStreamer's ``representation_272`` logic on
       the upsampled (positions, rotations) at 30 fps to produce (T30, 272).

The output layout matches ``humanml3d_272`` so the existing
``eval_with_motionstreamer_evaluator.py`` script can consume it directly.

Usage::

    python3 tools/convert_momask263_to_h3d272.py \
        --pred_dir_263 work_dirs/momask_eval/momask_pred_263 \
        --out_dir_272  work_dirs/momask_eval/momask_pred_272_30fps \
        --src_fps 20 --dst_fps 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.transform import Slerp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup -- import MoMask's recover_from_ric / cont6d_to_matrix.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
MOMASK_ROOT = REPO_ROOT / "ref_repo" / "Momask" / "momask-codes"
sys.path.insert(0, str(MOMASK_ROOT))

from utils.motion_process import recover_from_ric, recover_root_rot_pos  # noqa: E402
from common.quaternion import cont6d_to_matrix  # noqa: E402


# ---------------------------------------------------------------------------
# Decoder: 263 (20 fps) -> (positions, rotations) global at 20 fps
# ---------------------------------------------------------------------------

def decode_263_to_pose(motion_263: np.ndarray):
    """Decode (T, 263) -> (positions [T,22,3] global, rotations [T,22,3,3] local).

    The local rotations are *parent-relative* SMPL local rotations, with the
    root rotation set to a pure-yaw rotation R_y(cumsum(rot_velocity)) since
    HumanML3D-263 only stores root yaw via rot_velocity.
    """
    motion_t = torch.from_numpy(motion_263).float()
    positions = recover_from_ric(motion_t, 22).numpy()  # (T, 22, 3)

    rot_block = motion_263[..., 67:193].reshape(-1, 21, 6)
    R_nonroot = cont6d_to_matrix(torch.from_numpy(rot_block).float()).numpy()  # (T, 21, 3, 3)

    # MoMask convention: rot_velocity stores the *half-angle* of the frame-to-frame
    # yaw rotation (it comes from arcsin(quat_y) where quat = (cos(θ/2), 0, sin(θ/2), 0)).
    # Therefore cumsum(rot_velocity) = θ_total / 2, and the actual root yaw rotation is
    # R_y(θ_total) = R_y(2 * cumsum(rot_velocity)).  The previous version used
    # R_y(cumsum(rot_velocity)) which is exactly half the correct angle.
    rot_vel = motion_263[..., 0]
    half_yaw = np.zeros_like(rot_vel)
    half_yaw[1:] = np.cumsum(rot_vel[:-1])
    yaw_t = 2.0 * half_yaw  # actual rotation angle, radians
    c = np.cos(yaw_t)
    s = np.sin(yaw_t)
    R_root = np.zeros((len(yaw_t), 3, 3), dtype=np.float64)
    R_root[:, 0, 0] = c
    R_root[:, 0, 2] = s
    R_root[:, 1, 1] = 1.0
    R_root[:, 2, 0] = -s
    R_root[:, 2, 2] = c

    R_all = np.concatenate([R_root[:, None], R_nonroot], axis=1).astype(np.float64)  # (T, 22, 3, 3)
    return positions.astype(np.float64), R_all, yaw_t


# ---------------------------------------------------------------------------
# Temporal resampling: 20 fps -> 30 fps
# ---------------------------------------------------------------------------

def linear_resample_positions(pos_src: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linearly interpolate (T_src, J, 3) along time."""
    T_src, J, _ = pos_src.shape
    duration = (T_src - 1) / src_fps
    T_dst = int(round(duration * dst_fps)) + 1
    T_dst = max(2, T_dst)
    src_times = np.arange(T_src, dtype=np.float64) / src_fps
    dst_times = np.linspace(0.0, duration, T_dst)

    pos_dst = np.empty((T_dst, J, 3), dtype=np.float64)
    for j in range(J):
        for d in range(3):
            pos_dst[:, j, d] = np.interp(dst_times, src_times, pos_src[:, j, d])
    return pos_dst


def slerp_rotations(R_src: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Slerp (T_src, J, 3, 3) along time."""
    T_src, J, _, _ = R_src.shape
    duration = (T_src - 1) / src_fps
    T_dst = int(round(duration * dst_fps)) + 1
    T_dst = max(2, T_dst)
    src_times = np.arange(T_src, dtype=np.float64) / src_fps
    dst_times = np.linspace(0.0, duration, T_dst)

    R_dst = np.empty((T_dst, J, 3, 3), dtype=np.float64)
    for j in range(J):
        try:
            rots = ScipyRotation.from_matrix(R_src[:, j])
            slerp = Slerp(src_times, rots)
            R_dst[:, j] = slerp(dst_times).as_matrix()
        except Exception:
            # Fallback to nearest-neighbour for degenerate cases.
            idx = np.clip(np.round(dst_times * src_fps).astype(np.int64), 0, T_src - 1)
            R_dst[:, j] = R_src[idx, j]
    return R_dst


# ---------------------------------------------------------------------------
# Encoder: (positions, rotations) at any fps -> (T, 272)
# ---------------------------------------------------------------------------

def encode_h3d272(positions: np.ndarray, R_all: np.ndarray) -> np.ndarray:
    """Encode (T, 22, 3) global positions + (T, 22, 3, 3) local rotations to (T, 272).

    Faithful re-implementation of MotionStreamer's ``representation_272`` logic.
    """
    T, J, _ = positions.shape
    assert J == 22

    position_data = positions.astype(np.float64).copy()

    # 1) Anchor root xz to origin per frame (root-centred xz).
    position_data[:, :, 0] -= position_data[:, 0:1, 0]
    position_data[:, :, 2] -= position_data[:, 0:1, 2]

    # 2) Per-frame heading from root rotation matrix.
    #    global_heading = -atan2(R_root[0, 2], R_root[2, 2]) (mirrors MS code).
    global_heading = -np.arctan2(R_all[:, 0, 0, 2], R_all[:, 0, 2, 2])
    cg = np.cos(global_heading)
    sg = np.sin(global_heading)
    global_heading_rot = np.zeros((T, 3, 3), dtype=np.float64)
    global_heading_rot[:, 0, 0] = cg
    global_heading_rot[:, 0, 2] = sg
    global_heading_rot[:, 1, 1] = 1.0
    global_heading_rot[:, 2, 0] = -sg
    global_heading_rot[:, 2, 2] = cg

    # 3) Heading delta (frame-to-frame) for the first 6 dims of final_x.
    heading_delta_rot = np.einsum("tab,tbc->tac", global_heading_rot[1:], np.linalg.inv(global_heading_rot[:-1]))

    # 4) Apply heading rotation to remove yaw from positions and root rotation.
    positions_no_heading = np.einsum("tab,tjb->tja", global_heading_rot, position_data)

    R_root_no_heading = np.einsum("tab,tbc->tac", global_heading_rot, R_all[:, 0])
    R_no_heading = R_all.copy()
    R_no_heading[:, 0] = R_root_no_heading

    # 5) Velocities (no heading) = positions_no_heading[t+1] - positions_no_heading[t].
    velocities_no_heading = np.diff(positions_no_heading, axis=0)
    velocities_no_heading = np.concatenate(
        [np.zeros((1, J, 3), dtype=np.float64), velocities_no_heading], axis=0
    )

    # 6) Root xz velocity = velocities_no_heading[:, 0, [0, 2]] but MotionStreamer
    #    instead uses the *delta of root xz position* before heading anchoring.
    #    Equivalently we recover it from the original (pre-anchor) position data:
    root_xz = positions[:, 0, [0, 2]].astype(np.float64)  # (T, 2) in global frame
    root_xz_delta = np.diff(root_xz, axis=0)
    # Rotate delta into the t-th frame's heading-canonical frame.
    cg2d = np.cos(global_heading[:-1])  # (T-1,)
    sg2d = np.sin(global_heading[:-1])
    rot_xz = np.stack(
        [cg2d * root_xz_delta[:, 0] + sg2d * root_xz_delta[:, 1],
         -sg2d * root_xz_delta[:, 0] + cg2d * root_xz_delta[:, 1]],
        axis=-1,
    )
    root_xz_vel_no_heading = np.concatenate([np.zeros((1, 2), dtype=np.float64), rot_xz], axis=0)

    # 7) Pack into (T, 272).
    final = np.zeros((T, 272), dtype=np.float32)
    final[:, 0:2] = root_xz_vel_no_heading.astype(np.float32)
    # heading delta rot 6D row-major (first 2 rows of heading_delta_rot)
    heading_delta_6d = np.zeros((T, 6), dtype=np.float64)
    heading_delta_6d[1:] = heading_delta_rot[:, :2, :].reshape(-1, 6)
    heading_delta_6d[0] = np.array([1.0, 0, 0, 0, 1.0, 0])
    final[:, 2:8] = heading_delta_6d.astype(np.float32)
    final[:, 8:74] = positions_no_heading.reshape(T, -1).astype(np.float32)
    final[:, 74:140] = velocities_no_heading.reshape(T, -1).astype(np.float32)
    rot_6d_rowmajor = R_no_heading[:, :, :2, :].reshape(T, J, 6)
    final[:, 140:272] = rot_6d_rowmajor.reshape(T, -1).astype(np.float32)
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir_263", required=True)
    p.add_argument("--out_dir_272", required=True)
    p.add_argument("--src_fps", type=float, default=20.0)
    p.add_argument("--dst_fps", type=float, default=30.0)
    args = p.parse_args()

    src = Path(args.pred_dir_263)
    dst = Path(args.out_dir_272)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.npy"))
    print(f"[+] {len(files)} input files in {src}")
    print(f"[+] resampling {args.src_fps} fps -> {args.dst_fps} fps")

    n_ok = n_err = 0
    for f in tqdm(files, ncols=80):
        try:
            m263 = np.load(str(f))
            if m263.ndim != 2 or m263.shape[1] != 263 or len(m263) < 2:
                n_err += 1
                continue
            positions, R_all, _yaw = decode_263_to_pose(m263)
            if abs(args.dst_fps - args.src_fps) > 1e-6:
                positions = linear_resample_positions(positions, args.src_fps, args.dst_fps)
                R_all = slerp_rotations(R_all, args.src_fps, args.dst_fps)
            m272 = encode_h3d272(positions, R_all)
            np.save(str(dst / f.name), m272)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"  [!] {f.name}: {e}")

    print(f"[+] wrote {n_ok} files to {dst} ({n_err} errors)")


if __name__ == "__main__":
    main()
