#!/usr/bin/env python3
"""Convert HumanML3D-263 motion features into HumanML3D-272 features.

Targets MoMask outputs (which live in HumanML3D-263 space) so that they can be
evaluated by MotionStreamer's TMR-272 evaluator without modifying the eval
pipeline.

HumanML3D-263 layout (from HumanML3D + MoMask ``utils/motion_process.py``):
    [..., 0]      : root rot velocity (yaw delta), 1 dim
    [..., 1:3]    : root linear xz velocity, 2 dims (in heading-canonical frame)
    [..., 3]      : root y position, 1 dim
    [..., 4:67]   : 21 non-root joint positions in per-frame heading-canonical
                    frame, root at xz origin, 21*3 = 63 dims
    [..., 67:193] : 21 non-root joint rotations, **column-major** 6D
                    [R[:,0]; R[:,1]] per joint, 21*6 = 126 dims
    [..., 193:259]: 22 joint velocities in heading-canonical frame, 22*3 = 66 dims
    [..., 259:263]: 4 foot contact, 4 dims
    Total: 4 + 63 + 126 + 66 + 4 = 263

HumanML3D-272 layout (from MotionStreamer ``representation_272.py``):
    [..., 0:2]   : root xz velocity (no heading), 2 dims; [0, :2] = 0
    [..., 2:8]   : heading delta rotation 6D (R_y(yaw[t+1] - yaw[t])),
                   **row-major** first 2 rows; [0, 2:8] = identity6D
    [..., 8:74]  : 22 joint positions (no heading, root at xz origin), 22*3 = 66
    [..., 74:140]: 22 joint velocities (no heading), 22*3 = 66; [0, 74:140] = 0
    [..., 140:272]: 22 joint local rotations, **row-major** 6D
                    [R[0,:]; R[1,:]] per joint, 22*6 = 132
    Total: 8 + 66 + 66 + 132 = 272

Mapping (lossless except root rotation, which 263 doesn't carry):
    272[..., 0:2]      <- 263[..., 1:3]                      (root xz vel)
    272[..., 2:8]      <- R_y(263[..., 0]) -> 6D row-major   (heading delta rot)
    272[..., 8:11]     <- (0, 263[..., 3], 0)                (root joint pos)
    272[..., 11:74]    <- 263[..., 4:67]                     (21 non-root pos)
    272[..., 74:140]   <- 263[..., 193:259]                  (22 joint vels)
    272[..., 140:146]  <- identity 6D                         (root rotation
                                                                no heading; 263
                                                                represents root
                                                                only via yaw, so
                                                                no-heading root
                                                                rotation = I)
    272[..., 146:272]  <- col-major-to-row-major(263[..., 67:193])

Usage::

    python3 tools/convert_hml263_to_h3d272.py \
        --pred_dir_263 work_dirs/momask_eval/momask_pred_263 \
        --out_dir_272  work_dirs/momask_eval/momask_pred_272

The output is a flat directory of ``<id>.npy`` files (T, 272).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def cont6d_colmajor_to_matrix(cont6d: np.ndarray) -> np.ndarray:
    """Column-major 6D = [R[:,0]; R[:,1]] -> 3x3 rotation matrix.

    Args:
        cont6d: (..., 6)
    Returns:
        R: (..., 3, 3)
    """
    a = cont6d[..., 0:3]
    b = cont6d[..., 3:6]
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    z = np.cross(a, b, axis=-1)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
    y = np.cross(z, a, axis=-1)
    R = np.stack([a, y, z], axis=-1)  # columns = a, y, z (right-handed)
    return R


def matrix_to_rowmajor_6d(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> row-major 6D = [R[0,:]; R[1,:]].

    Args:
        R: (..., 3, 3)
    Returns:
        cont6d: (..., 6)
    """
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def cont6d_colmajor_to_rowmajor(cont6d: np.ndarray) -> np.ndarray:
    """Convert column-major 6D directly to row-major 6D."""
    R = cont6d_colmajor_to_matrix(cont6d)
    return matrix_to_rowmajor_6d(R)


def rotation_y_to_rowmajor_6d(angle: np.ndarray) -> np.ndarray:
    """R_y(angle) -> row-major 6D.

    R_y(t) = [[ cos t, 0, sin t],
              [   0,   1,   0  ],
              [-sin t, 0, cos t]]
    First 2 rows: [cos t, 0, sin t, 0, 1, 0]

    Args:
        angle: (...)
    Returns:
        cont6d: (..., 6)
    """
    c = np.cos(angle)
    s = np.sin(angle)
    z = np.zeros_like(angle)
    o = np.ones_like(angle)
    return np.stack([c, z, s, z, o, z], axis=-1)


# ---------------------------------------------------------------------------
# 263 -> 272 conversion
# ---------------------------------------------------------------------------

IDENTITY_6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def hml263_to_h3d272(motion_263: np.ndarray) -> np.ndarray:
    """Convert a (T, 263) HumanML3D feature to a (T, 272) HumanML3D-272 feature.

    Args:
        motion_263: (T, 263) float, un-standardized.
    Returns:
        motion_272: (T, 272) float32.
    """
    motion_263 = motion_263.astype(np.float64)
    T = motion_263.shape[0]
    assert motion_263.shape[1] == 263, f"expected 263-dim, got {motion_263.shape[1]}"

    # ---- 1. heading delta rot 6D ----
    # In 263, [..., 0] is rot_velocity at frame t describing the *next* frame
    # transition (matching MoMask's recover_root_rot_pos: r_rot_ang[1:] = rot_vel[:-1]).
    # In 272, final_x[1:, 2:8] = matrix_to_rotation_6d(global_heading_diff_rot)
    # where global_heading_diff[t] = global_heading[t+1] - global_heading[t].
    # By inspection these are the same delta-yaw, so:
    rot_vel = motion_263[..., 0]  # (T,)
    heading_diff_rot_6d = rotation_y_to_rowmajor_6d(rot_vel[1:])  # (T-1, 6)

    out_2_8 = np.zeros((T, 6), dtype=np.float32)
    out_2_8[0] = IDENTITY_6D
    out_2_8[1:] = heading_diff_rot_6d.astype(np.float32)

    # ---- 2. root xz velocity (no heading) ----
    # 263[..., 1:3] is "root linear velocity in heading-canonical frame", 2 dims.
    # 272[..., :2] is root xz velocity no heading, also 2 dims.
    out_0_2 = np.zeros((T, 2), dtype=np.float32)
    out_0_2[1:] = motion_263[1:, 1:3].astype(np.float32)
    # Note: 272 sets [0, :2] = 0 (the first frame has no preceding velocity).

    # ---- 3. joint positions (no heading, root at xz origin) ----
    root_y = motion_263[..., 3:4]                               # (T, 1)
    nonroot_pos = motion_263[..., 4:67].reshape(T, 21, 3)        # (T, 21, 3)
    root_pos = np.concatenate(
        [np.zeros((T, 1), dtype=np.float64), root_y, np.zeros((T, 1), dtype=np.float64)],
        axis=-1,
    ).reshape(T, 1, 3)
    positions_22 = np.concatenate([root_pos, nonroot_pos], axis=1)  # (T, 22, 3)
    out_8_74 = positions_22.reshape(T, -1).astype(np.float32)

    # ---- 4. joint velocities (no heading) ----
    # 263[..., 193:259] = local_velocity for all 22 joints.
    out_74_140 = motion_263[..., 193:259].astype(np.float32)

    # ---- 5. joint rotations 6D (no heading) ----
    # Root rotation 6D no heading -> identity (HumanML3D 263 carries no
    # explicit root tilt; only yaw, which is already removed by face_z).
    nonroot_6d_col = motion_263[..., 67:193].reshape(T, 21, 6)
    nonroot_6d_row = cont6d_colmajor_to_rowmajor(nonroot_6d_col)
    rotations_22_6d = np.zeros((T, 22, 6), dtype=np.float64)
    rotations_22_6d[:, 0] = IDENTITY_6D
    rotations_22_6d[:, 1:] = nonroot_6d_row
    out_140_272 = rotations_22_6d.reshape(T, -1).astype(np.float32)

    final_x = np.zeros((T, 272), dtype=np.float32)
    final_x[:, :2] = out_0_2
    final_x[:, 2:8] = out_2_8
    final_x[:, 8:74] = out_8_74
    final_x[:, 74:140] = out_74_140
    final_x[:, 140:272] = out_140_272
    return final_x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir_263", required=True, help="Directory of <id>.npy (T, 263) inputs.")
    p.add_argument("--out_dir_272", required=True, help="Directory to write <id>.npy (T, 272) outputs.")
    args = p.parse_args()

    src = Path(args.pred_dir_263)
    dst = Path(args.out_dir_272)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.npy"))
    print(f"[+] {len(files)} input files in {src}")

    written = 0
    for f in tqdm(files, ncols=80):
        try:
            m263 = np.load(str(f))
            if m263.ndim != 2 or m263.shape[1] != 263:
                continue
            m272 = hml263_to_h3d272(m263)
            np.save(str(dst / f.name), m272)
            written += 1
        except Exception as e:
            print(f"  [!] {f.name}: {e}")

    print(f"[+] wrote {written} files to {dst}")


if __name__ == "__main__":
    main()
