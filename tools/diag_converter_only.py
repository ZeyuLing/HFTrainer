#!/usr/bin/env python3
"""Diagnose convert_momask263_to_h3d272 in isolation.

Bypasses ``process_file`` (uniform_skeleton + face_z + canonical bone scale),
``lerp``, and ``slerp``: synthesize a 263 representation FROM a known SMPL-style
motion (positions + per-joint local rotations), then run the converter on that
263 and re-encode to 272.  Compare with the *expected* 272 obtained by feeding
the same (positions, rotations) directly into ``encode_h3d272``.

Differences here can only come from the converter's decoder + encoder logic
(half-angle yaw integration, root rotation reconstruction, root xz delta
projection, etc.).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "ref_repo" / "Momask" / "momask-codes"))

from convert_momask263_to_h3d272 import decode_263_to_pose, encode_h3d272  # noqa: E402
from common.quaternion import quaternion_to_cont6d_np, qrot_np  # noqa: E402


def _R_y(theta):
    """Rotation matrix R_y(theta) using the same convention as encode_h3d272."""
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = c
    R[0, 2] = s
    R[1, 1] = 1.0
    R[2, 0] = -s
    R[2, 2] = c
    return R


def _quat_y(theta):
    """Quaternion (w, x, y, z) representing R_y(theta) using half-angle convention."""
    return np.array([np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0])


def synthesize_263(positions: np.ndarray, R_local: np.ndarray, root_yaw: np.ndarray) -> np.ndarray:
    """Build a HumanML3D-263 vector from full SMPL-style motion.

    Args:
        positions: (T, 22, 3) global joint positions.
        R_local: (T, 22, 3, 3) parent-relative local rotations (root slot is unused
                 since 263 only stores yaw via rot_velocity).
        root_yaw: (T,) global root yaw in radians.

    Returns:
        m263: (T-1, 263) packed in the canonical-frame format described in
              MoMask's ``process_file``.
    """
    T = positions.shape[0]
    assert R_local.shape == (T, 22, 3, 3)
    assert root_yaw.shape == (T,)

    # rotate positions into per-frame canonical frame (yaw removed)
    pos_canon = np.zeros_like(positions)
    for t in range(T):
        pos_canon[t] = (_R_y(-root_yaw[t]) @ positions[t].T).T  # (22, 3)

    # subtract root xz at every frame
    pos_local = pos_canon.copy()
    pos_local[:, :, 0] -= pos_local[:, 0:1, 0]
    pos_local[:, :, 2] -= pos_local[:, 0:1, 2]

    # ric_data: positions[1:, 1:22, :].reshape(T-1, 63)
    ric_data = pos_local[1:, 1:22, :].reshape(T - 1, 63)

    # rot_data: 6D representation of non-root local rotations (column-major)
    rot_data = np.zeros((T - 1, 21, 6))
    for t in range(1, T):
        for j in range(21):
            R = R_local[t, j + 1]  # joint j+1 (skip root)
            rot_data[t - 1, j, 0:3] = R[:, 0]  # col 0
            rot_data[t - 1, j, 3:6] = R[:, 1]  # col 1
    rot_data = rot_data.reshape(T - 1, 21 * 6)

    # local velocities in canonical frame
    canonical_vel = pos_canon[1:] - pos_canon[:-1]  # (T-1, 22, 3) global velocity
    # rotate by R_y(-yaw[t]) is already done above (we rotated positions). but
    # we still need to apply R_y(-yaw[t]) to the global velocities.  Easier:
    # local_vel_t = pos_canon[t+1] - pos_canon[t] BEFORE subtracting root xz.
    local_vel = canonical_vel.reshape(T - 1, 22 * 3)

    # root angular velocity = arcsin(quat_y) of frame-to-frame yaw delta
    yaw_delta = root_yaw[1:] - root_yaw[:-1]
    rot_velocity = np.sin(yaw_delta / 2.0)  # = quat[2] of pure-y frame delta
    rot_velocity = np.arcsin(rot_velocity)  # = yaw_delta / 2.0 (half-angle)
    # → cumsum(rot_velocity) = yaw_total / 2 (matches MoMask convention)

    # root xz velocity in canonical frame (i.e., rotate global delta by R_y(-yaw[t]))
    root_xz_global_delta = positions[1:, 0, [0, 2]] - positions[:-1, 0, [0, 2]]
    cs = np.cos(root_yaw[:-1])
    sn = np.sin(root_yaw[:-1])
    l_velocity = np.stack(
        [cs * root_xz_global_delta[:, 0] - sn * root_xz_global_delta[:, 1],
         sn * root_xz_global_delta[:, 0] + cs * root_xz_global_delta[:, 1]],
        axis=-1,
    )

    # root y (height)
    root_y = positions[:-1, 0, 1:2]  # (T-1, 1)

    # foot contacts: zeros (not used by decoder)
    feet = np.zeros((T - 1, 4))

    root_data = np.concatenate([rot_velocity[:, None], l_velocity, root_y], axis=-1)  # (T-1, 4)
    m263 = np.concatenate([root_data, ric_data, rot_data, local_vel, feet], axis=-1)
    assert m263.shape[1] == 263, m263.shape
    return m263


def main():
    # Synthesize a simple but non-trivial motion: T frames, root walks +x with
    # increasing yaw, body slightly bobbing up/down, joints fixed at canonical
    # offsets.
    T = 60
    yaw = np.linspace(0.0, 1.5, T)  # 1.5 rad over 60 frames
    root_pos = np.zeros((T, 3))
    root_pos[:, 0] = np.linspace(0, 2.0, T)
    root_pos[:, 1] = 1.0 + 0.05 * np.sin(np.linspace(0, 4 * np.pi, T))
    root_pos[:, 2] = np.linspace(0, 1.0, T)

    # Non-root joint positions: arrange in a line behind root in CANONICAL frame,
    # then rotate to global frame using R_y(yaw).
    canonical_offsets = np.zeros((22, 3))
    for j in range(1, 22):
        canonical_offsets[j, 0] = 0.1 * j  # spread along x
        canonical_offsets[j, 1] = -0.05 * j  # decreasing height
        canonical_offsets[j, 2] = 0.0

    positions = np.zeros((T, 22, 3))
    for t in range(T):
        rotated = (_R_y(yaw[t]) @ canonical_offsets.T).T
        positions[t] = rotated + root_pos[t]
        positions[t, 0] = root_pos[t]  # ensure root at root_pos

    # Local rotations: identity for non-root (so ric_data is sufficient to encode pose),
    # and root global = R_y(yaw[t]).
    R_local = np.zeros((T, 22, 3, 3))
    for t in range(T):
        R_local[t, 0] = _R_y(yaw[t])
        for j in range(1, 22):
            R_local[t, j] = np.eye(3)

    # Path A: directly encode (positions, R_local) -> expected_272
    expected_272 = encode_h3d272(positions, R_local)

    # Path B: synthesize 263 from same (positions, R_local, yaw),
    # then run converter -> got_272
    m263 = synthesize_263(positions, R_local, yaw)
    pos_dec, R_dec, _ = decode_263_to_pose(m263)
    got_272 = encode_h3d272(pos_dec, R_dec)

    # Sanity: lengths should match (m263 has T-1 frames)
    print(f"expected_272 shape={expected_272.shape}, got_272 shape={got_272.shape}")
    Tmin = min(len(expected_272), len(got_272))
    diff = expected_272[:Tmin] - got_272[:Tmin]

    blocks = [
        ("root_xz_vel  [0:2]", 0, 2),
        ("heading_d_6d [2:8]", 2, 8),
        ("joints_pos  [8:74]", 8, 74),
        ("joints_vel  [74:140]", 74, 140),
        ("joints_rot  [140:272]", 140, 272),
    ]
    print(f"{'block':25s} | {'max_abs':>10s} {'mean_abs':>10s} {'rms':>10s}")
    print("-" * 70)
    for name, a, b in blocks:
        sl = diff[:, a:b]
        print(f"{name:25s} | {np.max(np.abs(sl)):10.6f} {np.mean(np.abs(sl)):10.6f} "
              f"{np.sqrt(np.mean(sl**2)):10.6f}")

    # Specifically inspect joint-0 position recovery (decoder vs ground-truth positions
    # in CANONICAL frame, which is what encode_h3d272 ultimately uses).
    # Recompute pos_canon for original positions as the reference:
    pos_canon_ref = np.zeros_like(positions)
    for t in range(T):
        pos_canon_ref[t] = (_R_y(-yaw[t]) @ positions[t].T).T

    # And the decoder output is positions in GLOBAL frame; rotate it to canonical.
    pos_canon_got = np.zeros_like(pos_dec)
    # Decoder's yaw integration:
    #   half_yaw[1:] = cumsum(rot_vel[:-1]); yaw_t = 2 * half_yaw
    rot_vel = m263[..., 0]
    half_yaw = np.zeros_like(rot_vel)
    half_yaw[1:] = np.cumsum(rot_vel[:-1])
    yaw_dec = 2.0 * half_yaw
    for t in range(len(pos_dec)):
        pos_canon_got[t] = (_R_y(-yaw_dec[t]) @ pos_dec[t].T).T

    Tcanon = min(len(pos_canon_ref) - 1, len(pos_canon_got))  # m263 dropped frame 0
    pos_canon_ref_aligned = pos_canon_ref[1:1 + Tcanon]  # m263 starts from frame 1
    pos_canon_got_aligned = pos_canon_got[:Tcanon]
    diff_canon = pos_canon_ref_aligned - pos_canon_got_aligned
    print(f"\ncanonical-frame joint position diff (decoder global -> canonical):")
    print(f"  max_abs={np.max(np.abs(diff_canon)):.6f}, mean_abs={np.mean(np.abs(diff_canon)):.6f}")

    # Compare yaw recovery
    print(f"\nyaw recovery: ref[1:5]={yaw[1:5]}, dec[0:4]={yaw_dec[0:4]}")
    print(f"  max yaw diff={np.max(np.abs(yaw[1:1+len(yaw_dec)] - yaw_dec)):.6f}")


if __name__ == "__main__":
    main()
