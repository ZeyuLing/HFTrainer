#!/usr/bin/env python3
"""Convert HyMotion motion_135 eval output to SMPL-X NPZ format for GMR retargeting.

motion_135 format (from eval scripts):
    motion_135: (T, 135) - [transl(3) + 22*rot6d(132)]
    positions:  (T, 22, 3) - Joint positions (optional)
    translation: (T, 3) - Translation (same as motion_135[:, :3])

The rot6d in motion_135 uses the **same row-major layout** as HyMotion M2M's internal
representation: [R00,R01, R10,R11, R20,R21]. We must reorder [0,2,4,1,3,5] to convert
to column-major before Gram-Schmidt decoding.

SMPL-X NPZ output format (for GMR):
    pose_body:   (T, 63)   - Body pose in axis-angle (21 joints x 3)
    root_orient: (T, 3)    - Root orientation in axis-angle
    trans:       (T, 3)    - Translation
    betas:       (10,)     - Shape parameters (zeros)
    gender:      str       - "neutral"
    mocap_frame_rate: int  - FPS (default 30)
"""
import argparse
import numpy as np
from pathlib import Path


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix.

    HyMotion outputs rot6d in row-major layout: [R00,R01, R10,R11, R20,R21]
    Gram-Schmidt expects column-major layout: [R00,R10,R20, R01,R11,R21]
    We reorder [0,2,4,1,3,5] to convert row-major → column-major before decoding.

    Args:
        rot6d: (..., 6) array of 6D rotation representations (row-major)
    Returns:
        rotmat: (..., 3, 3) array of rotation matrices
    """
    # Row-major → column-major reorder
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    # Normalize first column
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)

    # Second column: Gram-Schmidt orthogonalization
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)

    # Third column: cross product
    b3 = np.cross(b1, b2)

    rotmat = np.stack([b1, b2, b3], axis=-1)
    return rotmat


def rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle representation."""
    from scipy.spatial.transform import Rotation as R

    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    rot = R.from_matrix(rotmat_flat)
    aa_flat = rot.as_rotvec()
    return aa_flat.reshape(*orig_shape, 3)


def convert_motion135_to_smplx(input_npz, output_npz, fps=30):
    """Convert motion_135 NPZ to SMPL-X NPZ format.

    Args:
        input_npz: Path to motion_135 NPZ
        output_npz: Path to save SMPL-X NPZ
        fps: Motion frame rate
    """
    data = np.load(input_npz, allow_pickle=True)

    motion = data['motion_135']  # (T, 135)
    T = motion.shape[0]
    print(f"Input motion_135 shape: {motion.shape}")
    print(f"Frames: {T}")

    # Split: first 3 = translation, rest = 22×6 rot6d
    transl = motion[:, :3]                           # (T, 3)
    rot6d = motion[:, 3:].reshape(T, 22, 6)          # (T, 22, 6)

    # Convert rot6d -> rotation matrix -> axis-angle
    rotmat = rot6d_to_rotmat(rot6d)                   # (T, 22, 3, 3)
    aa = rotmat_to_axis_angle(rotmat)                 # (T, 22, 3)

    # Split root and body
    root_orient = aa[:, 0, :]                         # (T, 3) - pelvis
    pose_body = aa[:, 1:22, :].reshape(T, -1)         # (T, 63) - 21 body joints

    print(f"root_orient shape: {root_orient.shape}")
    print(f"pose_body shape: {pose_body.shape}")
    print(f"transl shape: {transl.shape}")

    # Save as SMPL-X NPZ
    np.savez(
        output_npz,
        pose_body=pose_body.astype(np.float32),
        root_orient=root_orient.astype(np.float32),
        trans=transl.astype(np.float32),
        betas=np.zeros(10, dtype=np.float32),
        gender="neutral",
        mocap_frame_rate=np.array(fps),
    )
    print(f"Saved SMPL-X NPZ to: {output_npz}")

    # Sanity check
    check = np.load(output_npz, allow_pickle=True)
    for k in check.files:
        v = check[k]
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert motion_135 to SMPL-X NPZ for GMR")
    parser.add_argument("input", type=str, help="Path to motion_135 NPZ")
    parser.add_argument("output", type=str, help="Path to save SMPL-X NPZ")
    parser.add_argument("--fps", type=int, default=30, help="Motion frame rate")
    args = parser.parse_args()

    convert_motion135_to_smplx(args.input, args.output, args.fps)
