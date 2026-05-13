#!/usr/bin/env python3
"""Convert HyMotion M2M output (rot6d) to SMPL-X NPZ format for GMR retargeting.

HyMotion M2M output format:
    rot6d:       (B, T, 22, 6)  - 6D rotation for 22 SMPL joints
    transl:      (B, T, 3)      - Root translation
    keypoints3d: (B, T, 52, 3)  - 3D joint positions (optional)
    latent:      (B, T, 198)    - Raw latent (optional)

SMPL-X NPZ format (for GMR):
    pose_body:   (T, 63)        - Body pose in axis-angle (21 joints x 3)
    root_orient: (T, 3)         - Root orientation in axis-angle
    trans:       (T, 3)         - Translation
    betas:       (16,)          - Shape parameters (default zeros)
    gender:      str            - "neutral"
    mocap_frame_rate: int       - FPS (default 30)

Conversion: rot6d -> rotation_matrix -> axis-angle
    Joint 0 (pelvis) -> root_orient
    Joints 1-21      -> pose_body
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
    """Convert rotation matrix to axis-angle representation.

    Args:
        rotmat: (..., 3, 3) rotation matrices
    Returns:
        aa: (..., 3) axis-angle vectors
    """
    from scipy.spatial.transform import Rotation as R

    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)

    # scipy Rotation expects proper rotation matrices
    rot = R.from_matrix(rotmat_flat)
    aa_flat = rot.as_rotvec()

    return aa_flat.reshape(*orig_shape, 3)


def convert_hymotion_to_smplx(
    input_npz: str,
    output_npz: str,
    sample_idx: int = 0,
    fps: int = 30,
):
    """Convert HyMotion NPZ output to SMPL-X NPZ format.

    Args:
        input_npz: Path to HyMotion output NPZ
        output_npz: Path to save SMPL-X NPZ
        sample_idx: Which sample in the batch to convert (default 0)
        fps: Motion frame rate
    """
    data = np.load(input_npz, allow_pickle=True)

    # Extract rot6d and translation
    rot6d = data['rot6d']      # (B, T, 22, 6)
    transl = data['transl']    # (B, T, 3)

    print(f"Input rot6d shape: {rot6d.shape}")
    print(f"Input transl shape: {transl.shape}")

    # Select single sample
    rot6d = rot6d[sample_idx]    # (T, 22, 6)
    transl = transl[sample_idx]  # (T, 3)

    T = rot6d.shape[0]
    print(f"Frames: {T}, Joints: {rot6d.shape[1]}")

    # Convert rot6d -> rotation matrix -> axis-angle
    rotmat = rot6d_to_rotmat(rot6d)        # (T, 22, 3, 3)
    aa = rotmat_to_axis_angle(rotmat)       # (T, 22, 3)

    # Split root and body
    root_orient = aa[:, 0, :]               # (T, 3) - pelvis
    pose_body = aa[:, 1:22, :].reshape(T, -1)  # (T, 63) - 21 body joints

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
    parser = argparse.ArgumentParser(description="Convert HyMotion output to SMPL-X NPZ for GMR")
    parser.add_argument("input", type=str, help="Path to HyMotion output NPZ")
    parser.add_argument("output", type=str, help="Path to save SMPL-X NPZ")
    parser.add_argument("--sample-idx", type=int, default=0, help="Batch sample index")
    parser.add_argument("--fps", type=int, default=30, help="Motion frame rate")
    args = parser.parse_args()

    convert_hymotion_to_smplx(args.input, args.output, args.sample_idx, args.fps)
