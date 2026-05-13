#!/usr/bin/env python3
"""Batch-convert motion_135 NPZ files to SMPL mesh-ready JSON for web visualization.

For each NPZ file, produces a JSON consumable by load_smpl.js SkinnedMesh renderer.
Output format matches what score_m2m's /api/smpl returns:

  {
    "type": "frames",
    "fps": 30,
    "frames": [
      [{                           // frame 0 — array of 1 person
        "id": 0,
        "gender": "neutral",
        "smpl_type": "smplx",      // or "smplh"
        "Rh": [[rx, ry, rz]],      // 1×3 root orientation (axis-angle)
        "Th": [[tx, ty, tz]],      // 1×3 translation
        "poses": [[p0, p1, ...]],  // 1×N body joint axis-angles (flattened)
        "shapes": [[0,...,0]],      // 1×16 shape coefficients
        "mocap_framerate": 30,
      }],
      ...
    ]
  }

This produces FULL SMPL MESH rendering (not skeleton-only).

Usage:
    python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
        --npz-dir output/embodied_t2m_v4/data/npz \
        --output-dir output/embodied_t2m_v4/data/smpl_mesh

    # Single file
    python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
        --npz-file data/embodied_debug/v6_e2e_test/npz/wave_hand.npz \
        --output-dir data/embodied_debug/v6_e2e_test/smpl_mesh
"""
import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path


def rot6d_to_axis_angle_np(rot6d: np.ndarray) -> np.ndarray:
    """Convert row-major rot6d (..., 6) to axis-angle (..., 3).

    HyMotion stores rot6d in row-major: [R00,R01, R10,R11, R20,R21].
    Must reorder [0,2,4,1,3,5] to column-major before Gram-Schmidt.
    """
    from scipy.spatial.transform import Rotation as R

    # Row-major -> column-major reorder
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    # Gram-Schmidt orthogonalization
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)

    rotmat = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)

    # Rotation matrix -> axis-angle
    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    aa_flat = R.from_matrix(rotmat_flat).as_rotvec()
    return aa_flat.reshape(*orig_shape, 3).astype(np.float32)


def convert_single_npz(npz_path: str, smpl_type: str = "smplx",
                        gender: str = "neutral") -> dict:
    """Convert a single motion_135 NPZ to SMPL mesh JSON format.

    The motion_135 format: [transl(3) + 22*rot6d(132)] per frame.
    SMPL-X uses 55 joints (22 body + 3 face + 30 hands), but motion_135
    only has 22 body joints. We zero-pad the rest.

    Returns:
        dict with keys: type, fps, frames
    """
    data = np.load(npz_path, allow_pickle=True)
    motion = data['motion_135']  # (T, 135)
    fps = int(data.get('fps', 30))
    T = motion.shape[0]

    # Split: first 3 = translation, rest = 22*6 rot6d
    transl = motion[:, :3]                    # (T, 3)
    rot6d = motion[:, 3:].reshape(T, 22, 6)   # (T, 22, 6)

    # Convert rot6d -> axis-angle
    aa = rot6d_to_axis_angle_np(rot6d)         # (T, 22, 3)

    # Root orientation (joint 0) and body pose (joints 1-21)
    root_orient = aa[:, 0, :]                  # (T, 3)
    body_pose = aa[:, 1:22, :]                 # (T, 21, 3)

    # Build per-frame poses array matching SMPL type:
    # - SMPL: 24 joints * 3 = 72 params (root 3 + body 69)
    # - SMPL+H: 52 joints * 3 = 156 params (root 3 + body 69 + hands 84)
    # - SMPL-X: 55 joints * 3 = 165 params (root 3 + body 63 + jaw 3 + eyes 6 + hands 90)
    #
    # motion_135 has 22 joints (1 root + 21 body).
    # SMPL body joints = 23 (indices 1-23), so 21 body joints from motion_135
    # maps to SMPL joints 1-21 with joints 22-23 zero-padded.

    if smpl_type == "smplx":
        # SMPL-X: 55 joints total
        # [root(3) + body(21*3=63) + jaw(3) + leye(3) + reye(3) + lhand(15*3=45) + rhand(15*3=45)]
        # = 3 + 63 + 3 + 6 + 90 = 165
        # We only have 21 body joints (63 params), rest are zeros
        poses_per_frame = np.zeros((T, 165), dtype=np.float32)
        poses_per_frame[:, :3] = root_orient        # root
        poses_per_frame[:, 3:66] = body_pose.reshape(T, 63)  # 21 body joints
        # jaw, eyes, hands all zero
    elif smpl_type == "smplh":
        # SMPL+H: 52 joints total
        # [root(3) + body(21*3=63) + lhand(15*3=45) + rhand(15*3=45)]
        # = 3 + 63 + 90 = 156
        poses_per_frame = np.zeros((T, 156), dtype=np.float32)
        poses_per_frame[:, :3] = root_orient
        poses_per_frame[:, 3:66] = body_pose.reshape(T, 63)
    else:
        # SMPL: 24 joints total
        # [root(3) + body(23*3=69)]
        # = 72
        poses_per_frame = np.zeros((T, 72), dtype=np.float32)
        poses_per_frame[:, :3] = root_orient
        poses_per_frame[:, 3:66] = body_pose.reshape(T, 63)

    # Shape coefficients (all zeros - we don't have beta params)
    shapes = [[0.0] * 16]

    # Build frames list
    # The frontend score.html updateFrame expects:
    #   frames[t] = [ { id, gender, smpl_type, Rh, Th, poses, shapes, mocap_framerate } ]
    # Where:
    #   Rh: [1, 3] root orientation (axis-angle)
    #   Th: [1, 3] translation
    #   poses: [1, N] full poses (including root) flattened axis-angle
    #     If N == 69 (SMPL body-only), updateFrame uses poses_offset = -3
    #     Otherwise, bone[0] = root gets poses[0:3], bone[1] = poses[3:6], etc.

    frames = []
    for t in range(T):
        frame = [{
            "id": 0,
            "gender": gender,
            "smpl_type": smpl_type,
            "Rh": [root_orient[t].tolist()],
            "Th": [transl[t].tolist()],
            "poses": [poses_per_frame[t].tolist()],
            "shapes": shapes,
            "mocap_framerate": fps,
        }]
        frames.append(frame)

    return {
        "type": "frames",
        "fps": fps,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert motion_135 NPZ to SMPL mesh JSON for web visualization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz-dir", type=str, help="Directory of motion_135 NPZ files")
    group.add_argument("--npz-file", type=str, help="Single NPZ file to convert")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for JSON files")
    parser.add_argument("--smpl-type", type=str, default="smplh",
                        choices=["smpl", "smplh", "smplx"],
                        help="SMPL model type (default: smplh — best web asset support)")
    parser.add_argument("--gender", type=str, default="neutral",
                        choices=["neutral", "male", "female"],
                        help="SMPL gender (default: neutral)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already converted files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect NPZ files
    if args.npz_file:
        npz_files = [Path(args.npz_file)]
    else:
        npz_dir = Path(args.npz_dir)
        npz_files = sorted([f for f in npz_dir.iterdir() if f.suffix == '.npz'])

    print(f"Found {len(npz_files)} NPZ files to process")
    print(f"SMPL type: {args.smpl_type}, gender: {args.gender}")

    if not npz_files:
        print("No NPZ files found, exiting.")
        return

    success = 0
    failed = 0
    skipped = 0

    for npz_path in npz_files:
        stem = npz_path.stem
        json_path = output_dir / f"{stem}.json"

        if args.skip_existing and json_path.exists():
            skipped += 1
            continue

        try:
            result = convert_single_npz(str(npz_path), smpl_type=args.smpl_type,
                                         gender=args.gender)

            # Save JSON (compact format to reduce file size)
            with open(json_path, 'w') as f:
                json.dump(result, f, separators=(',', ':'))

            file_size = json_path.stat().st_size
            n_frames = len(result["frames"])
            print(f"  [{success+1}/{len(npz_files)}] {stem}: {n_frames} frames "
                  f"@ {result['fps']}fps -> {file_size/1024:.1f}KB")
            success += 1

        except Exception as e:
            import traceback
            print(f"  FAILED {stem}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nDone: {success} converted, {failed} failed, {skipped} skipped")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
