#!/usr/bin/env python3
"""Batch-convert motion_135 NPZ files to SMPL joint positions JSON for web visualization.

For each NPZ file in --npz-dir, this script:
  1. Loads motion_135 (T, 135) — [transl(3) + 22*rot6d(132)]
  2. Converts rot6d (row-major) -> axis-angle via [0,2,4,1,3,5] reorder + Gram-Schmidt
  3. Runs SmplxLite FK to compute 22 world-space joint positions per frame
  4. Saves as JSON: { fps, num_frames, joint_names, frames: [{joints: [[x,y,z]*22]}] }

Output JSON is consumable by Three.js for SMPL skeleton visualization.

Usage:
    python3 scripts/embodied/batch_npz_to_smpl_joints.py \
        --npz-dir output/embodied_t2m_v4/data/npz \
        --output-dir output/embodied_t2m_v4/data/smpl_joints
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hftrainer.models.motion.components.body_models.smplx_lite import SmplxLite


# SMPL-X 22 joint names
JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
]

# SMPL-X 22-joint kinematic tree (parent indices, -1 for root)
JOINT_PARENTS = [
    -1,  # 0 Pelvis
    0,   # 1 L_Hip
    0,   # 2 R_Hip
    0,   # 3 Spine1
    1,   # 4 L_Knee
    2,   # 5 R_Knee
    3,   # 6 Spine2
    4,   # 7 L_Ankle
    5,   # 8 R_Ankle
    6,   # 9 Spine3
    7,   # 10 L_Foot
    8,   # 11 R_Foot
    9,   # 12 Neck
    9,   # 13 L_Collar
    9,   # 14 R_Collar
    12,  # 15 Head
    13,  # 16 L_Shoulder
    14,  # 17 R_Shoulder
    16,  # 18 L_Elbow
    17,  # 19 R_Elbow
    18,  # 20 L_Wrist
    19,  # 21 R_Wrist
]

# Bones for visualization (pairs of joint indices)
BONES = [
    [0, 1], [0, 2], [0, 3],       # Pelvis -> hips, spine
    [1, 4], [2, 5],               # Hips -> knees
    [4, 7], [5, 8],               # Knees -> ankles
    [7, 10], [8, 11],             # Ankles -> feet
    [3, 6], [6, 9],               # Spine chain
    [9, 12], [9, 13], [9, 14],    # Spine3 -> neck, collars
    [12, 15],                     # Neck -> head
    [13, 16], [14, 17],           # Collars -> shoulders
    [16, 18], [17, 19],           # Shoulders -> elbows
    [18, 20], [19, 21],           # Elbows -> wrists
]


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


def convert_single_npz(npz_path: str, model: SmplxLite, device: torch.device) -> dict:
    """Convert a single motion_135 NPZ to joint positions dict.

    Returns:
        dict with keys: fps, num_frames, joint_names, joint_parents, bones, frames
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

    global_orient = aa[:, 0, :]                # (T, 3) - pelvis
    body_pose = aa[:, 1:22, :].reshape(T, -1)  # (T, 63) - 21 body joints

    # Run FK with SmplxLite
    transl_t = torch.from_numpy(transl).float().unsqueeze(0).to(device)       # (1, T, 3)
    global_orient_t = torch.from_numpy(global_orient).float().unsqueeze(0).to(device)  # (1, T, 3)
    body_pose_t = torch.from_numpy(body_pose).float().unsqueeze(0).to(device)  # (1, T, 63)

    with torch.no_grad():
        joints, _, _ = model.fk(
            transl=transl_t,
            global_orient=global_orient_t,
            body_pose=body_pose_t,
        )  # (1, T, 22, 3)

    joints_np = joints.squeeze(0).cpu().numpy()  # (T, 22, 3)

    # Build JSON structure
    frames = []
    for t in range(T):
        frame_joints = joints_np[t].tolist()  # [[x,y,z] * 22]
        frames.append({"joints": frame_joints})

    return {
        "fps": fps,
        "num_frames": T,
        "joint_names": JOINT_NAMES,
        "joint_parents": JOINT_PARENTS,
        "bones": BONES,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch convert motion_135 NPZ to SMPL joint positions JSON")
    parser.add_argument("--npz-dir", type=str, required=True, help="Directory of motion_135 NPZ files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for JSON files")
    parser.add_argument("--model-path", type=str, default="checkpoints/smpl_models/smplx", help="SMPL-X model path")
    parser.add_argument("--gender", type=str, default="neutral", help="SMPL-X gender")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already converted files")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all NPZ files
    npz_files = sorted([f for f in npz_dir.iterdir() if f.suffix == '.npz'])
    print(f"Found {len(npz_files)} NPZ files in {npz_dir}")

    if not npz_files:
        print("No NPZ files found, exiting.")
        return

    # Load SMPL-X model
    device = torch.device(args.device)
    print(f"Loading SmplxLite from {args.model_path} (gender={args.gender})...")
    model = SmplxLite(model_path=args.model_path, gender=args.gender).to(device)
    model.eval()
    print("Model loaded.")

    # Process each NPZ
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
            result = convert_single_npz(str(npz_path), model, device)

            # Save JSON (compact format to reduce file size)
            with open(json_path, 'w') as f:
                json.dump(result, f, separators=(',', ':'))

            file_size = json_path.stat().st_size
            print(f"  [{success+1}/{len(npz_files)}] {stem}: {result['num_frames']} frames @ {result['fps']}fps -> {file_size/1024:.1f}KB")
            success += 1

        except Exception as e:
            print(f"  FAILED {stem}: {e}")
            failed += 1

    print(f"\nDone: {success} converted, {failed} failed, {skipped} skipped")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
