#!/usr/bin/env python3
"""Headless GMR retargeting: SMPL-X NPZ -> Robot joint positions (PKL).

Runs GMR retargeting without MuJoCo viewer — suitable for headless containers.

Usage:
    python scripts/embodied/gmr_retarget_headless.py \
        --smplx_file /tmp/hymotion_smplx.npz \
        --robot unitree_g1 \
        --save_path /tmp/g1_retarget.pkl

Output PKL dict:
    fps:        float       - Motion frame rate
    root_pos:   (T, 3)      - Root translation
    root_rot:   (T, 4)      - Root rotation (xyzw quaternion)
    dof_pos:    (T, N_dof)  - Joint positions (29 for G1)
"""
import argparse
import os
import sys
import pathlib
import pickle
import time

import numpy as np
import xml.etree.ElementTree as ET

# Add GMR to path
GMR_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent / "ref_repo" / "GMR"
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast


# G1 29-DOF joint order (from MuJoCo kinematic tree depth-first traversal)
G1_JOINT_ORDER = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]

# G1 joint limits (radians) from g1_bm.xml
G1_JOINT_LIMITS = {
    'left_hip_pitch_joint': (-2.5307, 2.8798),
    'left_hip_roll_joint': (-0.5236, 2.9671),
    'left_hip_yaw_joint': (-2.7576, 2.7576),
    'left_knee_joint': (-0.087267, 2.8798),
    'left_ankle_pitch_joint': (-0.87267, 0.5236),
    'left_ankle_roll_joint': (-0.2618, 0.2618),
    'right_hip_pitch_joint': (-2.5307, 2.8798),
    'right_hip_roll_joint': (-2.9671, 0.5236),
    'right_hip_yaw_joint': (-2.7576, 2.7576),
    'right_knee_joint': (-0.087267, 2.8798),
    'right_ankle_pitch_joint': (-0.87267, 0.5236),
    'right_ankle_roll_joint': (-0.2618, 0.2618),
    'waist_yaw_joint': (-2.618, 2.618),
    'waist_roll_joint': (-0.52, 0.52),
    'waist_pitch_joint': (-0.52, 0.52),
    'left_shoulder_pitch_joint': (-3.0892, 2.6704),
    'left_shoulder_roll_joint': (-1.5882, 2.2515),
    'left_shoulder_yaw_joint': (-2.618, 2.618),
    'left_elbow_joint': (-1.0472, 2.0944),
    'left_wrist_roll_joint': (-1.97222, 1.97222),
    'left_wrist_pitch_joint': (-1.61443, 1.61443),
    'left_wrist_yaw_joint': (-1.61443, 1.61443),
    'right_shoulder_pitch_joint': (-3.0892, 2.6704),
    'right_shoulder_roll_joint': (-2.2515, 1.5882),
    'right_shoulder_yaw_joint': (-2.618, 2.618),
    'right_elbow_joint': (-1.0472, 2.0944),
    'right_wrist_roll_joint': (-1.97222, 1.97222),
    'right_wrist_pitch_joint': (-1.61443, 1.61443),
    'right_wrist_yaw_joint': (-1.61443, 1.61443),
}


def clamp_joint_limits(dof_pos, joint_order=G1_JOINT_ORDER, joint_limits=G1_JOINT_LIMITS, soft=True):
    """Clamp joint positions to their mechanical limits.

    GMR's IK solver doesn't always respect joint limits, especially for extreme
    poses. This post-processing step prevents physically impossible configurations.

    Args:
        dof_pos: (T, N_dof) joint positions array
        joint_order: ordered list of joint names matching dof_pos columns
        joint_limits: dict mapping joint_name -> (min_rad, max_rad)
        soft: if True, use smooth tanh-based clamping to avoid discontinuities.
              if False, use hard np.clip (original behavior).

    Returns:
        clamped_dof_pos: (T, N_dof) clamped joint positions
        num_clamped: number of joint-frame pairs that were clamped
    """
    clamped = dof_pos.copy()
    num_clamped = 0
    for i, joint_name in enumerate(joint_order):
        if joint_name in joint_limits:
            lo, hi = joint_limits[joint_name]
            below = clamped[:, i] < lo
            above = clamped[:, i] > hi
            if soft:
                # Soft tanh-based clamping: smoothly approaches limits
                mid = (lo + hi) / 2.0
                half_range = (hi - lo) / 2.0
                # Scale factor controls transition sharpness near limits
                # Higher values = sharper clamp (closer to hard clip)
                scale = 0.9  # uses ~90% of range linearly, smooth saturation near edges
                clamped[:, i] = mid + half_range * np.tanh((clamped[:, i] - mid) / (half_range * scale))
            else:
                clamped[:, i] = np.clip(clamped[:, i], lo, hi)
            num_clamped += np.sum(below) + np.sum(above)
    return clamped, int(num_clamped)


def compute_ground_offset(retarget, smplx_data_frames):
    """Pre-scan all frames to find ground offset (lowest body Z position).

    This mirrors GMR's fbx_offline_to_robot.py::offset_to_ground() — without
    this, scaled SMPL-X targets can have feet below Z=0, causing the IK solver
    to produce a crouched pose with pelvis too low.

    Args:
        retarget: GMR retargeter instance
        smplx_data_frames: list of per-frame SMPL-X data dicts

    Returns:
        float: ground offset (lowest Z across all frames/bodies)
    """
    offset = np.inf
    for frame_data in smplx_data_frames:
        human_data = retarget.to_numpy(frame_data)
        human_data = retarget.scale_human_data(
            human_data, retarget.human_root_name, retarget.human_scale_table
        )
        human_data = retarget.offset_human_data(
            human_data, retarget.pos_offsets1, retarget.rot_offsets1
        )
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if pos[2] < offset:
                offset = pos[2]
    return offset


def main():
    parser = argparse.ArgumentParser(description="Headless GMR retargeting")
    parser.add_argument("--smplx_file", required=True, help="Input SMPL-X NPZ file")
    parser.add_argument("--robot", default="unitree_g1", help="Target robot type")
    parser.add_argument("--save_path", required=True, help="Output PKL file")
    parser.add_argument("--tgt_fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--offset-to-ground", action="store_true", default=False,
                        help="Enable GMR per-frame foot grounding. WARNING: GMR's "
                             "offset_human_data_to_ground treats Z (horizontal in its "
                             "Y-up frame) as the ground axis and collapses the Z "
                             "translation (turns curved paths into back-and-forth lines). "
                             "Keep this OFF; vertical placement is handled by the global "
                             "set_ground_offset. (default: False)")
    parser.add_argument("--no-offset-to-ground", dest="offset_to_ground", action="store_false",
                        help="Disable per-frame foot grounding (default behavior)")
    parser.add_argument("--actual-human-height", type=float, default=None,
                        help="Override auto-detected human height (default: 1.66+0.1*betas[0])")
    parser.add_argument("--posture-cost", type=float, default=20.0,
                        help="IK temporal-consistency regularizer (posture target = "
                             "previous frame). ~57%% less joint-accel jitter, trajectory "
                             "preserved. 0 disables. (default: 20.0)")
    args = parser.parse_args()

    SMPLX_FOLDER = GMR_ROOT / "assets" / "body_models"

    print(f"Loading SMPL-X from: {args.smplx_file}")
    smplx_data, body_model, smplx_output, auto_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    actual_human_height = auto_human_height
    if args.actual_human_height is not None:
        actual_human_height = args.actual_human_height
        print(f"  Human height override: {actual_human_height:.3f}m (auto-detected was: {auto_human_height:.3f}m)")
    print(f"  Human height used: {actual_human_height:.3f}m")
    print(f"  Frames: {smplx_data['pose_body'].shape[0]}")

    # Align FPS
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps
    )
    print(f"  Aligned FPS: {aligned_fps}, Frames after alignment: {len(smplx_data_frames)}")

    # Initialize retargeting
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        posture_cost=args.posture_cost,
    )
    print(f"  Retargeting to: {args.robot}")

    # Pre-compute ground offset (like GMR's fbx_offline_to_robot.py)
    # This finds the lowest body Z across all frames and sets it as ground
    # reference, preventing the IK solver from producing crouched poses.
    print(f"  Computing ground offset...")
    ground_offset = compute_ground_offset(retarget, smplx_data_frames)
    retarget.set_ground_offset(ground_offset)
    print(f"  Ground offset: {ground_offset:.4f}")
    print(f"  offset_to_ground (per-frame foot grounding): {args.offset_to_ground}")

    # Run retargeting frame by frame
    qpos_list = []
    t0 = time.time()
    for i, frame_data in enumerate(smplx_data_frames):
        qpos = retarget.retarget(frame_data, offset_to_ground=args.offset_to_ground)
        qpos_list.append(qpos)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"  Frame {i+1}/{len(smplx_data_frames)} ({fps:.1f} fps)")

    elapsed = time.time() - t0
    print(f"  Retargeting done: {len(qpos_list)} frames in {elapsed:.2f}s ({len(qpos_list)/elapsed:.1f} fps)")

    # Save output
    root_pos = np.array([q[:3] for q in qpos_list])
    # GMR outputs wxyz quaternion, convert to xyzw for ProtoMotions compatibility
    root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])
    dof_pos = np.array([q[7:] for q in qpos_list])

    # Clamp joint positions to mechanical limits (soft clamping for smooth transitions)
    dof_pos, num_clamped = clamp_joint_limits(dof_pos, soft=True)
    total_entries = dof_pos.shape[0] * dof_pos.shape[1]
    print(f"  Joint limit clamping (soft): {num_clamped}/{total_entries} values clamped ({100*num_clamped/total_entries:.1f}%)")

    # Temporal smoothing of dof_pos to reduce IK solver frame-to-frame jitter
    from scipy.signal import savgol_filter
    T = dof_pos.shape[0]
    sg_win = min(7, T if T % 2 == 1 else T - 1)  # ~0.23s at 30Hz
    if sg_win >= 5:
        dof_pos = savgol_filter(dof_pos, window_length=sg_win, polyorder=3, axis=0)
        print(f"  Temporal smoothing: Savitzky-Golay (window={sg_win}, poly=3) on dof_pos")
    else:
        print(f"  Temporal smoothing: skipped (too few frames: {T})")

    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,       # (T, 3)
        "root_rot": root_rot,       # (T, 4) xyzw
        "dof_pos": dof_pos,         # (T, 29) for G1
        "local_body_pos": None,
        "link_body_list": None,
    }

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(motion_data, f)

    print(f"\nSaved to: {args.save_path}")
    print(f"  root_pos: {root_pos.shape}, range: [{root_pos.min():.3f}, {root_pos.max():.3f}]")
    print(f"  root_rot: {root_rot.shape}")
    print(f"  dof_pos:  {dof_pos.shape}, range: [{dof_pos.min():.3f}, {dof_pos.max():.3f}]")


if __name__ == "__main__":
    main()
