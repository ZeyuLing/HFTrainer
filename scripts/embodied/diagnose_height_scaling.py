#!/usr/bin/env python3
"""Diagnose GMR height scaling issue.

Tests different actual_human_height values to find the one that produces
a pelvis height close to G1's nominal standing height (0.796m).

Also checks whether the input SMPL-X motion has bent legs.
"""
import argparse
import pathlib
import sys
import numpy as np

GMR_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent / "ref_repo" / "GMR"
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", required=True)
    args = parser.parse_args()

    SMPLX_FOLDER = GMR_ROOT / "assets" / "body_models"

    # Load SMPL-X
    smplx_data, body_model, smplx_output, auto_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    print(f"=== SMPL-X Diagnostics ===")
    print(f"Auto-detected height: {auto_height:.3f}m  (formula: 1.66 + 0.1*betas[0])")
    betas = smplx_data["betas"]
    if len(betas.shape) == 1:
        print(f"betas[0:5]: {betas[:5]}")
    else:
        print(f"betas[0, 0:5]: {betas[0, :5]}")

    # Check SMPL-X joint positions at frame 0
    joints_f0 = smplx_output.joints[0].detach().numpy()
    print(f"\nFrame 0 SMPL-X joint positions (Y-up):")
    print(f"  pelvis (joint 0):     {joints_f0[0]}")
    print(f"  left_hip (joint 1):   {joints_f0[1]}")
    print(f"  right_hip (joint 2):  {joints_f0[2]}")
    print(f"  left_knee (joint 4):  {joints_f0[4]}")
    print(f"  right_knee (joint 5): {joints_f0[5]}")
    print(f"  left_ankle (joint 7): {joints_f0[7]}")
    print(f"  right_ankle (joint 8):{joints_f0[8]}")

    # Estimate actual height from joint positions
    # Height from foot to head
    head_y = joints_f0[15, 1] if joints_f0.shape[0] > 15 else joints_f0[12, 1]
    foot_y = min(joints_f0[7, 1], joints_f0[8, 1])  # ankle Y
    joint_height = head_y - foot_y
    print(f"\n  Head-to-ankle joint height: {joint_height:.3f}m")
    print(f"  Pelvis Y (height above origin): {joints_f0[0, 1]:.3f}m")
    print(f"  trans[0]: {smplx_data['trans'][0]}")

    # Check if legs are bent at frame 0
    pose_body = smplx_data["pose_body"]  # (T, 63)
    print(f"\nFrame 0 body pose (axis-angle, 63 dims = 21 joints * 3):")
    print(f"  pose_body shape: {pose_body.shape}")
    # Joint indices in pose_body: 0=L_Hip, 1=R_Hip, 2=Spine1, 3=L_Knee, 4=R_Knee, 5=Spine2
    for j_idx, j_name in [(0, "L_Hip"), (1, "R_Hip"), (3, "L_Knee"), (4, "R_Knee"),
                           (6, "L_Ankle"), (7, "R_Ankle")]:
        aa = pose_body[0, j_idx*3:(j_idx+1)*3]
        angle = np.linalg.norm(aa)
        print(f"  {j_name}: axis-angle={aa}, angle={np.degrees(angle):.2f}°")

    # Get aligned frames
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    print(f"\nAligned: {len(smplx_data_frames)} frames at {aligned_fps} fps")

    # Test different height values
    heights_to_test = [auto_height, 1.8, 2.0, 2.1, 2.2]
    print(f"\n=== Testing different actual_human_height values ===")
    print(f"{'Height':>8} | {'Scale':>6} | {'Pelvis Z':>10} | {'L_Knee DOF':>12} | {'R_Knee DOF':>12} | {'L_Ankle DOF':>12}")
    print("-" * 80)

    for test_height in heights_to_test:
        retarget = GMR(
            actual_human_height=test_height,
            src_human="smplx",
            tgt_robot="unitree_g1",
            verbose=False,
        )
        # Compute ground offset
        offset = np.inf
        for frame_data in smplx_data_frames[:10]:  # only first 10 frames for speed
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
        retarget.set_ground_offset(offset)

        # Retarget frame 0
        qpos = retarget.retarget(smplx_data_frames[0], offset_to_ground=True)
        root_pos = qpos[:3]
        root_rot = qpos[3:7]
        dof_pos = qpos[7:]

        # root_pos is in MuJoCo Z-up frame
        pelvis_z = root_pos[2]
        scale = retarget.human_scale_table["pelvis"]

        # G1 DOF order: l_hip_pitch, l_hip_roll, l_hip_yaw, l_knee, l_ankle_pitch, l_ankle_roll, ...
        l_knee = dof_pos[3]
        r_knee = dof_pos[9]
        l_ankle = dof_pos[4]

        print(f"{test_height:>8.2f} | {scale:>6.3f} | {pelvis_z:>10.4f} | {l_knee:>12.4f} | {r_knee:>12.4f} | {l_ankle:>12.4f}")

    # Also test with no scaling at all (actual_human_height = human_height_assumption / base_scale)
    # To get scale=1.0 for pelvis: 0.9 * ratio = 1.0 → ratio = 1/0.9 → height = 1.8/0.9 = 2.0
    print(f"\nTarget: pelvis_z ≈ 0.796m, knee DOF ≈ 0.005-0.01 (nearly straight)")
    print(f"Reference motion stats: pelvis_z=0.796, l_knee=0.005, r_knee=0.011")


if __name__ == "__main__":
    main()
