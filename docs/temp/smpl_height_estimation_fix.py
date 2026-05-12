"""
Height Estimation Fix for load_smplx_file()

This demonstrates the corrected function that estimates human height from 
FK-computed joint positions instead of relying on betas[0] (which is always 0).

Key joint indices in SMPL-X:
  - Joint 15: head
  - Joint 10: left_foot
  - Joint 11: right_foot
  - Coordinate system: Y-axis is vertical
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def estimate_human_height_from_joints(joints_world, frame_indices=None, 
                                      head_joint_idx=15, 
                                      foot_joint_indices=(10, 11)):
    """
    Estimate human height from world-space joint positions.
    
    Args:
        joints_world: (num_frames, num_joints, 3) world-space joint positions from SMPL-X FK
        frame_indices: which frames to use for height estimation (default: use all frames)
        head_joint_idx: index of head joint (default: 15 in SMPL-X)
        foot_joint_indices: indices of foot joints (default: (10, 11) in SMPL-X)
    
    Returns:
        human_height: estimated height in meters
        frame_heights: (num_frames,) array of per-frame height estimates (for diagnostics)
    """
    if frame_indices is None:
        frame_indices = slice(None)
    
    joints_subset = joints_world[frame_indices]  # (num_frames_to_use, num_joints, 3)
    
    # Extract Y coordinates (vertical axis)
    head_y = joints_subset[:, head_joint_idx, 1]  # (num_frames_to_use,)
    
    # Get minimum Y from both feet
    foot_y = joints_subset[:, list(foot_joint_indices), 1]  # (num_frames_to_use, 2)
    min_foot_y = np.min(foot_y, axis=1)  # (num_frames_to_use,)
    
    # Height per frame
    frame_heights = head_y - min_foot_y  # (num_frames_to_use,)
    
    # Use median to be robust to outliers
    human_height = np.median(frame_heights)
    
    return human_height, frame_heights


def load_smplx_file_fixed(smplx_file, smplx_body_model_path):
    """
    Fixed version of load_smplx_file() with FK-based height estimation.
    
    This replaces the hardcoded formula:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    
    With actual measurement from FK-computed joint positions.
    """
    import smplx
    
    smplx_data = np.load(smplx_file, allow_pickle=True)
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
    )
    
    num_frames = smplx_data["pose_body"].shape[0]
    betas_raw = torch.tensor(smplx_data["betas"]).float().view(1, -1)
    
    # Truncate/pad betas to match model's expected num_betas (default 10)
    num_betas = body_model.num_betas if hasattr(body_model, 'num_betas') else 10
    if betas_raw.shape[-1] > num_betas:
        betas_raw = betas_raw[..., :num_betas]
    elif betas_raw.shape[-1] < num_betas:
        betas_raw = torch.cat([betas_raw, torch.zeros(1, num_betas - betas_raw.shape[-1])], dim=-1)
    betas_tensor = betas_raw.expand(num_frames, -1)
    
    # Forward kinematics
    smplx_output = body_model(
        betas=betas_tensor,
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),  # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),  # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(),  # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    
    # ====== FIX: Estimate height from FK joint positions ======
    joints_world = smplx_output.joints.detach().numpy()  # (num_frames, num_joints, 3)
    
    # Use middle 50% of frames for height estimation (skip start/end frames which may be noisy)
    start_frame = num_frames // 4
    end_frame = 3 * num_frames // 4
    frame_indices = slice(start_frame, end_frame)
    
    human_height, frame_heights = estimate_human_height_from_joints(
        joints_world, 
        frame_indices=frame_indices,
        head_joint_idx=15,
        foot_joint_indices=(10, 11)
    )
    
    # Clamp to reasonable human height range [1.4m, 2.2m]
    human_height = max(1.4, min(2.2, human_height))
    
    print(f"[load_smplx_file] Height estimation:")
    print(f"  Estimated height: {human_height:.3f} m")
    print(f"  Frame height stats: min={np.min(frame_heights):.3f}m, max={np.max(frame_heights):.3f}m, "
          f"median={np.median(frame_heights):.3f}m, std={np.std(frame_heights):.3f}m")
    print(f"  Used frames: {start_frame}-{end_frame} (middle 50%)")
    
    # ========================================================
    
    return smplx_data, body_model, smplx_output, human_height


def load_gvhmr_pred_file_fixed(gvhmr_pred_file, smplx_body_model_path):
    """
    Fixed version of load_gvhmr_pred_file() with FK-based height estimation.
    
    Same fix applied as in load_smplx_file_fixed().
    """
    import smplx
    
    gvhmr_pred = torch.load(gvhmr_pred_file)
    smpl_params_global = gvhmr_pred['smpl_params_global']
    
    betas = np.pad(smpl_params_global['betas'][0], (0, 6))
    
    smplx_data = {
        'pose_body': smpl_params_global['body_pose'].numpy(),
        'betas': betas,
        'root_orient': smpl_params_global['global_orient'].numpy(),
        'trans': smpl_params_global['transl'].numpy(),
        "mocap_frame_rate": torch.tensor(30),
    }

    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender="neutral",
        use_pca=False,
    )
    
    num_frames = smpl_params_global['body_pose'].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1),  # (16,)
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),  # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),  # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(),  # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )
    
    # ====== FIX: Estimate height from FK joint positions ======
    joints_world = smplx_output.joints.detach().numpy()  # (num_frames, num_joints, 3)
    
    # Use middle 50% of frames for height estimation
    start_frame = num_frames // 4
    end_frame = 3 * num_frames // 4
    frame_indices = slice(start_frame, end_frame)
    
    human_height, frame_heights = estimate_human_height_from_joints(
        joints_world,
        frame_indices=frame_indices,
        head_joint_idx=15,
        foot_joint_indices=(10, 11)
    )
    
    # Clamp to reasonable human height range [1.4m, 2.2m]
    human_height = max(1.4, min(2.2, human_height))
    
    print(f"[load_gvhmr_pred_file] Height estimation:")
    print(f"  Estimated height: {human_height:.3f} m")
    print(f"  Frame height stats: min={np.min(frame_heights):.3f}m, max={np.max(frame_heights):.3f}m, "
          f"median={np.median(frame_heights):.3f}m, std={np.std(frame_heights):.3f}m")
    print(f"  Used frames: {start_frame}-{end_frame} (middle 50%)")
    
    # ========================================================
    
    return smplx_data, body_model, smplx_output, human_height


# Test script
if __name__ == "__main__":
    print("Height Estimation Fix for SMPL-X Retargeting")
    print("=" * 60)
    print()
    print("This fix addresses the bug where human height is always 1.66m")
    print("due to betas[0] being 0 in motion_135 format.")
    print()
    print("Key improvements:")
    print("  1. Measures height from FK-computed joint positions")
    print("  2. Uses head (joint 15) and feet (joints 10, 11) positions")
    print("  3. Robust to outliers using median across middle 50% of frames")
    print("  4. Clamps result to reasonable range [1.4m, 2.2m]")
    print()
    print("Joint indices in SMPL-X:")
    print("  - Joint 15: head")
    print("  - Joint 10: left_foot")
    print("  - Joint 11: right_foot")
    print("  - Y-axis: vertical")
    print()
    print("To apply this fix:")
    print("  Replace lines 50-53 in load_smplx_file() with:")
    print("    joints_world = smplx_output.joints.detach().numpy()")
    print("    human_height, _ = estimate_human_height_from_joints(joints_world)")
    print()
    print("  Replace lines 105-108 in load_gvhmr_pred_file() similarly.")
