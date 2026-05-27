#!/usr/bin/env python3
"""
Converter from SMPLX-55 format to 135-dimensional HumanML3D evaluation format.

This utility converts PRISM predictions from stored SMPLX-55 format (axis-angle)
to the 135-dimensional format used for metric evaluation:
  - 3D translation + 22 joints with 6D rotations = 135D

The 135D format is used by eval_with_motionclip_evaluator.py for computing:
  - R-Precision (text-motion alignment)
  - MM-Distance (motion diversity)
  - FID (Fréchet Inception Distance)
  - Diversity (motion variation)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, Tuple


class SMPLX55To135DConverter:
    """Convert SMPLX-55 format to 135-dimensional evaluation format."""
    
    # SMPL-H 22 joint indices in body_pose ordering
    # These are the joints used in HumanML3D evaluation
    HUMANML3D_JOINT_INDICES = [
        0,   # Hips/Pelvis (global_orient [0])
        1,   # Right Hip
        2,   # Right Knee  
        3,   # Right Ankle
        4,   # Left Hip
        5,   # Left Knee
        6,   # Left Ankle
        7,   # Spine
        8,   # Chest/Thorax
        9,   # Neck
        10,  # Head
        11,  # Right Shoulder
        12,  # Right Elbow
        13,  # Right Wrist
        14,  # Left Shoulder
        15,  # Left Elbow
        16,  # Left Wrist
        17,  # Right Hand (first joint)
        18,  # Left Hand (first joint)
        19,  # Right Foot
        20,  # Left Foot
    ]
    
    @staticmethod
    def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
        """
        Convert axis-angle rotations to rotation matrices.
        
        Args:
            axis_angle: (N, 3) array of axis-angle rotations
            
        Returns:
            (N, 3, 3) rotation matrices
        """
        rotation = Rotation.from_rotvec(axis_angle)
        return rotation.as_matrix()
    
    @staticmethod
    def rotation_matrix_to_6d(rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrices to 6D representation (first 2 rows).
        
        Args:
            rotation_matrix: (..., 3, 3) rotation matrices
            
        Returns:
            (..., 6) 6D rotation representation
        """
        assert rotation_matrix.shape[-2:] == (3, 3)
        # Take first 2 rows and flatten
        return rotation_matrix[..., :2, :].reshape(*rotation_matrix.shape[:-2], 6)
    
    @staticmethod
    def extract_humanml3d_joints(global_orient: np.ndarray, 
                                  body_pose: np.ndarray) -> np.ndarray:
        """
        Extract 22 HumanML3D joints from SMPLX format.
        
        Args:
            global_orient: (T, 3) root orientation in axis-angle
            body_pose: (T, 63) 21 body joints in axis-angle (21 * 3)
            
        Returns:
            (T, 22, 3) selected joints in axis-angle
        """
        T = global_orient.shape[0]
        joints_22 = np.zeros((T, 22, 3), dtype=np.float32)
        
        # First joint is global_orient
        joints_22[:, 0, :] = global_orient
        
        # Rest are from body_pose
        # Indices in HUMANML3D_JOINT_INDICES[1:] correspond to body_pose joints
        for i, body_idx in enumerate(SMPLX55To135DConverter.HUMANML3D_JOINT_INDICES[1:]):
            if body_idx < 21:  # Valid body_pose index
                joints_22[:, i + 1, :] = body_pose[:, body_idx * 3:(body_idx + 1) * 3]
            else:
                # For joints beyond standard body pose (hands, etc)
                # Use zero rotation as fallback
                joints_22[:, i + 1, :] = 0
        
        return joints_22
    
    @classmethod
    def convert_smplx55_to_135d(cls, smplx_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert SMPLX-55 format to 135D evaluation format.
        
        Args:
            smplx_data: Dictionary from np.load() with keys:
                - 'transl': (T, 3) translation
                - 'global_orient': (T, 3) root rotation [axis-angle]
                - 'body_pose': (T, 63) 21 joints [axis-angle]
                
        Returns:
            motion_135d: (T, 135) array with format:
                - [0:3]: translation (XYZ)
                - [3:135]: 22 joints × 6D rotation (132D)
        """
        transl = smplx_data['transl']           # (T, 3)
        global_orient = smplx_data['global_orient']  # (T, 3)
        body_pose = smplx_data['body_pose']     # (T, 63)
        
        T = transl.shape[0]
        
        # Extract 22 HumanML3D joints
        joints_22_aa = cls.extract_humanml3d_joints(global_orient, body_pose)  # (T, 22, 3)
        
        # Convert axis-angle to rotation matrices
        # Reshape to (T*22, 3) for batch processing
        joints_aa_flat = joints_22_aa.reshape(-1, 3)
        rot_matrices_flat = cls.axis_angle_to_rotation_matrix(joints_aa_flat)  # (T*22, 3, 3)
        
        # Reshape back to (T, 22, 3, 3)
        rot_matrices = rot_matrices_flat.reshape(T, 22, 3, 3)
        
        # Convert to 6D representation
        rot_6d = cls.rotation_matrix_to_6d(rot_matrices)  # (T, 22, 6)
        
        # Flatten 22 joints × 6D to (T, 132)
        rot_6d_flat = rot_6d.reshape(T, -1)
        
        # Combine translation + rotations
        motion_135d = np.concatenate([transl, rot_6d_flat], axis=1)  # (T, 135)
        
        return motion_135d.astype(np.float32)
    
    @classmethod
    def convert_batch(cls, npz_file_paths: list) -> Dict[str, np.ndarray]:
        """
        Convert batch of SMPLX-55 NPZ files to 135D format.
        
        Args:
            npz_file_paths: List of paths to NPZ files
            
        Returns:
            Dictionary mapping filename stems to 135D motion arrays
        """
        results = {}
        for npz_path in npz_file_paths:
            try:
                data = np.load(npz_path)
                motion_135d = cls.convert_smplx55_to_135d(data)
                stem = Path(npz_path).stem
                results[stem] = motion_135d
            except Exception as e:
                print(f"Error converting {npz_path}: {e}")
        
        return results


def demonstrate_conversion():
    """Demonstrate format conversion with a sample motion."""
    from pathlib import Path
    
    print("=" * 70)
    print("SMPLX-55 to 135D Format Conversion Example")
    print("=" * 70)
    
    # Load sample SMPLX-55 motion
    eval_dir = Path("work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten")
    sample_npz = eval_dir / "humanml3d_10006.npz"
    
    if not sample_npz.exists():
        print(f"Sample file not found: {sample_npz}")
        return
    
    print(f"\nLoading: {sample_npz}")
    smplx_data = np.load(sample_npz)
    
    print("\n--- SMPLX-55 Format (Input) ---")
    print(f"transl shape:       {smplx_data['transl'].shape}")
    print(f"global_orient:      {smplx_data['global_orient'].shape}")
    print(f"body_pose:          {smplx_data['body_pose'].shape}")
    print(f"Total frames (T):   {smplx_data['transl'].shape[0]}")
    
    # Convert to 135D
    converter = SMPLX55To135DConverter()
    motion_135d = converter.convert_smplx55_to_135d(smplx_data)
    
    print("\n--- 135D Format (Output) ---")
    print(f"Shape:              {motion_135d.shape}")
    print(f"Dtype:              {motion_135d.dtype}")
    print(f"Breakdown:          3 (transl) + 132 (22 joints × 6D) = 135")
    
    print("\n--- Value Statistics ---")
    print(f"Min value:          {np.min(motion_135d):.6f}")
    print(f"Max value:          {np.max(motion_135d):.6f}")
    print(f"Mean value:         {np.mean(motion_135d):.6f}")
    print(f"Std dev:            {np.std(motion_135d):.6f}")
    
    # Show per-component stats
    transl_135d = motion_135d[:, :3]
    rot_135d = motion_135d[:, 3:]
    
    print(f"\nTranslation (first 3 dims):")
    print(f"  Min: {np.min(transl_135d):.6f}, Max: {np.max(transl_135d):.6f}")
    print(f"\nRotations (remaining 132 dims):")
    print(f"  Min: {np.min(rot_135d):.6f}, Max: {np.max(rot_135d):.6f}")
    
    print("\n" + "=" * 70)
    return motion_135d


if __name__ == "__main__":
    from pathlib import Path
    
    motion_135d = demonstrate_conversion()
    
    if motion_135d is not None:
        print("\nConversion successful! The 135D motion is ready for metric evaluation.")
        print("\nUsage in evaluation:")
        print("  python scripts/eval/eval_with_motionclip_evaluator.py \\")
        print("      --pred_dir work_dirs/.../eval_hml3d_rewritten \\")
        print("      --device cuda:0")

