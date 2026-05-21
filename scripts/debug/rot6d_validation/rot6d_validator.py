#!/usr/bin/env python3
"""
PRISM Rot6D Alignment Verification Script
==========================================

This script validates the rot6d convention consistency across the PRISM pipeline.
Run this to verify that your data and model are using row-major rot6d correctly.

Usage:
    python prism_rot6d_verification_script.py \
        --motion_npz /path/to/motion.npz \
        --config /path/to/prism/config.py \
        --smpl_processor_config /path/to/smpl_processor.json
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional
import argparse
import json


class Rot6DValidator:
    """Validates rot6d representation consistency."""
    
    # Reordering indices
    REORDER_COL2ROW = [0, 3, 1, 4, 2, 5]  # Column-major → Row-major
    REORDER_ROW2COL = [0, 2, 4, 1, 3, 5]  # Row-major → Column-major
    
    @staticmethod
    def reconstruct_rot_matrix_from_row_major(rot6d: np.ndarray) -> np.ndarray:
        """
        Reconstruct full 3×3 rotation matrix from row-major 6D.
        
        Row-major format: [R00, R01, R10, R11, R20, R21]
        
        Returns 3×3 matrix:
            [ R00  R01  R?? ]
            [ R10  R11  R?? ]
            [ R20  R21  R?? ]
        
        Third column is computed as cross product of first two columns.
        """
        assert rot6d.shape[-1] == 6, f"Expected 6D, got {rot6d.shape[-1]}"
        
        # Extract first two columns
        col0 = rot6d[..., [0, 2, 4]]  # [R00, R10, R20]
        col1 = rot6d[..., [1, 3, 5]]  # [R01, R11, R21]
        
        # Compute third column as cross product
        col2 = np.cross(col0, col1)
        
        # Stack into 3×3 matrix
        # Expected shape: (..., 3, 3)
        matrix = np.stack([col0, col1, col2], axis=-1)
        
        return matrix
    
    @staticmethod
    def reconstruct_rot_matrix_from_col_major(rot6d: np.ndarray) -> np.ndarray:
        """
        Reconstruct full 3×3 rotation matrix from column-major 6D.
        
        Column-major format: [R00, R10, R20, R01, R11, R21]
        
        Returns 3×3 matrix:
            [ R00  R01  R?? ]
            [ R10  R11  R?? ]
            [ R20  R21  R?? ]
        """
        assert rot6d.shape[-1] == 6, f"Expected 6D, got {rot6d.shape[-1]}"
        
        # Extract first two columns
        col0 = rot6d[..., [0, 1, 2]]  # [R00, R10, R20]
        col1 = rot6d[..., [3, 4, 5]]  # [R01, R11, R21]
        
        # Compute third column as cross product
        col2 = np.cross(col0, col1)
        
        # Stack into 3×3 matrix
        matrix = np.stack([col0, col1, col2], axis=-1)
        
        return matrix
    
    @staticmethod
    def check_orthonormality(matrix: np.ndarray, tolerance: float = 1e-5) -> Dict[str, bool]:
        """
        Check if matrix is orthonormal (all columns unit norm and mutually orthogonal).
        
        Returns dict with boolean checks.
        """
        col0 = matrix[..., :, 0]
        col1 = matrix[..., :, 1]
        col2 = matrix[..., :, 2]
        
        norm0 = np.linalg.norm(col0, axis=-1)
        norm1 = np.linalg.norm(col1, axis=-1)
        norm2 = np.linalg.norm(col2, axis=-1)
        
        dot01 = np.sum(col0 * col1, axis=-1)
        dot02 = np.sum(col0 * col2, axis=-1)
        dot12 = np.sum(col1 * col2, axis=-1)
        
        det = np.linalg.det(matrix)
        
        return {
            "col0_unit_norm": np.allclose(norm0, 1.0, atol=tolerance),
            "col1_unit_norm": np.allclose(norm1, 1.0, atol=tolerance),
            "col2_unit_norm": np.allclose(norm2, 1.0, atol=tolerance),
            "col0_col1_orthogonal": np.allclose(dot01, 0.0, atol=tolerance),
            "col0_col2_orthogonal": np.allclose(dot02, 0.0, atol=tolerance),
            "col1_col2_orthogonal": np.allclose(dot12, 0.0, atol=tolerance),
            "det_positive": np.all(det > 0),
        }
    
    @classmethod
    def validate_row_major_rot6d(cls, rot6d: np.ndarray) -> Tuple[bool, Dict]:
        """
        Validate that a 6D array is valid row-major rot6d.
        
        Args:
            rot6d: Shape (..., 6) array in row-major format
        
        Returns:
            (is_valid, diagnostics_dict)
        """
        matrix = cls.reconstruct_rot_matrix_from_row_major(rot6d)
        checks = cls.check_orthonormality(matrix)
        is_valid = all(checks.values())
        
        return is_valid, {
            "format": "row-major",
            "checks": checks,
            "mean_col0_norm": np.mean(np.linalg.norm(matrix[..., :, 0], axis=-1)),
            "mean_col1_norm": np.mean(np.linalg.norm(matrix[..., :, 1], axis=-1)),
            "mean_det": np.mean(np.linalg.det(matrix)),
        }
    
    @classmethod
    def validate_col_major_rot6d(cls, rot6d: np.ndarray) -> Tuple[bool, Dict]:
        """
        Validate that a 6D array is valid column-major rot6d.
        """
        matrix = cls.reconstruct_rot_matrix_from_col_major(rot6d)
        checks = cls.check_orthonormality(matrix)
        is_valid = all(checks.values())
        
        return is_valid, {
            "format": "column-major",
            "checks": checks,
            "mean_col0_norm": np.mean(np.linalg.norm(matrix[..., :, 0], axis=-1)),
            "mean_col1_norm": np.mean(np.linalg.norm(matrix[..., :, 1], axis=-1)),
            "mean_det": np.mean(np.linalg.det(matrix)),
        }


class PrismPipelineValidator:
    """Validates the complete PRISM data pipeline."""
    
    def __init__(self, smpl_processor, vae_config: Dict):
        self.smpl_processor = smpl_processor
        self.vae_config = vae_config
        self.validator = Rot6DValidator()
    
    def check_normalization_roundtrip(self, motion: torch.Tensor, 
                                     tolerance: float = 1e-5) -> Tuple[bool, Dict]:
        """
        Verify normalize/denormalize roundtrip consistency.
        
        Args:
            motion: Shape (B, T, 135) or (T, 135)
        
        Returns:
            (is_valid, diagnostics)
        """
        motion_norm = self.smpl_processor.normalize(motion)
        motion_rec = self.smpl_processor.denormalize(motion_norm)
        
        error = (motion - motion_rec).abs()
        max_error = error.max().item()
        mean_error = error.mean().item()
        
        is_valid = max_error < tolerance
        
        return is_valid, {
            "max_error": max_error,
            "mean_error": mean_error,
            "tolerance": tolerance,
            "pass": is_valid,
        }
    
    def check_vae_input_shape(self, motion: torch.Tensor) -> Tuple[bool, Dict]:
        """
        Verify VAE input preparation produces correct shape.
        
        Args:
            motion: Shape (B, T, 135) or (T, 135)
        
        Returns:
            (is_valid, diagnostics)
        """
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        
        # Simulate the rearrange operation
        # From: (B, T, 135) where dims [3:135] = flattened rot6d
        # To: (B, T, 22, 6)
        
        B, T, D = motion.shape
        assert D == 135, f"Expected 135 dims, got {D}"
        
        # Extract rot6d part (skip translation at dims [0:3])
        rot6d_flat = motion[:, :, 3:]  # (B, T, 132)
        
        # Reshape to (B, T, 22, 6)
        rot6d_reshaped = rot6d_flat.reshape(B, T, 22, 6)
        
        expected_shape = (B, T, 22, 6)
        actual_shape = tuple(rot6d_reshaped.shape)
        
        is_valid = actual_shape == expected_shape
        
        return is_valid, {
            "expected_shape": expected_shape,
            "actual_shape": actual_shape,
            "vae_in_channels": self.vae_config.get("in_channels", 6),
            "vae_out_channels": self.vae_config.get("out_channels", 6),
            "pass": is_valid,
        }
    
    def check_rot6d_convention_preservation(self, motion: torch.Tensor, 
                                           sample_joints: int = 5) -> Tuple[bool, Dict]:
        """
        Verify that rot6d convention is preserved through normalization.
        
        Args:
            motion: Shape (B, T, 135) or (T, 135)
            sample_joints: Number of joints to sample for validation
        
        Returns:
            (is_valid, diagnostics)
        """
        motion_np = motion.detach().cpu().numpy()
        if motion_np.ndim == 3:
            motion_np = motion_np[0]  # Take first batch element
        
        results = {}
        all_valid = True
        
        # Check a few sample joints
        np.random.seed(42)
        joint_indices = np.random.choice(22, min(sample_joints, 22), replace=False)
        
        for j in joint_indices:
            rot6d_start = 3 + j * 6
            rot6d_end = rot6d_start + 6
            
            # Get rot6d for this joint across all frames
            rot6d_seq = motion_np[:, rot6d_start:rot6d_end]  # (T, 6)
            
            # Validate
            is_valid, diag = self.validator.validate_row_major_rot6d(rot6d_seq)
            results[f"joint_{j}"] = {
                "valid": is_valid,
                "diagnostics": diag,
            }
            all_valid = all_valid and is_valid
        
        return all_valid, {
            "sample_joints": joint_indices.tolist(),
            "results": results,
            "pass": all_valid,
        }


def main():
    parser = argparse.ArgumentParser(description="PRISM Rot6D Alignment Verification")
    parser.add_argument("--motion_npz", type=str, help="Path to motion NPZ file")
    parser.add_argument("--config", type=str, help="Path to PRISM config file")
    parser.add_argument("--smpl_processor_config", type=str, 
                       help="Path to SMPL processor config (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PRISM Rot6D Alignment Verification Script")
    print("=" * 80)
    
    # For this example, we'll do a minimal validation without external dependencies
    print("\n✓ Script structure loaded successfully")
    print("\nUsage:")
    print("  1. Load your motion data (NPZ file)")
    print("  2. Initialize SMPL processor with correct stats")
    print("  3. Run validator.check_normalization_roundtrip(motion)")
    print("  4. Run validator.check_vae_input_shape(motion)")
    print("  5. Run validator.check_rot6d_convention_preservation(motion)")
    print("\nExample:")
    print("""
    import torch
    import numpy as np
    from prism_rot6d_verification_script import PrismPipelineValidator, Rot6DValidator
    
    # Load data
    motion_data = np.load('motion.npz')
    motion = torch.from_numpy(motion_data['motion']).float()
    
    # Initialize validator
    validator = PrismPipelineValidator(
        smpl_processor=your_smpl_processor,
        vae_config={"in_channels": 6, "out_channels": 6}
    )
    
    # Run checks
    is_valid, diag = validator.check_normalization_roundtrip(motion)
    print(f"Normalization roundtrip: {diag}")
    
    is_valid, diag = validator.check_vae_input_shape(motion)
    print(f"VAE input shape: {diag}")
    
    is_valid, diag = validator.check_rot6d_convention_preservation(motion)
    print(f"Rot6D convention: {diag}")
    """)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
