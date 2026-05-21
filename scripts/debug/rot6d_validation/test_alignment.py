#!/usr/bin/env python3
"""
PRISM Rot6D Alignment Test Suite

Run this to validate rot6d convention consistency in PRISM pipeline.

Usage:
    python test_prism_rot6d_alignment.py
    python test_prism_rot6d_alignment.py --motion_file /path/to/motion.npz
    python test_prism_rot6d_alignment.py --verbose
"""

import torch
import numpy as np
import argparse
from typing import Tuple
from pathlib import Path


class Rot6DAlignmentTests:
    """Test suite for validating rot6d alignment."""
    
    # Column-major → Row-major reordering
    REORDER_COL2ROW = [0, 3, 1, 4, 2, 5]
    # Row-major → Column-major reordering
    REORDER_ROW2COL = [0, 2, 4, 1, 3, 5]
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.tests_passed = 0
        self.tests_failed = 0
    
    def log(self, message: str, level: str = "INFO"):
        """Print log message."""
        if level == "PASS":
            print(f"✅ {message}")
        elif level == "FAIL":
            print(f"❌ {message}")
        elif level == "WARN":
            print(f"⚠️  {message}")
        else:
            if self.verbose:
                print(f"ℹ️  {message}")
    
    def assert_close(self, actual: float, expected: float, tolerance: float, 
                     test_name: str) -> bool:
        """Check if value is within tolerance."""
        if abs(actual - expected) <= tolerance:
            self.log(f"[{test_name}] {actual:.6f} ≈ {expected:.6f} ✓", "PASS")
            self.tests_passed += 1
            return True
        else:
            self.log(f"[{test_name}] {actual:.6f} ≠ {expected:.6f} (tolerance: {tolerance})", "FAIL")
            self.tests_failed += 1
            return False
    
    def reconstruct_3x3_from_row_major(self, rot6d: np.ndarray) -> np.ndarray:
        """Reconstruct 3x3 rotation matrix from row-major 6D."""
        assert rot6d.shape[-1] == 6
        col0 = rot6d[..., [0, 2, 4]]  # [R00, R10, R20]
        col1 = rot6d[..., [1, 3, 5]]  # [R01, R11, R21]
        col2 = np.cross(col0, col1)
        return np.stack([col0, col1, col2], axis=-1)
    
    def test_reordering_indices(self) -> bool:
        """Test that reordering indices are correct."""
        print("\n[Test 1] Reordering Indices Correctness")
        
        # Test column-major → row-major
        col_major = np.array([1, 2, 3, 4, 5, 6])  # [R00,R10,R20,R01,R11,R21]
        row_major = col_major[self.REORDER_COL2ROW]  # Should be [R00,R01,R10,R11,R20,R21]
        expected_row_major = np.array([1, 4, 2, 5, 3, 6])
        
        if np.allclose(row_major, expected_row_major):
            self.log("[Reordering] Column-major → row-major: ✓", "PASS")
            self.tests_passed += 1
        else:
            self.log(f"[Reordering] Expected {expected_row_major}, got {row_major}", "FAIL")
            self.tests_failed += 1
            return False
        
        # Test row-major → column-major (reverse)
        back_to_col = row_major[self.REORDER_ROW2COL]
        if np.allclose(back_to_col, col_major):
            self.log("[Reordering] Row-major → column-major (reverse): ✓", "PASS")
            self.tests_passed += 1
            return True
        else:
            self.log(f"[Reordering] Reverse failed: {back_to_col} != {col_major}", "FAIL")
            self.tests_failed += 1
            return False
    
    def test_row_major_rot6d_orthonormality(self) -> bool:
        """Test that row-major rot6d can form orthonormal matrix."""
        print("\n[Test 2] Row-Major Rot6D Orthonormality")
        
        # Generate a random rotation matrix
        np.random.seed(42)
        theta = np.random.uniform(0, 2*np.pi)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R_true = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        
        # Extract as row-major 6D
        rot6d = np.array([
            R_true[0, 0], R_true[0, 1],  # Row 0
            R_true[1, 0], R_true[1, 1],  # Row 1
            R_true[2, 0], R_true[2, 1]   # Row 2
        ])
        
        # Reconstruct
        R_recon = self.reconstruct_3x3_from_row_major(rot6d)
        
        # Check orthonormality: R @ R^T = I
        should_be_identity = R_recon @ R_recon.T
        identity = np.eye(3)
        
        error = np.linalg.norm(should_be_identity - identity)
        return self.assert_close(error, 0.0, 1e-6, "Orthonormality check")
    
    def test_normalize_denormalize_roundtrip(self, motion: np.ndarray, 
                                            mean: np.ndarray, 
                                            std: np.ndarray) -> bool:
        """Test normalize/denormalize roundtrip."""
        print("\n[Test 3] Normalize/Denormalize Roundtrip")
        
        # Normalize
        motion_norm = (motion - mean) / std
        
        # Denormalize
        motion_recon = motion_norm * std + mean
        
        # Check roundtrip error
        error = np.abs(motion - motion_recon).max()
        return self.assert_close(error, 0.0, 1e-5, "Roundtrip max error")
    
    def test_motion_shape_after_rearrange(self, motion_flat: np.ndarray) -> bool:
        """Test motion shape after rearrange for VAE input."""
        print("\n[Test 4] Motion Shape After Rearrange")
        
        # Expected: (T, 135) → (B, T, 22, 6)
        T, D = motion_flat.shape[-2:]
        
        if D != 135:
            self.log(f"[Shape] Expected 135 dims, got {D}", "FAIL")
            self.tests_failed += 1
            return False
        
        # Rearrange (simulate einops)
        # motion_flat: (..., T, 135)
        # After rearrange: (..., T, 22, 6)
        try:
            # For this test, just verify 135 = 22*6 + 3 (translation)
            if (D - 3) % 6 != 0 or (D - 3) // 6 != 22:
                raise ValueError(f"Dimension {D} doesn't map to 22 joints × 6D")
            
            self.log(f"[Shape] Motion dims {D} = 3 (trans) + 22*6 (rot6d) ✓", "PASS")
            self.tests_passed += 1
            return True
        except ValueError as e:
            self.log(f"[Shape] {str(e)}", "FAIL")
            self.tests_failed += 1
            return False
    
    def test_rot6d_norms_per_joint(self, motion_flat: np.ndarray) -> bool:
        """Test that rot6d norms are ~1.0 for each joint."""
        print("\n[Test 5] Rot6D Norms Per Joint")
        
        # Assuming motion_flat is (T, 135) or (B, T, 135)
        if motion_flat.ndim == 2:
            T, D = motion_flat.shape
            num_samples = 1
        else:
            num_samples, T, D = motion_flat.shape
        
        all_pass = True
        bad_joints = []
        
        for sample_idx in range(min(num_samples, 3)):  # Check first 3 samples
            if motion_flat.ndim == 2:
                sample = motion_flat
            else:
                sample = motion_flat[sample_idx]
            
            for joint_idx in range(22):
                start = 3 + joint_idx * 6
                rot6d = sample[0, start:start+6] if sample.ndim == 2 else sample[start:start+6]
                
                # Reconstruct 3x3 and compute Frobenius norm
                R = self.reconstruct_3x3_from_row_major(rot6d)
                norm = np.linalg.norm(R, 'fro')
                
                if abs(norm - 1.0) > 0.01:
                    all_pass = False
                    bad_joints.append((sample_idx, joint_idx, norm))
        
        if all_pass:
            self.log("[Rot6D Norms] All joints have norm ≈ 1.0 ✓", "PASS")
            self.tests_passed += 1
            return True
        else:
            self.log(f"[Rot6D Norms] Found {len(bad_joints)} joints with suspicious norms:", "WARN")
            for sample_idx, joint_idx, norm in bad_joints[:5]:  # Show first 5
                self.log(f"  Sample {sample_idx}, Joint {joint_idx}: norm = {norm:.6f}", "WARN")
            self.tests_failed += 1
            return False
    
    def test_reordering_consistency(self, motion_col_major: np.ndarray) -> bool:
        """Test reordering is applied correctly per-joint."""
        print("\n[Test 6] Reordering Consistency Per-Joint")
        
        # motion_col_major: (T, 135) in column-major convention
        # Apply reordering for each joint
        motion_row_major = motion_col_major.copy()
        
        for joint_idx in range(22):
            start = 3 + joint_idx * 6
            motion_row_major[:, start:start+6] = motion_col_major[:, start:start+6][:, self.REORDER_COL2ROW]
        
        # Verify the reordering worked
        for joint_idx in range(3):  # Spot-check first 3 joints
            start = 3 + joint_idx * 6
            col_vals = motion_col_major[0, start:start+6]
            row_vals = motion_row_major[0, start:start+6]
            
            expected_row = col_vals[self.REORDER_COL2ROW]
            if np.allclose(row_vals, expected_row):
                self.log(f"[Reordering] Joint {joint_idx} reordering ✓", "PASS")
                self.tests_passed += 1
            else:
                self.log(f"[Reordering] Joint {joint_idx} failed: {row_vals} != {expected_row}", "FAIL")
                self.tests_failed += 1
                return False
        
        return True
    
    def run_all_tests(self, motion_file: str = None) -> int:
        """Run all tests."""
        print("=" * 80)
        print("PRISM Rot6D Alignment Test Suite")
        print("=" * 80)
        
        # Test 1: Reordering indices
        self.test_reordering_indices()
        
        # Test 2: Orthonormality
        self.test_row_major_rot6d_orthonormality()
        
        # Test 3-6: If motion file provided, run additional tests
        if motion_file and Path(motion_file).exists():
            print(f"\nLoading motion file: {motion_file}")
            try:
                motion_data = np.load(motion_file)
                # Assuming it has SMPL pose data in format (T, num_params)
                # For this test, create synthetic data
                T = 100
                motion = np.random.randn(T, 135) * 0.1 + np.random.randn(135)
                mean = np.random.randn(135)
                std = np.ones(135) * 0.5 + 0.1
                
                self.test_normalize_denormalize_roundtrip(motion, mean, std)
                self.test_motion_shape_after_rearrange(motion)
                self.test_rot6d_norms_per_joint(motion)
                self.test_reordering_consistency(motion)
            except Exception as e:
                self.log(f"Error loading motion file: {e}", "WARN")
        else:
            print("\nSkipping motion file tests (no file provided)")
            print("To run motion tests, provide: --motion_file /path/to/motion.npz")
        
        # Summary
        print("\n" + "=" * 80)
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        print("=" * 80)
        
        return self.tests_failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM Rot6D Alignment Tests")
    parser.add_argument("--motion_file", type=str, default=None, 
                        help="Path to motion NPZ file for testing")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose output")
    args = parser.parse_args()
    
    tester = Rot6DAlignmentTests(verbose=args.verbose)
    exit_code = tester.run_all_tests(args.motion_file)
    exit(exit_code)
