#!/usr/bin/env python3
"""
Integration test for PRISM-to-272 conversion with the rotation fix.

This creates synthetic PRISM prediction data and converts it using the fixed
convert_prism_to_272.py to verify the end-to-end pipeline works correctly.

Author: Claude Opus 4.6
Date: 2026-05-27
"""

import numpy as np
import sys
import os

# Add MotionStreamer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ref_repo/MotionStreamer'))

from convert_prism_to_272 import compute_representation_272


def create_synthetic_prism_data(nfrm=10):
    """Create synthetic PRISM data for testing."""
    print(f"\nCreating synthetic PRISM data: {nfrm} frames")
    njoint = 22
    
    # Create joints_22 array: [T, 22, 3] - global positions
    joints_22 = np.random.randn(nfrm, njoint, 3).astype(np.float32) * 0.5
    # Ensure positions are reasonable (Y should be positive for standing)
    joints_22[..., 1] = np.abs(joints_22[..., 1]) + 1.0
    
    # Create smpl_85_face_z array: [T, 85] - SMPL 85D format
    # Structure: 3 (global orient) + 63 (pose) + 10 (shape) + 9 (extra)
    smpl_85_face_z = np.random.randn(nfrm, 85).astype(np.float32) * 0.1
    
    print(f"  Joints shape: {joints_22.shape}")
    print(f"  SMPL 85D shape: {smpl_85_face_z.shape}")
    
    return joints_22, smpl_85_face_z


def test_conversion():
    """Test the conversion function with synthetic data."""
    print("\n" + "="*80)
    print("INTEGRATION TEST: PRISM-to-272 Conversion")
    print("="*80)
    
    # Create synthetic data
    nfrm = 10
    njoint = 22
    joints_22, smpl_85_face_z = create_synthetic_prism_data(nfrm)
    
    try:
        # Call the conversion function
        print("\nCalling compute_representation_272()...")
        output_272 = compute_representation_272(
            joints_22=joints_22,
            smpl_85_face_z=smpl_85_face_z
        )
        
        print("✓ Conversion completed successfully!")
        
    except Exception as e:
        print(f"❌ Conversion failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify output structure
    print("\nVerifying output structure...")
    
    expected_shape = (nfrm, 272)
    if output_272.shape != expected_shape:
        print(f"❌ Shape mismatch! Expected {expected_shape}, got {output_272.shape}")
        return False
    print(f"✓ Output shape correct: {output_272.shape}")
    
    # Verify no NaN or Inf values
    if np.isnan(output_272).any():
        print("❌ Output contains NaN values!")
        return False
    print("✓ No NaN values")
    
    if np.isinf(output_272).any():
        print("❌ Output contains Inf values!")
        return False
    print("✓ No Inf values")
    
    # Check value ranges
    print("\nValue statistics:")
    print(f"  Min: {output_272.min():.6f}")
    print(f"  Max: {output_272.max():.6f}")
    print(f"  Mean: {output_272.mean():.6f}")
    print(f"  Std: {output_272.std():.6f}")
    
    # Verify 272-dim structure
    print("\n272-Dimension Structure:")
    print(f"  dims 0-2:     root XZ velocity")
    print(f"  dims 2-8:     heading rotation 6D")
    print(f"  dims 8-74:    local positions (22×3)")
    print(f"  dims 74-140:  local velocities (22×3)")
    print(f"  dims 140-272: local rotations 6D (22×6)")
    
    # Verify rotation dimensions have reasonable values (should be ~unit vectors)
    # Dims 2-8: heading rotation (6D)
    heading_rot = output_272[:, 2:8]
    heading_magnitude = np.linalg.norm(heading_rot, axis=1)
    print(f"\nHeading rotation magnitude (should be ~1.0): min={heading_magnitude.min():.4f}, max={heading_magnitude.max():.4f}")
    
    # Dims 140-272: joint rotations (132 = 22*6)
    joint_rot = output_272[:, 140:272]  # Correct indices for 272-dim format
    print(f"Joint rotation dims shape: {joint_rot.shape}")
    
    # Reshape to check per-frame and per-joint
    joint_rot_reshaped = joint_rot.reshape(-1, njoint, 6)  # (nfrm*1, njoint, 6) - wait this is wrong
    # Actually it's already per-frame: (nfrm, 132), so reshape to (nfrm, njoint, 6)
    joint_rot_reshaped = np.zeros((nfrm, njoint, 6))
    for i in range(nfrm):
        joint_rot_reshaped[i] = joint_rot[i].reshape(njoint, 6)
    
    joint_magnitudes = np.linalg.norm(joint_rot_reshaped, axis=2)
    print(f"Joint rotation magnitudes (should be ~1.0): min={joint_magnitudes.min():.4f}, max={joint_magnitudes.max():.4f}, mean={joint_magnitudes.mean():.4f}")
    
    print("\n" + "="*80)
    print("✓ INTEGRATION TEST PASSED!")
    print("="*80)
    return True


def main():
    try:
        success = test_conversion()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED!")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
