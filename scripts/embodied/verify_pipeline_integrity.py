#!/usr/bin/env python3
"""Verify embodied pipeline integrity and identify specific bugs.

This script checks:
1. G1 body indices (verify foot_body_indices are correct)
2. Coordinate frame conversions
3. Quaternion conventions
4. Joint limits
5. Ground height reasonableness

Usage:
    python scripts/embodied/verify_pipeline_integrity.py \
        --mjcf ref_repo/ProtoMotions/data/robot_assets/g1/mjcf/g1_holo_compat.xml \
        [--test-motion /tmp/gmr.pkl]
"""

import argparse
import sys
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R

def check_mjcf_bodies(mjcf_path):
    """Check body indices in MJCF."""
    try:
        import mujoco
    except ImportError:
        print("ERROR: mujoco not installed. Skipping body index check.")
        return
    
    print("\n" + "="*70)
    print("CHECKING: MuJoCo Body Indices")
    print("="*70)
    
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    
    print(f"\nTotal bodies: {model.nbody}")
    print("\nBody index mapping:")
    print("-" * 70)
    
    ankle_indices = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjOBJ_BODY, i)
        print(f"  {i:2d}: {name}")
        if "ankle" in (name or "").lower():
            ankle_indices.append(i)
    
    print("\n" + "-" * 70)
    print(f"Detected ankle bodies: {ankle_indices}")
    print(f"Hardcoded foot_body_indices: [7, 13]")
    
    if set(ankle_indices) == {7, 13}:
        print("✓ PASS: Body indices appear correct")
    else:
        print("✗ FAIL: Body indices may be WRONG!")
        print(f"  Expected: [7, 13]")
        print(f"  Found:    {ankle_indices}")
    
    return ankle_indices


def check_coordinate_frames():
    """Verify coordinate frame conversion."""
    print("\n" + "="*70)
    print("CHECKING: Coordinate Frame Conversions")
    print("="*70)
    
    # Test SMPL-X Y-up → MuJoCo Z-up conversion
    from scipy.spatial.transform import Rotation as R
    
    # GMR's rot_offset (wxyz): [0.5, -0.5, -0.5, -0.5]
    # Should represent 120° rotation around [0.577, 0.577, 0.577] (111 axis)
    rot_offset_wxyz = np.array([0.5, -0.5, -0.5, -0.5])
    rot_offset_xyzw = rot_offset_wxyz[[1, 2, 3, 0]]  # Convert to xyzw
    
    print(f"\nGMR rot_offset (wxyz): {rot_offset_wxyz}")
    print(f"GMR rot_offset (xyzw): {rot_offset_xyzw}")
    
    rot = R.from_quat(rot_offset_xyzw)
    print(f"Euler angles: {rot.as_euler('xyz', degrees=True)}")
    print(f"Rotation angle (deg): {np.linalg.norm(rot.as_rotvec()) * 180 / np.pi:.1f}")
    
    # Test position conversion
    test_pos_smplx = np.array([1.0, 2.0, 3.0])  # X=right, Y=up, Z=forward
    test_pos_mujoco = rot.inv().apply(test_pos_smplx)
    
    print(f"\nTest position conversion:")
    print(f"  SMPL-X (X-right, Y-up, Z-forward):  {test_pos_smplx}")
    print(f"  MuJoCo (X-forward, Y-lateral, Z-up): {test_pos_mujoco}")
    print(f"  Expected mapping: [Z, X, Y] = [3.0, 1.0, 2.0]")
    print(f"  Close match: {np.allclose(test_pos_mujoco, [3.0, 1.0, 2.0], atol=1e-6)}")
    
    if np.allclose(test_pos_mujoco, [3.0, 1.0, 2.0], atol=1e-6):
        print("✓ PASS: Coordinate frame conversion is correct")
    else:
        print("✗ FAIL: Coordinate frame conversion may be WRONG!")


def check_quaternion_convention():
    """Verify quaternion conversion functions."""
    print("\n" + "="*70)
    print("CHECKING: Quaternion Conventions")
    print("="*70)
    
    # Test the conversion functions
    def quat_xyzw_to_wxyz(q):
        return q[..., [3, 0, 1, 2]]
    
    def quat_wxyz_to_xyzw(q):
        return q[..., [1, 2, 3, 0]]
    
    # Test with a known quaternion
    q_xyzw = np.array([0.0, 0.707107, 0.0, 0.707107])  # 90° around Y
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    q_xyzw_back = quat_wxyz_to_xyzw(q_wxyz)
    
    print(f"\nTest quaternion (xyzw): {q_xyzw}")
    print(f"Converted to wxyz: {q_wxyz}")
    print(f"Converted back to xyzw: {q_xyzw_back}")
    print(f"Round-trip successful: {np.allclose(q_xyzw, q_xyzw_back)}")
    
    # Verify scipy uses xyzw
    r = R.from_quat(q_xyzw)
    print(f"\nVerify scipy uses xyzw format:")
    print(f"  Input (90° around Y): {q_xyzw}")
    print(f"  As Euler: {r.as_euler('xyz', degrees=True)}")
    print(f"  Expected: [0, 90, 0] (or close)")
    
    expected = np.array([0, 90, 0])
    actual = r.as_euler('xyz', degrees=True)
    if np.allclose(actual, expected, atol=1.0):
        print("✓ PASS: Quaternion conventions are correct")
    else:
        print("✗ FAIL: Quaternion convention may be WRONG!")


def check_joint_limits_in_pkl(pkl_path):
    """Check if DOF values are reasonable for G1."""
    import pickle
    
    print("\n" + "="*70)
    print(f"CHECKING: Joint Limits in {pkl_path}")
    print("="*70)
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    dof_pos = data["dof_pos"]
    print(f"\nDOF positions shape: {dof_pos.shape}")
    print(f"DOF value ranges:")
    for i in range(min(8, dof_pos.shape[1])):  # Show first 8 DOFs
        vals = dof_pos[:, i]
        print(f"  DOF {i}: [{vals.min():.4f}, {vals.max():.4f}] (mean: {vals.mean():.4f})")
    
    # Common G1 joint limits (approximate)
    g1_limits = {
        "hip_pitch": [-1.4, 1.4],      # DOF 0, 6
        "hip_roll": [-0.6, 0.6],       # DOF 1, 7
        "hip_yaw": [-1.5, 1.5],        # DOF 2, 8
        "knee": [-0.1, 2.5],           # DOF 3, 9
        "ankle_pitch": [-0.7, 0.7],    # DOF 4, 10
        "ankle_roll": [-0.6, 0.6],     # DOF 5, 11
    }
    
    print("\n" + "-" * 70)
    print("Common violations check:")
    
    violations = 0
    for i in range(dof_pos.shape[1]):
        vals = dof_pos[:, i]
        if vals.min() < -3.14 or vals.max() > 3.14:
            print(f"✗ DOF {i}: EXTREME VALUES [{vals.min():.4f}, {vals.max():.4f}]")
            violations += 1
    
    if violations == 0:
        print("✓ PASS: No extreme DOF values detected")
    else:
        print(f"✗ FAIL: {violations} DOFs have concerning values")


def check_ground_height(pkl_path, mjcf_path):
    """Check if ground height is reasonable."""
    import pickle
    
    print("\n" + "="*70)
    print(f"CHECKING: Ground Height Reasonableness")
    print("="*70)
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    root_pos = data["root_pos"]
    print(f"\nRoot position Z (height):")
    print(f"  Min: {root_pos[:, 2].min():.4f}m")
    print(f"  Max: {root_pos[:, 2].max():.4f}m")
    print(f"  Mean: {root_pos[:, 2].mean():.4f}m")
    print(f"  G1 standing height (reference): 0.796m")
    
    expected_z = 0.796
    actual_z = root_pos[:, 2].mean()
    
    if abs(actual_z - expected_z) < 0.1:
        print(f"✓ PASS: Ground height is reasonable")
    else:
        print(f"✗ FAIL: Ground height may be WRONG!")
        print(f"  Error: {abs(actual_z - expected_z):.3f}m")


def main():
    parser = argparse.ArgumentParser(description="Verify embodied pipeline integrity")
    parser.add_argument("--mjcf", required=True, help="Path to G1 MJCF XML")
    parser.add_argument("--test-motion", help="Path to GMR PKL file for further checks")
    args = parser.parse_args()
    
    mjcf_path = pathlib.Path(args.mjcf)
    if not mjcf_path.exists():
        print(f"ERROR: MJCF not found: {mjcf_path}")
        sys.exit(1)
    
    # Run checks
    check_mjcf_bodies(mjcf_path)
    check_coordinate_frames()
    check_quaternion_convention()
    
    if args.test_motion:
        pkl_path = pathlib.Path(args.test_motion)
        if not pkl_path.exists():
            print(f"\nWARNING: Test motion not found: {pkl_path}")
        else:
            check_joint_limits_in_pkl(pkl_path)
            check_ground_height(pkl_path, mjcf_path)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nRun this script to verify the pipeline before/after fixes:")
    print("  python scripts/embodied/verify_pipeline_integrity.py \\")
    print("      --mjcf ref_repo/ProtoMotions/data/robot_assets/g1/mjcf/g1_holo_compat.xml \\")
    print("      --test-motion /path/to/gmr.pkl")
    print()


if __name__ == "__main__":
    main()
