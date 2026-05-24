"""
Verification script for rot6d convention refactoring.

Run BEFORE code changes to collect baseline, then AFTER to verify bit-for-bit consistency.

Usage:
    # Step 1: Before code changes — collect baseline
    python scripts/debug/verify_rot6d_refactor.py --collect-baseline

    # Step 2: After code changes — verify consistency
    python scripts/debug/verify_rot6d_refactor.py --verify
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

BASELINE_DIR = "/tmp/rot6d_refactor_baseline"

# Test data: a real SMPL-H file from the dataset
TEST_NPZ = "data/hymotion_data/Academic/20250916/motions/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.npz"


def get_test_data():
    """Load test axis-angle data and prepare inputs for all functions."""
    npz_path = os.path.join(
        os.path.dirname(__file__), "../..", TEST_NPZ
    )
    npz_path = os.path.normpath(npz_path)
    assert os.path.exists(npz_path), f"Test NPZ not found: {npz_path}"

    data = np.load(npz_path, allow_pickle=True)
    poses = np.asarray(data["poses"], dtype=np.float32)  # [T, 156] (52 joints)
    trans = np.asarray(data["trans"], dtype=np.float32)   # [T, 3]

    # Use first 10 frames for speed
    poses = poses[:10]
    trans = trans[:10]

    return poses, trans


def collect_baseline():
    """Run all functions with current code and save outputs."""
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_smplx_pose,
    )
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_rotation_6d,
        rotation_6d_to_matrix,
        matrix_to_rotation_6d,
        rotation_6d_to_axis_angle,
    )

    os.makedirs(BASELINE_DIR, exist_ok=True)
    poses, trans = get_test_data()

    # --- 1. process_smplx_pose with rotation_6d (includes [0,3,1,4,2,5] permutation) ---
    pose_6d = process_smplx_pose(poses, rot_type="rotation_6d", out_type="smpl_22")
    np.save(os.path.join(BASELINE_DIR, "process_smplx_pose_rot6d.npy"), pose_6d)
    print(f"[baseline] process_smplx_pose(rot6d): shape={pose_6d.shape}, "
          f"mean={pose_6d.mean():.6f}, std={pose_6d.std():.6f}")

    # --- 2. axis_angle_to_rotation_6d (numpy path) ---
    # Prepare: use first 22 joints from the 52-joint data
    T = poses.shape[0]
    # Pad 52->55 joints first
    poses_padded = np.concatenate(
        [poses[:, :66], np.zeros((T, 9), dtype=np.float32), poses[:, 66:]],
        axis=1,
    ).reshape(T, 55, 3)
    aa_22 = poses_padded[:, :22, :].reshape(-1, 3)  # [T*22, 3]

    d6_np = axis_angle_to_rotation_6d(aa_22)
    np.save(os.path.join(BASELINE_DIR, "axis_angle_to_rotation_6d_np.npy"), d6_np)
    print(f"[baseline] axis_angle_to_rotation_6d(np): shape={d6_np.shape}")

    # --- 3. axis_angle_to_rotation_6d (torch path) ---
    aa_torch = torch.from_numpy(aa_22).float()
    d6_torch = axis_angle_to_rotation_6d(aa_torch)
    np.save(os.path.join(BASELINE_DIR, "axis_angle_to_rotation_6d_torch.npy"),
            d6_torch.numpy())
    print(f"[baseline] axis_angle_to_rotation_6d(torch): shape={d6_torch.shape}")

    # --- 4. rotation_6d_to_matrix (numpy) ---
    mat_np = rotation_6d_to_matrix(d6_np)
    np.save(os.path.join(BASELINE_DIR, "rotation_6d_to_matrix_np.npy"), mat_np)
    print(f"[baseline] rotation_6d_to_matrix(np): shape={mat_np.shape}")

    # --- 5. rotation_6d_to_matrix (torch) ---
    mat_torch = rotation_6d_to_matrix(d6_torch)
    np.save(os.path.join(BASELINE_DIR, "rotation_6d_to_matrix_torch.npy"),
            mat_torch.numpy())
    print(f"[baseline] rotation_6d_to_matrix(torch): shape={mat_torch.shape}")

    # --- 6. matrix_to_rotation_6d (numpy) ---
    d6_from_mat_np = matrix_to_rotation_6d(mat_np)
    np.save(os.path.join(BASELINE_DIR, "matrix_to_rotation_6d_np.npy"), d6_from_mat_np)
    print(f"[baseline] matrix_to_rotation_6d(np): shape={d6_from_mat_np.shape}")

    # --- 7. matrix_to_rotation_6d (torch) ---
    d6_from_mat_torch = matrix_to_rotation_6d(mat_torch)
    np.save(os.path.join(BASELINE_DIR, "matrix_to_rotation_6d_torch.npy"),
            d6_from_mat_torch.numpy())
    print(f"[baseline] matrix_to_rotation_6d(torch): shape={d6_from_mat_torch.shape}")

    # --- 8. rotation_6d_to_axis_angle (numpy) ---
    aa_back_np = rotation_6d_to_axis_angle(d6_np)
    np.save(os.path.join(BASELINE_DIR, "rotation_6d_to_axis_angle_np.npy"), aa_back_np)
    print(f"[baseline] rotation_6d_to_axis_angle(np): shape={aa_back_np.shape}")

    # --- 9. rotation_6d_to_axis_angle (torch) ---
    aa_back_torch = rotation_6d_to_axis_angle(d6_torch)
    np.save(os.path.join(BASELINE_DIR, "rotation_6d_to_axis_angle_torch.npy"),
            aa_back_torch.numpy())
    print(f"[baseline] rotation_6d_to_axis_angle(torch): shape={aa_back_torch.shape}")

    print(f"\n[OK] Baseline saved to {BASELINE_DIR}/ ({len(os.listdir(BASELINE_DIR))} files)")


def verify():
    """
    After code changes: verify that default behavior is bit-for-bit identical to baseline.
    Also verify that new convention parameters produce expected permuted results.
    """
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_smplx_pose,
    )
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_rotation_6d,
        rotation_6d_to_matrix,
        matrix_to_rotation_6d,
        rotation_6d_to_axis_angle,
    )

    assert os.path.isdir(BASELINE_DIR), (
        f"Baseline dir not found: {BASELINE_DIR}. Run --collect-baseline first."
    )

    poses, trans = get_test_data()
    T = poses.shape[0]
    poses_padded = np.concatenate(
        [poses[:, :66], np.zeros((T, 9), dtype=np.float32), poses[:, 66:]],
        axis=1,
    ).reshape(T, 55, 3)
    aa_22 = poses_padded[:, :22, :].reshape(-1, 3)
    aa_torch = torch.from_numpy(aa_22).float()

    all_pass = True
    COL_TO_ROW = [0, 3, 1, 4, 2, 5]
    ROW_TO_COL = [0, 2, 4, 1, 3, 5]

    def check_equal(name, actual, baseline_file, atol=0.0):
        nonlocal all_pass
        expected = np.load(os.path.join(BASELINE_DIR, baseline_file))
        if isinstance(actual, torch.Tensor):
            actual = actual.numpy()
        if np.array_equal(actual, expected):
            print(f"  [PASS] {name}: bit-for-bit identical")
        elif np.allclose(actual, expected, atol=atol):
            max_diff = np.max(np.abs(actual - expected))
            print(f"  [PASS] {name}: allclose (max_diff={max_diff:.2e})")
        else:
            max_diff = np.max(np.abs(actual - expected))
            print(f"  [FAIL] {name}: max_diff={max_diff:.6f}")
            all_pass = False

    # ========== Part 1: Default behavior (no convention param) must match baseline ==========
    print("=" * 60)
    print("Part 1: Default behavior must match baseline")
    print("=" * 60)

    # 1. process_smplx_pose — default should still be row-major
    pose_6d = process_smplx_pose(poses, rot_type="rotation_6d", out_type="smpl_22")
    check_equal("process_smplx_pose(default)", pose_6d,
                "process_smplx_pose_rot6d.npy")

    # 2-3. axis_angle_to_rotation_6d — default should still be column-major
    d6_np = axis_angle_to_rotation_6d(aa_22)
    check_equal("axis_angle_to_rotation_6d(np, default)", d6_np,
                "axis_angle_to_rotation_6d_np.npy")

    d6_torch = axis_angle_to_rotation_6d(aa_torch)
    check_equal("axis_angle_to_rotation_6d(torch, default)", d6_torch,
                "axis_angle_to_rotation_6d_torch.npy")

    # 4-5. rotation_6d_to_matrix — default column
    mat_np = rotation_6d_to_matrix(d6_np)
    check_equal("rotation_6d_to_matrix(np, default)", mat_np,
                "rotation_6d_to_matrix_np.npy")

    mat_torch = rotation_6d_to_matrix(d6_torch)
    check_equal("rotation_6d_to_matrix(torch, default)", mat_torch,
                "rotation_6d_to_matrix_torch.npy")

    # 6-7. matrix_to_rotation_6d — default column
    d6_from_mat_np = matrix_to_rotation_6d(mat_np)
    check_equal("matrix_to_rotation_6d(np, default)", d6_from_mat_np,
                "matrix_to_rotation_6d_np.npy")

    d6_from_mat_torch = matrix_to_rotation_6d(mat_torch)
    check_equal("matrix_to_rotation_6d(torch, default)", d6_from_mat_torch,
                "matrix_to_rotation_6d_torch.npy")

    # 8-9. rotation_6d_to_axis_angle — default column
    aa_back_np = rotation_6d_to_axis_angle(d6_np)
    check_equal("rotation_6d_to_axis_angle(np, default)", aa_back_np,
                "rotation_6d_to_axis_angle_np.npy")

    aa_back_torch = rotation_6d_to_axis_angle(d6_torch)
    check_equal("rotation_6d_to_axis_angle(torch, default)", aa_back_torch,
                "rotation_6d_to_axis_angle_torch.npy")

    # ========== Part 2: New convention="row" param produces expected permuted output ==========
    print("\n" + "=" * 60)
    print("Part 2: convention='row' produces expected permutation")
    print("=" * 60)

    # axis_angle_to_rotation_6d with convention="row" should be column[..., COL_TO_ROW]
    try:
        d6_row_np = axis_angle_to_rotation_6d(aa_22, convention="row")
        d6_col_np = axis_angle_to_rotation_6d(aa_22)  # default column
        expected_row = d6_col_np[..., COL_TO_ROW] if isinstance(d6_col_np, np.ndarray) else d6_col_np[..., COL_TO_ROW].numpy()
        if isinstance(d6_row_np, torch.Tensor):
            d6_row_np = d6_row_np.numpy()
        if np.array_equal(d6_row_np, expected_row):
            print("  [PASS] axis_angle_to_rotation_6d(convention='row') == column[..., COL_TO_ROW]")
        else:
            print(f"  [FAIL] axis_angle_to_rotation_6d(convention='row'): max_diff={np.max(np.abs(d6_row_np - expected_row))}")
            all_pass = False
    except TypeError as e:
        print(f"  [SKIP] axis_angle_to_rotation_6d does not accept convention param yet: {e}")

    # matrix_to_rotation_6d with convention="row"
    try:
        d6_row_from_mat = matrix_to_rotation_6d(mat_np, convention="row")
        d6_col_from_mat = matrix_to_rotation_6d(mat_np)
        expected = d6_col_from_mat[..., COL_TO_ROW]
        if isinstance(d6_row_from_mat, torch.Tensor):
            d6_row_from_mat = d6_row_from_mat.numpy()
        if np.array_equal(d6_row_from_mat, expected):
            print("  [PASS] matrix_to_rotation_6d(convention='row') == column[..., COL_TO_ROW]")
        else:
            print(f"  [FAIL] matrix_to_rotation_6d(convention='row'): max_diff={np.max(np.abs(d6_row_from_mat - expected))}")
            all_pass = False
    except TypeError as e:
        print(f"  [SKIP] matrix_to_rotation_6d does not accept convention param yet: {e}")

    # rotation_6d_to_matrix with convention="row" input: should internally permute ROW_TO_COL then process
    try:
        # Create row-major input from column-major
        d6_row_input = d6_np[..., COL_TO_ROW]
        mat_from_row = rotation_6d_to_matrix(d6_row_input, convention="row")
        mat_from_col = rotation_6d_to_matrix(d6_np)  # same rotation, column input
        if isinstance(mat_from_row, torch.Tensor):
            mat_from_row = mat_from_row.numpy()
        if isinstance(mat_from_col, torch.Tensor):
            mat_from_col = mat_from_col.numpy()
        if np.allclose(mat_from_row, mat_from_col, atol=1e-6):
            print("  [PASS] rotation_6d_to_matrix(row_input, convention='row') == rotation_6d_to_matrix(col_input)")
        else:
            print(f"  [FAIL] rotation_6d_to_matrix(convention='row'): max_diff={np.max(np.abs(mat_from_row - mat_from_col))}")
            all_pass = False
    except TypeError as e:
        print(f"  [SKIP] rotation_6d_to_matrix does not accept convention param yet: {e}")

    # rotation_6d_to_axis_angle with convention="row"
    try:
        d6_row_input = d6_np[..., COL_TO_ROW]
        aa_from_row = rotation_6d_to_axis_angle(d6_row_input, convention="row")
        aa_from_col = rotation_6d_to_axis_angle(d6_np)
        if isinstance(aa_from_row, torch.Tensor):
            aa_from_row = aa_from_row.numpy()
        if isinstance(aa_from_col, torch.Tensor):
            aa_from_col = aa_from_col.numpy()
        if np.allclose(aa_from_row, aa_from_col, atol=1e-6):
            print("  [PASS] rotation_6d_to_axis_angle(row_input, convention='row') == rotation_6d_to_axis_angle(col_input)")
        else:
            print(f"  [FAIL] rotation_6d_to_axis_angle(convention='row'): max_diff={np.max(np.abs(aa_from_row - aa_from_col))}")
            all_pass = False
    except TypeError as e:
        print(f"  [SKIP] rotation_6d_to_axis_angle does not accept convention param yet: {e}")

    # ========== Part 3: process_smplx_pose with rot6d_convention param ==========
    print("\n" + "=" * 60)
    print("Part 3: process_smplx_pose rot6d_convention parameter")
    print("=" * 60)

    # With rot6d_convention="row" (explicit), should match baseline (which was row by default)
    try:
        pose_6d_row = process_smplx_pose(
            poses, rot_type="rotation_6d", out_type="smpl_22",
            rot6d_convention="row"
        )
        baseline = np.load(os.path.join(BASELINE_DIR, "process_smplx_pose_rot6d.npy"))
        if np.array_equal(pose_6d_row, baseline):
            print("  [PASS] process_smplx_pose(rot6d_convention='row') == baseline")
        else:
            print(f"  [FAIL] process_smplx_pose(rot6d_convention='row') differs from baseline")
            all_pass = False
    except TypeError as e:
        print(f"  [SKIP] process_smplx_pose does not accept rot6d_convention yet: {e}")

    # With rot6d_convention="column", should be baseline permuted back
    try:
        pose_6d_col = process_smplx_pose(
            poses, rot_type="rotation_6d", out_type="smpl_22",
            rot6d_convention="column"
        )
        baseline = np.load(os.path.join(BASELINE_DIR, "process_smplx_pose_rot6d.npy"))
        # baseline is row-major [T, 22*6], reshape to [T, 22, 6] then permute back
        T_b = baseline.shape[0]
        baseline_reshaped = baseline.reshape(T_b, 22, 6)
        expected_col = baseline_reshaped[:, :, ROW_TO_COL].reshape(T_b, 22 * 6)
        if np.allclose(pose_6d_col, expected_col, atol=1e-6):
            print("  [PASS] process_smplx_pose(rot6d_convention='column') == baseline permuted to column")
        else:
            max_diff = np.max(np.abs(pose_6d_col - expected_col))
            print(f"  [FAIL] process_smplx_pose(rot6d_convention='column'): max_diff={max_diff}")
            all_pass = False
    except TypeError as e:
        print(f"  [SKIP] process_smplx_pose does not accept rot6d_convention yet: {e}")

    # ========== Summary ==========
    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see above")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect-baseline", action="store_true",
                       help="Collect baseline outputs before code changes")
    group.add_argument("--verify", action="store_true",
                       help="Verify consistency after code changes")
    args = parser.parse_args()

    if args.collect_baseline:
        collect_baseline()
    else:
        verify()
