"""
Verify that normalizing ROW-MAJOR rot6d data with COLUMN-MAJOR rot6d statistics
produces extreme out-of-distribution values.

Background:
- Stats file (smplx55_stats_hymotion_aug.json) stores rot6d in COLUMN-MAJOR format:
  [R00, R10, R20, R01, R11, R21]
- LoadSmplx55's process_smplx_pose produces ROW-MAJOR format:
  [R00, R01, R10, R11, R20, R21]
- Column→Row permutation (applied by LoadSmplx55): [0, 2, 4, 1, 3, 5]
  i.e. col_major[[0,3,1,4,2,5]] -> row_major
- Row→Column permutation: [0, 3, 1, 4, 2, 5]

If training data is ROW-MAJOR but stats are COLUMN-MAJOR, normalization will be
mismatched on dimensions 1-4 (dims 0 and 5 happen to be correct since they map
to themselves in the permutation for 2-column case... actually let's check).
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hftrainer.datasets.motion.motionhub.transforms.load_smplx import process_smplx_pose


def main():
    # =========================================================================
    # 1. Load stats JSON (COLUMN-MAJOR rot6d)
    # =========================================================================
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data/statistic/smplx55_stats_hymotion_aug.json"
    )
    with open(stats_path, "r") as f:
        stats = json.load(f)

    # Extract global_orient rot6d stats (6 dims, column-major)
    go_mean_colmaj = np.array(stats["global_orient"]["rotation_6d"]["mean"], dtype=np.float32)
    go_std_colmaj = np.array(stats["global_orient"]["rotation_6d"]["std"], dtype=np.float32)

    # Extract body_pose rot6d stats (21 joints * 6 = 126 dims, column-major)
    bp_mean_colmaj = np.array(stats["body_pose"]["rotation_6d"]["mean"], dtype=np.float32)
    bp_std_colmaj = np.array(stats["body_pose"]["rotation_6d"]["std"], dtype=np.float32)

    print("=" * 80)
    print("STATS FORMAT MISMATCH VERIFICATION")
    print("=" * 80)
    print(f"\nStats file: {stats_path}")
    print(f"Stats format: COLUMN-MAJOR rot6d [R00, R10, R20, R01, R11, R21]")
    print(f"Data format (from LoadSmplx55): ROW-MAJOR rot6d [R00, R01, R10, R11, R20, R21]")
    print(f"\nglobal_orient stats shape: mean={go_mean_colmaj.shape}, std={go_std_colmaj.shape}")
    print(f"body_pose stats shape: mean={bp_mean_colmaj.shape}, std={bp_std_colmaj.shape}")

    # =========================================================================
    # 2. Load a real motion file via process_smplx_pose (produces ROW-MAJOR)
    # =========================================================================
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data/motionhub/amass_sup/smplx_55"
    )
    npz_path = os.path.join(base_dir, "CNRS/288/12_L_2_stageii_0_360.npz")
    print(f"\nLoading motion file: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    poses = data["poses"]  # [T, 165] axis-angle
    T = poses.shape[0]
    print(f"Motion length: {T} frames")

    # Process with process_smplx_pose -> ROW-MAJOR rot6d
    # For smpl_22: joint 0 = global_orient, joints 1-21 = body_pose
    pose_rot6d_rowmaj = process_smplx_pose(poses, rot_type="rotation_6d", out_type="smpl_22")
    # Shape: [T, 22*6] = [T, 132]
    print(f"Processed pose shape: {pose_rot6d_rowmaj.shape} (ROW-MAJOR rot6d)")

    # Split into global_orient and body_pose
    go_data_rowmaj = pose_rot6d_rowmaj[:, :6]       # [T, 6]
    bp_data_rowmaj = pose_rot6d_rowmaj[:, 6:]       # [T, 126]

    # =========================================================================
    # 3. Define permutations
    # =========================================================================
    # Column-major to Row-major permutation (per 6-dim block):
    # col[0,3,1,4,2,5] -> row  (this is what LoadSmplx55 applies)
    col_to_row = [0, 3, 1, 4, 2, 5]
    # Row-major to Column-major:
    # row[0,2,4,1,3,5] -> col
    row_to_col = [0, 2, 4, 1, 3, 5]

    def permute_stats_to_rowmaj(mean_col, std_col, n_joints):
        """Permute column-major stats to row-major ordering."""
        mean_row = np.zeros_like(mean_col)
        std_row = np.zeros_like(std_col)
        for j in range(n_joints):
            for i, src in enumerate(col_to_row):
                mean_row[j * 6 + i] = mean_col[j * 6 + src]
                std_row[j * 6 + i] = std_col[j * 6 + src]
        return mean_row, std_row

    def permute_data_to_colmaj(data_row, n_joints):
        """Permute row-major data to column-major ordering."""
        T = data_row.shape[0]
        data_col = np.zeros_like(data_row)
        for j in range(n_joints):
            for i, src in enumerate(row_to_col):
                data_col[:, j * 6 + i] = data_row[:, j * 6 + src]
        return data_col

    # =========================================================================
    # 4. MISMATCHED normalization: ROW-MAJOR data with COLUMN-MAJOR stats
    # =========================================================================
    print("\n" + "=" * 80)
    print("CASE 1: MISMATCHED NORMALIZATION (row-major data / column-major stats)")
    print("=" * 80)

    # Normalize global_orient
    go_norm_mismatch = (go_data_rowmaj - go_mean_colmaj) / go_std_colmaj
    # Normalize body_pose
    bp_norm_mismatch = (bp_data_rowmaj - bp_mean_colmaj) / bp_std_colmaj

    print("\n--- Global Orient (1 joint, 6 dims) ---")
    print(f"{'Dim':>4} | {'Col-Maj Label':>14} | {'Row-Maj Label':>14} | "
          f"{'Mean(norm)':>10} | {'Std(norm)':>10} | {'Max|norm|':>10} | {'%>3σ':>6}")
    print("-" * 85)

    col_labels = ["R00", "R10", "R20", "R01", "R11", "R21"]
    row_labels = ["R00", "R01", "R10", "R11", "R20", "R21"]

    extreme_count_go = 0
    for d in range(6):
        vals = go_norm_mismatch[:, d]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        max_abs = np.max(np.abs(vals))
        pct_extreme = 100.0 * np.mean(np.abs(vals) > 3.0)
        extreme_count_go += np.sum(np.abs(vals) > 3.0)
        flag = " *** EXTREME" if pct_extreme > 5.0 else ""
        print(f"{d:>4} | {col_labels[d]:>14} | {row_labels[d]:>14} | "
              f"{mean_v:>10.4f} | {std_v:>10.4f} | {max_abs:>10.4f} | {pct_extreme:>5.1f}%{flag}")

    print(f"\nTotal extreme samples (|z|>3): {extreme_count_go} / {T*6} = "
          f"{100.0*extreme_count_go/(T*6):.1f}%")

    print("\n--- Body Pose (21 joints, 126 dims) ---")
    print(f"{'Joint':>5} {'Dim':>4} | {'Col-Maj':>8} | {'Row-Maj':>8} | "
          f"{'Mean(norm)':>10} | {'Std(norm)':>10} | {'Max|norm|':>10} | {'%>3σ':>6}")
    print("-" * 90)

    extreme_count_bp = 0
    extreme_joints = []
    for j in range(21):
        for d in range(6):
            idx = j * 6 + d
            vals = bp_norm_mismatch[:, idx]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            max_abs = np.max(np.abs(vals))
            pct_extreme = 100.0 * np.mean(np.abs(vals) > 3.0)
            n_extreme = np.sum(np.abs(vals) > 3.0)
            extreme_count_bp += n_extreme
            if pct_extreme > 5.0:
                extreme_joints.append((j, d, mean_v, std_v, max_abs, pct_extreme))
                # Only print the extreme ones to keep output manageable
                print(f"  J{j:>2} d{d:>1} | {col_labels[d]:>8} | {row_labels[d]:>8} | "
                      f"{mean_v:>10.4f} | {std_v:>10.4f} | {max_abs:>10.4f} | {pct_extreme:>5.1f}% ***")

    print(f"\nTotal extreme body_pose samples (|z|>3): {extreme_count_bp} / {T*126} = "
          f"{100.0*extreme_count_bp/(T*126):.1f}%")
    print(f"Joints/dims with >5% extreme: {len(extreme_joints)} / 126")

    # =========================================================================
    # 5. CORRECT normalization: permute stats to match row-major data
    # =========================================================================
    print("\n" + "=" * 80)
    print("CASE 2: CORRECT NORMALIZATION (stats permuted to row-major)")
    print("=" * 80)

    go_mean_rowmaj, go_std_rowmaj = permute_stats_to_rowmaj(go_mean_colmaj, go_std_colmaj, 1)
    bp_mean_rowmaj, bp_std_rowmaj = permute_stats_to_rowmaj(bp_mean_colmaj, bp_std_colmaj, 21)

    go_norm_correct = (go_data_rowmaj - go_mean_rowmaj) / go_std_rowmaj
    bp_norm_correct = (bp_data_rowmaj - bp_mean_rowmaj) / bp_std_rowmaj

    print("\n--- Global Orient (1 joint, 6 dims) ---")
    print(f"{'Dim':>4} | {'Label':>8} | {'Mean(norm)':>10} | {'Std(norm)':>10} | "
          f"{'Max|norm|':>10} | {'%>3σ':>6}")
    print("-" * 65)

    extreme_count_go_correct = 0
    for d in range(6):
        vals = go_norm_correct[:, d]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        max_abs = np.max(np.abs(vals))
        pct_extreme = 100.0 * np.mean(np.abs(vals) > 3.0)
        extreme_count_go_correct += np.sum(np.abs(vals) > 3.0)
        print(f"{d:>4} | {row_labels[d]:>8} | {mean_v:>10.4f} | {std_v:>10.4f} | "
              f"{max_abs:>10.4f} | {pct_extreme:>5.1f}%")

    print(f"\nTotal extreme samples (|z|>3): {extreme_count_go_correct} / {T*6} = "
          f"{100.0*extreme_count_go_correct/(T*6):.1f}%")

    print("\n--- Body Pose (21 joints) - Summary ---")
    extreme_count_bp_correct = 0
    max_pct_correct = 0
    for j in range(21):
        for d in range(6):
            idx = j * 6 + d
            vals = bp_norm_correct[:, idx]
            pct_extreme = 100.0 * np.mean(np.abs(vals) > 3.0)
            extreme_count_bp_correct += np.sum(np.abs(vals) > 3.0)
            max_pct_correct = max(max_pct_correct, pct_extreme)

    print(f"Total extreme body_pose samples (|z|>3): {extreme_count_bp_correct} / {T*126} = "
          f"{100.0*extreme_count_bp_correct/(T*126):.1f}%")
    print(f"Max per-dim extreme %: {max_pct_correct:.1f}%")

    # =========================================================================
    # 6. Direct comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    all_mismatch = np.concatenate([go_norm_mismatch.flatten(), bp_norm_mismatch.flatten()])
    all_correct = np.concatenate([go_norm_correct.flatten(), bp_norm_correct.flatten()])

    print(f"\n{'Metric':<35} | {'Mismatched':>12} | {'Correct':>12}")
    print("-" * 65)
    print(f"{'Overall mean of |normalized|':<35} | {np.mean(np.abs(all_mismatch)):>12.4f} | "
          f"{np.mean(np.abs(all_correct)):>12.4f}")
    print(f"{'Overall std of normalized':<35} | {np.std(all_mismatch):>12.4f} | "
          f"{np.std(all_correct):>12.4f}")
    print(f"{'Max |normalized| value':<35} | {np.max(np.abs(all_mismatch)):>12.4f} | "
          f"{np.max(np.abs(all_correct)):>12.4f}")
    print(f"{'% samples with |z| > 3':<35} | "
          f"{100.0*np.mean(np.abs(all_mismatch) > 3):>11.2f}% | "
          f"{100.0*np.mean(np.abs(all_correct) > 3):>11.2f}%")
    print(f"{'% samples with |z| > 5':<35} | "
          f"{100.0*np.mean(np.abs(all_mismatch) > 5):>11.2f}% | "
          f"{100.0*np.mean(np.abs(all_correct) > 5):>11.2f}%")
    print(f"{'% samples with |z| > 10':<35} | "
          f"{100.0*np.mean(np.abs(all_mismatch) > 10):>11.2f}% | "
          f"{100.0*np.mean(np.abs(all_correct) > 10):>11.2f}%")

    # =========================================================================
    # 7. Show dimension-by-dimension what's happening for global_orient
    # =========================================================================
    print("\n" + "=" * 80)
    print("DETAILED: Global Orient Dimension Mapping")
    print("=" * 80)
    print("\nColumn-major stats order: [R00, R10, R20, R01, R11, R21]")
    print("Row-major data order:    [R00, R01, R10, R11, R20, R21]")
    print("\nWhen we normalize row-major data[i] with col-major stats[i]:")
    print(f"{'Data dim':>10} | {'Data has':>8} | {'Stats for':>10} | {'Match?':>7} | "
          f"{'Data mean':>10} | {'Stats mean':>10} | {'Stats std':>10}")
    print("-" * 85)

    for i in range(6):
        data_label = row_labels[i]
        stats_label = col_labels[i]
        match = "YES" if data_label == stats_label else "NO"
        data_mean = np.mean(go_data_rowmaj[:, i])
        print(f"{i:>10} | {data_label:>8} | {stats_label:>10} | {match:>7} | "
              f"{data_mean:>10.4f} | {go_mean_colmaj[i]:>10.4f} | {go_std_colmaj[i]:>10.4f}")

    # =========================================================================
    # 8. Verdict
    # =========================================================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    mismatch_extreme_pct = 100.0 * np.mean(np.abs(all_mismatch) > 3)
    correct_extreme_pct = 100.0 * np.mean(np.abs(all_correct) > 3)

    if mismatch_extreme_pct > 5 * correct_extreme_pct:
        print(f"\n✗ CONFIRMED: Cross-normalization produces {mismatch_extreme_pct:.1f}% extreme "
              f"values vs {correct_extreme_pct:.1f}% with correct normalization.")
        print(f"  That's {mismatch_extreme_pct/max(correct_extreme_pct,0.001):.1f}x more extreme values!")
        print("\n  The column-major stats applied to row-major data cause dimensions to be")
        print("  normalized with wrong mean/std, producing out-of-distribution inputs to the model.")
    else:
        print(f"\n? Inconclusive: Mismatch={mismatch_extreme_pct:.2f}%, "
              f"Correct={correct_extreme_pct:.2f}%")
        print("  The permutation may not cause as much damage as expected for this sample.")


if __name__ == "__main__":
    main()
