#!/usr/bin/env python3
"""Diagnose KIMODO mesh rendering pipeline: check if bone quaternions
produce correct shoulder positions vs raw posed_joints.

This tests the EXACT same code path as score_m2m's mesh viewer:
  NPZ → _smpl_from_kimodo_lbs → compute_bone_quaternions → Three.js SkinnedMesh

If the roundtrip (global_rot → local_rot_30 → expand_77 → FK → local_rot_77 → quat)
introduces shoulder errors, the mesh viewer will show collapse even though the
raw posed_joints are correct.
"""
import sys
import os
import numpy as np
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "motion_annot_web", "score_m2m"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "motion_annot_web", "score_m2m", "soma_model"))

from utils_soma import SOMASKEL30_IN_NVSKEL77, SOMASKEL30_PARENT_INDICES, NVSKEL77_PARENT_INDICES

# SOMA77 shoulder-related joints
SOMA77_HIPS = 0
SOMA77_SPINE1 = 67
SOMA77_SPINE2 = 72
SOMA77_CHEST = 3   # Spine3
SOMA77_LEFT_ARM = 12
SOMA77_RIGHT_ARM = 40
SOMA77_LEFT_FOREARM = 13
SOMA77_RIGHT_FOREARM = 41

# SOMASKEL30 indices (for reference)
S30_INDICES = np.array(SOMASKEL30_IN_NVSKEL77, dtype=np.int64)


def test_roundtrip(npz_path, max_frames=30):
    """Test the full bone quaternion pipeline roundtrip.

    Compares:
    - Original posed_joints from NPZ (ground truth)
    - FK-reconstructed positions from bone quaternions (what mesh viewer shows)
    """
    data = np.load(npz_path, allow_pickle=True)

    gr_full = data["global_rot_mats"].astype(np.float32)  # (T, 77, 3, 3)
    pj_full = data["posed_joints"].astype(np.float32)     # (T, 77, 3)

    T = min(gr_full.shape[0], max_frames)
    gr_full = gr_full[:T]
    pj_full = pj_full[:T]

    # === Path 1: Raw posed_joints (skeleton viewer) ===
    orig_chest_y = pj_full[:, SOMA77_CHEST, 1].mean()
    orig_l_arm_y = pj_full[:, SOMA77_LEFT_ARM, 1].mean()
    orig_r_arm_y = pj_full[:, SOMA77_RIGHT_ARM, 1].mean()

    # === Path 2: Bone quaternion pipeline (mesh viewer) ===
    # Exactly mirrors _smpl_from_kimodo_lbs in motion_utils.py
    gr_30 = gr_full[:, S30_INDICES, :, :]  # (T, 30, 3, 3)
    pj_30 = pj_full[:, S30_INDICES, :]     # (T, 30, 3)
    root_positions = pj_full[:, 0, :]       # (T, 3) Hips

    # Import the exact same functions used by score_m2m
    from soma_forward import (
        _load_skin_model, _grm30_to_lrm30, _fk_77_numpy,
        compute_local_rotations_77
    )

    model = _load_skin_model()
    s30_in_77 = model["s30_in_77"]

    # Step 1: global → local (somaskel30)
    lrm_30 = _grm30_to_lrm30(gr_30)

    # Step 2: expand 30 → 77
    lrm_77 = np.tile(np.eye(3, dtype=np.float32), (T, 77, 1, 1))
    lrm_77[:, s30_in_77] = lrm_30

    # Step 3: FK → global_rot_77 + posed_joints_77
    fk_global_rot, fk_posed_joints = _fk_77_numpy(lrm_77, root_positions, model)

    # Compare FK-reconstructed vs original
    fk_chest_y = fk_posed_joints[:, SOMA77_CHEST, 1].mean()
    fk_l_arm_y = fk_posed_joints[:, SOMA77_LEFT_ARM, 1].mean()
    fk_r_arm_y = fk_posed_joints[:, SOMA77_RIGHT_ARM, 1].mean()

    # Also check global rotation matrices
    gr_diff_chest = np.linalg.norm(fk_global_rot[:, SOMA77_CHEST] - gr_full[:, SOMA77_CHEST], axis=(-2, -1)).mean()
    gr_diff_l_arm = np.linalg.norm(fk_global_rot[:, SOMA77_LEFT_ARM] - gr_full[:, SOMA77_LEFT_ARM], axis=(-2, -1)).mean()
    gr_diff_r_arm = np.linalg.norm(fk_global_rot[:, SOMA77_RIGHT_ARM] - gr_full[:, SOMA77_RIGHT_ARM], axis=(-2, -1)).mean()

    # Position difference for ALL 77 joints
    pos_diff = np.linalg.norm(fk_posed_joints - pj_full, axis=-1)  # (T, 77)
    mean_diff_all = pos_diff.mean()
    max_diff_all = pos_diff.max()

    # Shoulder-specific
    shoulder_joints = [SOMA77_CHEST, SOMA77_LEFT_ARM, SOMA77_RIGHT_ARM, SOMA77_LEFT_FOREARM, SOMA77_RIGHT_FOREARM]
    shoulder_diffs = pos_diff[:, shoulder_joints].mean(axis=0)

    fname = os.path.basename(npz_path)
    print(f"\n{'='*70}")
    print(f"File: {fname}  ({T} frames)")
    print(f"{'='*70}")

    print(f"\n--- Position comparison (original vs FK-reconstructed) ---")
    print(f"  All joints: mean_diff={mean_diff_all:.6f}m, max_diff={max_diff_all:.6f}m")
    for j, name in zip(shoulder_joints, ["Chest/Spine3", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm"]):
        print(f"  {name:15s}: mean_diff={shoulder_diffs[shoulder_joints.index(j)]:.6f}m")

    print(f"\n--- Shoulder Y positions ---")
    print(f"  {'':15s} {'Original':>10s} {'FK-recon':>10s} {'Diff':>10s}")
    print(f"  {'Chest_Y':15s} {orig_chest_y:10.4f} {fk_chest_y:10.4f} {fk_chest_y-orig_chest_y:10.6f}")
    print(f"  {'LArm_Y':15s} {orig_l_arm_y:10.4f} {fk_l_arm_y:10.4f} {fk_l_arm_y-orig_l_arm_y:10.6f}")
    print(f"  {'RArm_Y':15s} {orig_r_arm_y:10.4f} {fk_r_arm_y:10.4f} {fk_r_arm_y-orig_r_arm_y:10.6f}")
    print(f"  LArm-Chest(orig): {orig_l_arm_y-orig_chest_y:.4f}m  LArm-Chest(FK): {fk_l_arm_y-fk_chest_y:.4f}m")
    print(f"  RArm-Chest(orig): {orig_r_arm_y-orig_chest_y:.4f}m  RArm-Chest(FK): {fk_r_arm_y-fk_chest_y:.4f}m")

    print(f"\n--- Global rotation mat diff (Frobenius norm) ---")
    print(f"  Chest:    {gr_diff_chest:.8f}")
    print(f"  LeftArm:  {gr_diff_l_arm:.8f}")
    print(f"  RightArm: {gr_diff_r_arm:.8f}")

    # Check per-frame worst case for shoulders
    l_arm_y_diff_per_frame = fk_posed_joints[:, SOMA77_LEFT_ARM, 1] - pj_full[:, SOMA77_LEFT_ARM, 1]
    r_arm_y_diff_per_frame = fk_posed_joints[:, SOMA77_RIGHT_ARM, 1] - pj_full[:, SOMA77_RIGHT_ARM, 1]
    print(f"\n--- Per-frame shoulder Y diff (FK - original) ---")
    print(f"  LArm_Y: min={l_arm_y_diff_per_frame.min():.6f}, max={l_arm_y_diff_per_frame.max():.6f}, std={l_arm_y_diff_per_frame.std():.6f}")
    print(f"  RArm_Y: min={r_arm_y_diff_per_frame.min():.6f}, max={r_arm_y_diff_per_frame.max():.6f}, std={r_arm_y_diff_per_frame.std():.6f}")

    collapse = (mean_diff_all > 0.01)
    if collapse:
        print(f"\n  *** WARNING: FK roundtrip has >1cm mean error! Mesh may look different from skeleton ***")
    else:
        print(f"\n  OK: FK roundtrip error < 1cm — mesh rendering should match skeleton")

    return mean_diff_all, max_diff_all


def main():
    os.chdir(PROJECT_ROOT)

    # Check production KIMODO NPZs used by score_m2m
    base = "work_dirs/eval_8082_refresh_20260501/kimodo"
    if not os.path.isdir(base):
        print(f"ERROR: {base} not found")
        return

    # Test a few files from different subdirs
    test_files = []
    for sd in sorted(os.listdir(base))[:4]:
        npz_dirs = glob.glob(os.path.join(base, sd, "*/npz"))
        for nd in npz_dirs[:1]:
            files = sorted(glob.glob(os.path.join(nd, "*.npz")))[:3]
            test_files.extend(files)

    if not test_files:
        print("No NPZ files found")
        return

    print(f"Testing {len(test_files)} files through bone quaternion roundtrip...")

    errors = []
    for f in test_files:
        try:
            mean_err, max_err = test_roundtrip(f)
            errors.append((os.path.basename(f), mean_err, max_err))
        except Exception as e:
            print(f"\nERROR processing {f}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(errors)} files tested")
    print(f"{'='*70}")
    print(f"  {'File':<30s} {'Mean err':>12s} {'Max err':>12s}")
    for fname, me, mx in errors:
        flag = " *** LARGE ***" if me > 0.01 else ""
        print(f"  {fname:<30s} {me:>12.6f}m {mx:>12.6f}m{flag}")

    mean_all = np.mean([e[1] for e in errors])
    max_all = max([e[2] for e in errors])
    print(f"\n  Overall: mean_err={mean_all:.6f}m, max_err={max_all:.6f}m")

    if mean_all > 0.01:
        print("\n  VERDICT: Bone quaternion pipeline introduces >1cm error.")
        print("  The mesh viewer may show shoulder collapse even though NPZ data is correct.")
        print("  This is a RENDERING PIPELINE issue, NOT a retargeting issue.")
    else:
        print("\n  VERDICT: Bone quaternion roundtrip is accurate (<1cm).")
        print("  Mesh rendering should faithfully represent the NPZ data.")
        print("  No need to re-run inference or re-retarget.")


if __name__ == "__main__":
    main()
