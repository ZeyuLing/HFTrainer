#!/usr/bin/env python3
"""Check shoulder heights in existing KIMODO NPZ files to detect collapse."""
import numpy as np
import glob
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SMPL22 joint indices
CHEST = 9        # spine3
L_SHOULDER = 16  # left_shoulder
R_SHOULDER = 17  # right_shoulder
L_ELBOW = 18
R_ELBOW = 19

def check_dir(npz_dir, label, max_files=20):
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))[:max_files]
    if not files:
        print(f"  [{label}] No NPZ files found in {npz_dir}")
        return

    print(f"\n=== {label} ({len(files)} files) ===")
    header = "{:<12s} {:>10s} {:>10s} {:>10s} {:>12s} {:>12s} {:>10s} {:>10s}".format(
        "File", "Chest_Y", "LSho_Y", "RSho_Y", "LSho-Chest", "RSho-Chest", "LUpperArm", "RUpperArm")
    print(header)
    print("-" * len(header))

    collapse_count = 0
    all_l_rel = []
    all_r_rel = []
    all_l_upper = []
    all_r_upper = []

    for f in files:
        npz = np.load(f, allow_pickle=True)
        pos = npz["positions"]  # (T, 22, 3)

        chest_y = pos[:, CHEST, 1].mean()
        l_sho_y = pos[:, L_SHOULDER, 1].mean()
        r_sho_y = pos[:, R_SHOULDER, 1].mean()
        l_rel = l_sho_y - chest_y
        r_rel = r_sho_y - chest_y

        l_upper = np.linalg.norm(pos[:, L_ELBOW] - pos[:, L_SHOULDER], axis=-1).mean()
        r_upper = np.linalg.norm(pos[:, R_ELBOW] - pos[:, R_SHOULDER], axis=-1).mean()

        all_l_rel.append(l_rel)
        all_r_rel.append(r_rel)
        all_l_upper.append(l_upper)
        all_r_upper.append(r_upper)

        fname = os.path.basename(f)
        flag = ""
        if l_rel < -0.03 or r_rel < -0.03:
            flag = " *** COLLAPSED ***"
            collapse_count += 1

        row = "{:<12s} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.4f} {:>12.4f} {:>10.4f} {:>10.4f}{}".format(
            fname, chest_y, l_sho_y, r_sho_y, l_rel, r_rel, l_upper, r_upper, flag)
        print(row)

    print("\nSummary:")
    print("  Collapse count: {}/{}".format(collapse_count, len(files)))
    print("  Mean LShoulder-Chest: {:.4f}m".format(np.mean(all_l_rel)))
    print("  Mean RShoulder-Chest: {:.4f}m".format(np.mean(all_r_rel)))
    print("  Mean L upper arm len: {:.4f}m".format(np.mean(all_l_upper)))
    print("  Mean R upper arm len: {:.4f}m".format(np.mean(all_r_upper)))
    print("  Expected: shoulders ~0.02-0.05m above chest, upper arm ~0.25-0.30m")


def main():
    os.chdir(PROJECT_ROOT)

    # Check the main eval_8082_refresh directory (used by score_m2m)
    base = "work_dirs/eval_8082_refresh_20260501/kimodo"
    if os.path.isdir(base):
        subdirs = sorted(os.listdir(base))
        for sd in subdirs[:6]:  # check first 6 subdirs
            npz_dir_candidates = glob.glob(os.path.join(base, sd, "*/npz"))
            for nd in npz_dir_candidates[:1]:
                check_dir(nd, sd)
    else:
        print("eval_8082_refresh dir not found, checking alternatives...")

    # Also check the parallel_e3 directory
    base2 = "work_dirs/parallel_e3_20260426_1730/kimodo/uncond"
    if os.path.isdir(base2):
        subdirs = sorted(os.listdir(base2))
        for sd in subdirs[:3]:
            nd = os.path.join(base2, sd, "npz")
            if os.path.isdir(nd):
                check_dir(nd, "parallel_" + sd)

    # Check kimodo_swin_fix
    base3 = "work_dirs/kimodo_swin_fix_20260430"
    if os.path.isdir(base3):
        for sd in sorted(os.listdir(base3))[:3]:
            full = os.path.join(base3, sd)
            if not os.path.isdir(full):
                continue
            npz_dirs = glob.glob(os.path.join(full, "*/npz"))
            for nd in npz_dirs[:1]:
                check_dir(nd, "swin_" + sd)


if __name__ == "__main__":
    main()
