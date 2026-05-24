"""
Diagnose PD oscillation in physics simulation vs kinematic reference.
Computes per-joint angular acceleration, jerk, and tracking error.
"""
import json
import numpy as np
from pathlib import Path

# === Config ===
KIN_DIR = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4/data/smpl_mesh/")
PHYS_DIR = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4/data/smpl_mesh_physics/")

CASES = {
    "v4_crouch_001": "BAD (crouch - oscillation)",
    "v4_walk_001": "GOOD (walk - stable)",
}

JOINT_NAMES = [
    "Root", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
]
NUM_JOINTS = 22
FPS = 30
DT = 1.0 / FPS

JOINT_GROUPS = {
    "Legs (Hip/Knee/Ankle)": [1, 2, 4, 5, 7, 8],
    "Feet": [10, 11],
    "Spine/Torso": [0, 3, 6, 9],
    "Head/Neck": [12, 15],
    "Shoulders/Collar": [13, 14, 16, 17],
    "Arms (Elbow/Wrist)": [18, 19, 20, 21],
}


def load_poses(json_path):
    """Load poses from mesh JSON. Returns (T, 22, 3) axis-angle array."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data["frames"]
    fps = data.get("fps", 30)

    poses_list = []
    for frame in frames:
        # Each frame is a list of body dicts
        body = frame[0]
        pose_flat = body["poses"][0]  # 156 floats
        # First 66 = 22 joints * 3 (axis-angle)
        joint_aa = np.array(pose_flat[:66]).reshape(22, 3)
        poses_list.append(joint_aa)

    poses = np.stack(poses_list, axis=0)  # (T, 22, 3)
    return poses, fps


def axis_angle_to_angle(aa):
    """Convert axis-angle (T, 22, 3) to angle magnitude (T, 22)."""
    return np.linalg.norm(aa, axis=-1)


def compute_angular_velocity(poses, dt):
    """
    Compute angular velocity from axis-angle poses.
    Uses finite differences on axis-angle representation.
    Returns (T-1, 22, 3) angular velocity vectors.
    """
    # Simple finite difference on axis-angle (works for small rotations between frames)
    dpose = np.diff(poses, axis=0)  # (T-1, 22, 3)
    ang_vel = dpose / dt
    return ang_vel


def compute_angular_acceleration(ang_vel, dt):
    """Compute angular acceleration from angular velocity. Returns (T-2, 22, 3)."""
    d_ang_vel = np.diff(ang_vel, axis=0)
    ang_acc = d_ang_vel / dt
    return ang_acc


def compute_jerk(ang_acc, dt):
    """Compute angular jerk from angular acceleration. Returns (T-3, 22, 3)."""
    d_ang_acc = np.diff(ang_acc, axis=0)
    jerk = d_ang_acc / dt
    return jerk


def compute_metrics(poses, dt):
    """Compute all dynamics metrics for a pose sequence."""
    ang_vel = compute_angular_velocity(poses, dt)
    ang_acc = compute_angular_acceleration(ang_vel, dt)
    jerk = compute_jerk(ang_acc, dt)

    # Magnitudes (per joint per frame)
    vel_mag = np.linalg.norm(ang_vel, axis=-1)      # (T-1, 22)
    acc_mag = np.linalg.norm(ang_acc, axis=-1)      # (T-2, 22)
    jerk_mag = np.linalg.norm(jerk, axis=-1)        # (T-3, 22)

    return {
        "ang_vel": ang_vel,
        "ang_acc": ang_acc,
        "jerk": jerk,
        "vel_mag": vel_mag,
        "acc_mag": acc_mag,
        "jerk_mag": jerk_mag,
    }


def compute_tracking_error(kin_poses, phys_poses):
    """
    Compute per-joint tracking error (axis-angle difference).
    Returns (T, 22) angle error in radians.
    """
    T = min(len(kin_poses), len(phys_poses))
    kin = kin_poses[:T]
    phys = phys_poses[:T]

    # Difference in axis-angle space (approximation for tracking error)
    diff = phys - kin  # (T, 22, 3)
    error = np.linalg.norm(diff, axis=-1)  # (T, 22) in radians
    return error


def print_separator(char="=", length=80):
    print(char * length)


def analyze_case(case_name, label):
    print_separator()
    print(f"  CASE: {case_name} — {label}")
    print_separator()

    kin_path = KIN_DIR / f"{case_name}.json"
    phys_path = PHYS_DIR / f"{case_name}.json"

    if not kin_path.exists():
        print(f"  [SKIP] Kinematic file not found: {kin_path}")
        return None
    if not phys_path.exists():
        print(f"  [SKIP] Physics file not found: {phys_path}")
        return None

    kin_poses, fps = load_poses(kin_path)
    phys_poses, _ = load_poses(phys_path)
    dt = 1.0 / fps

    print(f"  Kinematic frames: {len(kin_poses)}, Physics frames: {len(phys_poses)}, FPS: {fps}")

    # Compute metrics
    kin_metrics = compute_metrics(kin_poses, dt)
    phys_metrics = compute_metrics(phys_poses, dt)

    # Tracking error
    tracking_err = compute_tracking_error(kin_poses, phys_poses)

    # === Per-joint analysis ===
    print(f"\n{'─'*80}")
    print(f"  PER-JOINT ANGULAR ACCELERATION (rad/s²) — Mean ± Std")
    print(f"{'─'*80}")
    print(f"  {'Joint':<14} {'Kin Mean':>10} {'Phys Mean':>10} {'Ratio':>8} {'Phys Std':>10} {'Track Err':>10}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")

    results_per_joint = []
    for j in range(NUM_JOINTS):
        kin_acc_j = kin_metrics["acc_mag"][:, j]
        phys_acc_j = phys_metrics["acc_mag"][:, j]
        track_j = tracking_err[:, j]

        kin_mean = kin_acc_j.mean()
        phys_mean = phys_acc_j.mean()
        ratio = phys_mean / (kin_mean + 1e-8)
        phys_std = phys_acc_j.std()
        track_mean = np.degrees(track_j.mean())

        results_per_joint.append({
            "joint": JOINT_NAMES[j],
            "joint_idx": j,
            "kin_acc_mean": kin_mean,
            "phys_acc_mean": phys_mean,
            "ratio": ratio,
            "phys_acc_std": phys_std,
            "track_err_deg": track_mean,
        })

        flag = " ⚠️" if ratio > 1.5 else ""
        print(f"  {JOINT_NAMES[j]:<14} {kin_mean:>10.1f} {phys_mean:>10.1f} {ratio:>8.2f}x {phys_std:>10.1f} {track_mean:>9.1f}°{flag}")

    # === Per-joint JERK ===
    print(f"\n{'─'*80}")
    print(f"  PER-JOINT ANGULAR JERK (rad/s³) — Mean")
    print(f"{'─'*80}")
    print(f"  {'Joint':<14} {'Kin Mean':>12} {'Phys Mean':>12} {'Ratio':>8}")
    print(f"  {'─'*14} {'─'*12} {'─'*12} {'─'*8}")

    jerk_ratios = []
    for j in range(NUM_JOINTS):
        kin_jerk_j = kin_metrics["jerk_mag"][:, j]
        phys_jerk_j = phys_metrics["jerk_mag"][:, j]

        kin_mean = kin_jerk_j.mean()
        phys_mean = phys_jerk_j.mean()
        ratio = phys_mean / (kin_mean + 1e-8)
        jerk_ratios.append(ratio)

        flag = " ⚠️" if ratio > 1.5 else ""
        print(f"  {JOINT_NAMES[j]:<14} {kin_mean:>12.0f} {phys_mean:>12.0f} {ratio:>8.2f}x{flag}")

    # === Group analysis ===
    print(f"\n{'─'*80}")
    print(f"  JOINT GROUP SUMMARY — Acceleration Ratio (Phys/Kin)")
    print(f"{'─'*80}")

    for group_name, joint_indices in JOINT_GROUPS.items():
        group_ratios = [results_per_joint[j]["ratio"] for j in joint_indices]
        group_track = [results_per_joint[j]["track_err_deg"] for j in joint_indices]
        avg_ratio = np.mean(group_ratios)
        avg_track = np.mean(group_track)
        print(f"  {group_name:<28} Acc Ratio: {avg_ratio:.2f}x  |  Track Err: {avg_track:.1f}°")

    # === Top offenders ===
    print(f"\n{'─'*80}")
    print(f"  TOP 5 WORST OSCILLATION JOINTS (by acc ratio)")
    print(f"{'─'*80}")
    sorted_joints = sorted(results_per_joint, key=lambda x: x["ratio"], reverse=True)
    for i, r in enumerate(sorted_joints[:5]):
        print(f"  #{i+1}: {r['joint']:<14} ratio={r['ratio']:.2f}x  phys_acc={r['phys_acc_mean']:.1f}  track_err={r['track_err_deg']:.1f}°")

    # === Temporal analysis: oscillation frequency ===
    print(f"\n{'─'*80}")
    print(f"  OSCILLATION FREQUENCY ANALYSIS (sign changes in angular acceleration)")
    print(f"{'─'*80}")

    # Count zero-crossings in angular acceleration per joint
    for j in sorted([r["joint_idx"] for r in sorted_joints[:5]]):
        phys_acc_vec = phys_metrics["ang_acc"][:, j, :]  # (T-2, 3)
        kin_acc_vec = kin_metrics["ang_acc"][:, j, :]

        # Count sign changes in each axis, average across axes
        phys_crossings = 0
        kin_crossings = 0
        for axis in range(3):
            phys_signs = np.sign(phys_acc_vec[:, axis])
            kin_signs = np.sign(kin_acc_vec[:, axis])
            phys_crossings += np.sum(np.abs(np.diff(phys_signs)) > 0)
            kin_crossings += np.sum(np.abs(np.diff(kin_signs)) > 0)

        T_phys = len(phys_acc_vec)
        T_kin = len(kin_acc_vec)
        phys_freq = phys_crossings / (3 * T_phys / fps) if T_phys > 0 else 0
        kin_freq = kin_crossings / (3 * T_kin / fps) if T_kin > 0 else 0

        print(f"  {JOINT_NAMES[j]:<14} Phys oscillation freq: {phys_freq:.1f} Hz  |  Kin: {kin_freq:.1f} Hz  |  Ratio: {phys_freq/(kin_freq+1e-8):.2f}x")

    # === Correlation: tracking error vs acceleration ratio ===
    print(f"\n{'─'*80}")
    print(f"  CORRELATION: Tracking Error vs Oscillation")
    print(f"{'─'*80}")

    track_errs = [r["track_err_deg"] for r in results_per_joint]
    acc_ratios = [r["ratio"] for r in results_per_joint]
    correlation = np.corrcoef(track_errs, acc_ratios)[0, 1]
    print(f"  Pearson correlation (track_error vs acc_ratio): {correlation:.3f}")

    # Also check per-frame temporal correlation for worst joint
    worst_j = sorted_joints[0]["joint_idx"]
    T_min = min(len(tracking_err), len(phys_metrics["acc_mag"]))
    frame_track = tracking_err[:T_min, worst_j]
    frame_acc = phys_metrics["acc_mag"][:T_min, worst_j]
    if len(frame_track) > 10:
        temporal_corr = np.corrcoef(frame_track, frame_acc)[0, 1]
        print(f"  Temporal correlation for worst joint ({JOINT_NAMES[worst_j]}): {temporal_corr:.3f}")

        # Check if high acceleration follows high tracking error (lag analysis)
        lags = [1, 2, 3, 5]
        print(f"  Lag analysis (does high error PRECEDE high oscillation?):")
        for lag in lags:
            if len(frame_track) > lag + 10:
                corr_lag = np.corrcoef(frame_track[:-lag], frame_acc[lag:])[0, 1]
                print(f"    Lag {lag} frames ({lag/fps*1000:.0f}ms): corr = {corr_lag:.3f}")

    return {
        "per_joint": results_per_joint,
        "jerk_ratios": jerk_ratios,
        "kin_metrics": kin_metrics,
        "phys_metrics": phys_metrics,
        "tracking_err": tracking_err,
    }


def compare_cases(results):
    """Compare bad vs good case."""
    print("\n")
    print_separator("=")
    print("  CROSS-CASE COMPARISON: BAD (crouch) vs GOOD (walk)")
    print_separator("=")

    if "v4_crouch_001" not in results or "v4_walk_001" not in results:
        print("  Cannot compare — missing case data")
        return

    bad = results["v4_crouch_001"]
    good = results["v4_walk_001"]

    print(f"\n  {'Joint':<14} {'Crouch Ratio':>13} {'Walk Ratio':>11} {'Crouch Err':>11} {'Walk Err':>9}")
    print(f"  {'─'*14} {'─'*13} {'─'*11} {'─'*11} {'─'*9}")

    for j in range(NUM_JOINTS):
        bad_r = bad["per_joint"][j]["ratio"]
        good_r = good["per_joint"][j]["ratio"]
        bad_e = bad["per_joint"][j]["track_err_deg"]
        good_e = good["per_joint"][j]["track_err_deg"]

        flag = " ⚠️" if bad_r > 1.5 and bad_r > good_r * 1.5 else ""
        print(f"  {JOINT_NAMES[j]:<14} {bad_r:>10.2f}x   {good_r:>8.2f}x   {bad_e:>8.1f}°   {good_e:>6.1f}°{flag}")

    # Which joints are UNIQUELY bad in crouch?
    print(f"\n{'─'*80}")
    print(f"  JOINTS UNIQUELY PROBLEMATIC IN CROUCH (ratio > 1.5x AND > 1.5x walk ratio)")
    print(f"{'─'*80}")

    problematic = []
    for j in range(NUM_JOINTS):
        bad_r = bad["per_joint"][j]["ratio"]
        good_r = good["per_joint"][j]["ratio"]
        if bad_r > 1.5 and bad_r > good_r * 1.5:
            problematic.append((j, bad_r, good_r))

    problematic.sort(key=lambda x: x[1], reverse=True)
    for j, bad_r, good_r in problematic:
        print(f"  {JOINT_NAMES[j]:<14} Crouch: {bad_r:.2f}x  Walk: {good_r:.2f}x  (Crouch is {bad_r/good_r:.1f}x worse)")


if __name__ == "__main__":
    results = {}
    for case_name, label in CASES.items():
        result = analyze_case(case_name, label)
        if result is not None:
            results[case_name] = result
        print("\n")

    compare_cases(results)

    # Final summary
    print("\n")
    print_separator("=")
    print("  DIAGNOSTIC SUMMARY")
    print_separator("=")
    if "v4_crouch_001" in results:
        bad = results["v4_crouch_001"]
        sorted_by_ratio = sorted(bad["per_joint"], key=lambda x: x["ratio"], reverse=True)

        print("\n  Key findings for v4_crouch_001:")
        print(f"  • Total joints with phys/kin acc ratio > 1.5x: {sum(1 for r in bad['per_joint'] if r['ratio'] > 1.5)}/{NUM_JOINTS}")
        print(f"  • Total joints with phys/kin acc ratio > 2.0x: {sum(1 for r in bad['per_joint'] if r['ratio'] > 2.0)}/{NUM_JOINTS}")
        print(f"  • Worst joint: {sorted_by_ratio[0]['joint']} (ratio={sorted_by_ratio[0]['ratio']:.2f}x)")
        print(f"  • Mean tracking error: {np.mean([r['track_err_deg'] for r in bad['per_joint']]):.1f}°")

        # Pattern detection
        leg_joints = [1, 2, 4, 5, 7, 8, 10, 11]
        torso_joints = [0, 3, 6, 9, 12, 15]
        arm_joints = [13, 14, 16, 17, 18, 19, 20, 21]

        leg_avg = np.mean([bad["per_joint"][j]["ratio"] for j in leg_joints])
        torso_avg = np.mean([bad["per_joint"][j]["ratio"] for j in torso_joints])
        arm_avg = np.mean([bad["per_joint"][j]["ratio"] for j in arm_joints])

        print(f"\n  Pattern analysis:")
        print(f"  • Leg joints avg ratio:   {leg_avg:.2f}x")
        print(f"  • Torso joints avg ratio: {torso_avg:.2f}x")
        print(f"  • Arm joints avg ratio:   {arm_avg:.2f}x")

        max_group = max([("Legs", leg_avg), ("Torso", torso_avg), ("Arms", arm_avg)], key=lambda x: x[1])
        print(f"  • WORST GROUP: {max_group[0]} ({max_group[1]:.2f}x)")
