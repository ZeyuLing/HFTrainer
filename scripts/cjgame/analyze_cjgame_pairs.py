"""Analyze CJGame_MB original/cleaned NPZ pairs."""
import numpy as np
import glob
import os
import json

# === FK ===
SMPL_PARENTS = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19], dtype=np.int32)
SMPL_TPOSE_OFFSETS = np.array([
    [0, 0, 0], [0.0585, -0.0823, 0], [-0.0585, -0.0823, 0], [0, 0.0643, 0],
    [0, -0.4003, 0], [0, -0.4003, 0], [0, 0.1217, 0], [0, -0.4260, 0],
    [0, -0.4260, 0], [0, 0.1187, 0], [0.0400, -0.0600, 0.1300], [-0.0400, -0.0600, 0.1300],
    [0, 0.1050, 0], [0.0740, 0.0200, -0.0100], [-0.0740, 0.0200, -0.0100], [0, 0.0700, 0],
    [0.1032, -0.0285, -0.0087], [-0.1032, -0.0285, -0.0087], [0.2576, 0, 0],
    [-0.2576, 0, 0], [0.2537, 0, 0], [-0.2537, 0, 0],
], dtype=np.float64)

def axis_angle_to_rotmat(aa):
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    angle = np.clip(angle, 1e-8, None)
    axis = aa / angle
    K = np.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype)
    K[..., 0, 1] = -axis[..., 2]; K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]; K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]; K[..., 2, 1] = axis[..., 0]
    sin_a = np.sin(angle[..., np.newaxis])
    cos_a = np.cos(angle[..., np.newaxis])
    eye = np.eye(3, dtype=aa.dtype)
    R = eye + sin_a * K + (1 - cos_a) * (K @ K)
    return R

def simple_fk(poses_aa, trans):
    T = poses_aa.shape[0]
    aa = poses_aa[:, :66].reshape(T, 22, 3).astype(np.float64)
    rotmats = axis_angle_to_rotmat(aa)
    joints = np.zeros((T, 22, 3), dtype=np.float64)
    global_rotmats = np.zeros((T, 22, 3, 3), dtype=np.float64)
    for j in range(22):
        parent = SMPL_PARENTS[j]
        offset = SMPL_TPOSE_OFFSETS[j]
        if parent == -1:
            global_rotmats[:, j] = rotmats[:, j]
            joints[:, j] = trans.astype(np.float64)
        else:
            global_rotmats[:, j] = global_rotmats[:, parent] @ rotmats[:, j]
            joints[:, j] = joints[:, parent] + (global_rotmats[:, parent] @ offset)
    return joints

# === Analysis ===
NPZ_DIR = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/data/lightai_data/CJGame_MB/npz/"

# Find pairs
all_files = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
originals = [f for f in all_files if not f.endswith("_cleaned.npz")]

results = {
    "summary": {"total_pairs": 0, "length_matched": 0, "length_changed": 0,
                "visible_diff": 0, "no_visible_diff": 0, "cleaned_has_quality_issues": 0},
    "length_changed": [],
    "visible_diff": [],
    "no_visible_diff": [],
    "quality_issues": [],
}

VISIBLE_THRESH_MM = 30.0
JOINT_JUMP_THRESH_MM = 80.0
ACCEL_THRESH_MM = 200.0
FROZEN_MIN_FRAMES = 10

for orig_path in originals:
    name = os.path.basename(orig_path).replace(".npz", "")
    cleaned_path = orig_path.replace(".npz", "_cleaned.npz")
    if not os.path.exists(cleaned_path):
        continue

    results["summary"]["total_pairs"] += 1

    orig = np.load(orig_path)
    cleaned = np.load(cleaned_path)

    o_poses, o_trans = orig["poses"], orig["trans"]
    c_poses, c_trans = cleaned["poses"], cleaned["trans"]

    len_match = o_poses.shape[0] == c_poses.shape[0]
    if len_match:
        results["summary"]["length_matched"] += 1
    else:
        results["summary"]["length_changed"] += 1
        results["length_changed"].append({"name": name, "orig_frames": int(o_poses.shape[0]), "cleaned_frames": int(c_poses.shape[0])})

    # FK
    o_joints = simple_fk(o_poses, o_trans) * 1000  # meters -> mm
    c_joints = simple_fk(c_poses, c_trans) * 1000

    # Diff (use min length)
    min_len = min(o_joints.shape[0], c_joints.shape[0])
    diff = np.linalg.norm(o_joints[:min_len] - c_joints[:min_len], axis=-1)  # (T, 22)
    per_frame_max = diff.max(axis=1)  # (T,)
    max_diff = float(per_frame_max.max())
    mean_diff = float(diff.mean())

    if max_diff > VISIBLE_THRESH_MM:
        results["summary"]["visible_diff"] += 1
        desc_parts = []
        if not len_match:
            desc_parts.append(f"length changed {o_poses.shape[0]}->{c_poses.shape[0]}")
        # Find which joints are most affected
        worst_frame = int(per_frame_max.argmax())
        worst_joint = int(diff[worst_frame].argmax())
        joint_names = ["pelvis","l_hip","r_hip","spine1","l_knee","r_knee","spine2",
                       "l_ankle","r_ankle","spine3","l_foot","r_foot","neck",
                       "l_collar","r_collar","head","l_shoulder","r_shoulder",
                       "l_elbow","r_elbow","l_wrist","r_wrist"]
        desc_parts.append(f"worst: joint {joint_names[worst_joint]} at frame {worst_frame}")
        results["visible_diff"].append({
            "name": name, "max_diff_mm": round(max_diff, 2), "mean_diff_mm": round(mean_diff, 2),
            "description": "; ".join(desc_parts)
        })
    else:
        results["summary"]["no_visible_diff"] += 1
        results["no_visible_diff"].append({
            "name": name, "max_diff_mm": round(max_diff, 2), "mean_diff_mm": round(mean_diff, 2)
        })

    # Quality issues on cleaned
    issues = []
    T_c = c_joints.shape[0]
    if T_c >= 2:
        vel = np.linalg.norm(np.diff(c_joints, axis=0), axis=-1)  # (T-1, 22)
        max_vel = float(vel.max())
        if max_vel > JOINT_JUMP_THRESH_MM:
            worst_f = int(vel.max(axis=1).argmax())
            worst_j = int(vel[worst_f].argmax())
            issues.append({"type": "joint_jump", "max_velocity_mm": round(max_vel, 2),
                          "frame": worst_f, "joint": joint_names[worst_j]})

    if T_c >= 3:
        accel = np.diff(c_joints, n=2, axis=0)  # (T-2, 22, 3)
        accel_mag = np.linalg.norm(accel, axis=-1)
        max_accel = float(accel_mag.max())
        if max_accel > ACCEL_THRESH_MM:
            worst_f = int(accel_mag.max(axis=1).argmax())
            worst_j = int(accel_mag[worst_f].argmax())
            issues.append({"type": "high_acceleration", "max_accel_mm_f2": round(max_accel, 2),
                          "frame": worst_f, "joint": joint_names[worst_j]})

    # Frozen trailing frames
    if T_c >= FROZEN_MIN_FRAMES:
        # Check if last N frames are identical
        last_poses = c_poses[-FROZEN_MIN_FRAMES:]
        diffs_tail = np.abs(last_poses[1:] - last_poses[:-1]).max(axis=1)
        if (diffs_tail < 1e-8).all():
            # Count how many frozen frames from end
            frozen_count = 1
            for i in range(T_c - 2, -1, -1):
                if np.abs(c_poses[i] - c_poses[i+1]).max() < 1e-8:
                    frozen_count += 1
                else:
                    break
            if frozen_count >= FROZEN_MIN_FRAMES:
                issues.append({"type": "frozen_trailing", "frozen_frames": frozen_count})

    if issues:
        results["summary"]["cleaned_has_quality_issues"] += 1
        results["quality_issues"].append({"name": name, "issues": [i["type"] for i in issues], "details": issues})

# Sort visible_diff by max_diff descending
results["visible_diff"].sort(key=lambda x: -x["max_diff_mm"])

out_path = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/docs/temp/cjgame_pair_analysis.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(json.dumps(results["summary"], indent=2))
print(f"\nReport saved to {out_path}")
print(f"\nTop 10 visible diffs:")
for item in results["visible_diff"][:10]:
    print(f"  {item['name']}: max={item['max_diff_mm']:.1f}mm, mean={item['mean_diff_mm']:.1f}mm — {item['description']}")
print(f"\nLength changed ({len(results['length_changed'])}):")
for item in results["length_changed"]:
    print(f"  {item['name']}: {item['orig_frames']} -> {item['cleaned_frames']}")
print(f"\nQuality issues ({len(results['quality_issues'])}):")
for item in results["quality_issues"][:20]:
    print(f"  {item['name']}: {item['issues']}")
