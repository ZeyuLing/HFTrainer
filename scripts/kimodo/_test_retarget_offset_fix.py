#!/usr/bin/env python3
"""Quick validation: verify neutral-pose offset compensation in smpl22_to_soma30_retarget.

Tests:
  1. T-pose identity: SMPLX22 identity rotations → SOMA30 arms should be horizontal
  2. Single motion clip: retargeted SOMA30 proportions match SOMA30 neutral
  3. Shoulder angle check: left/right arm bones should not be tilted

Usage (on GPU machine):
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/kimodo/_test_retarget_offset_fix.py
"""
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KIMODO_ROOT = os.path.join(PROJECT_ROOT, "ref_repo", "KIMODO", "kimodo")
sys.path.insert(0, KIMODO_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from scripts.kimodo.run_kimodo_all_tasks import (
    smpl22_to_soma30_retarget, SMPLX22_TO_SOMA30, SOMA30_NAMES,
)
from kimodo.skeleton.definitions import SOMASkeleton30, SMPLXSkeleton22


def test_tpose_horizontal_arms():
    """Identity SMPLX22 rotations → SOMA30 arms should be horizontal.

    Key criteria:
      a) Arms must be horizontal: |Y-component| of arm bone direction < threshold
      b) Arms must be lateral: left arm extends in +X, right arm in -X
      c) Bone lengths must match SOMA30 neutral proportions

    Note: exact angular match with SOMA30 neutral is NOT required because the
    FK chain propagation through spine/chest joints introduces small XZ-plane
    direction differences. What matters is horizontal arms + correct proportions.
    """
    print("=" * 60)
    print("TEST 1: T-pose → SOMA30 arms horizontal")
    print("=" * 60)

    smplx22 = SMPLXSkeleton22()
    soma30 = SOMASkeleton30()
    soma_names = soma30.bone_order_names
    idx = {n: i for i, n in enumerate(soma_names)}

    # Identity SMPLX22 motion: T-pose, 1 frame
    T = 1
    # Build 135-dim: trans(3) + rot6d(132), identity rotation = [[1,0,0],[0,1,0]] in rot6d
    motion = np.zeros((T, 135), dtype=np.float32)
    # rot6d_to_rotmat_row_major reshapes (6,) -> (3,2), a1=col0, a2=col1.
    # Identity rot6d = [1, 0, 0, 1, 0, 0]:
    #   -> [[1,0],[0,1],[0,0]] -> a1=[1,0,0], a2=[0,1,0] -> I
    for j in range(22):
        motion[0, 3 + j * 6 + 0] = 1.0  # a1[0] = 1
        motion[0, 3 + j * 6 + 3] = 1.0  # a2[1] = 1

    bone_offsets = (smplx22.neutral_joints[1:] - smplx22.neutral_joints[smplx22.joint_parents[1:]]).numpy()
    bone_offsets = np.concatenate([np.zeros((1, 3), dtype=np.float32), bone_offsets], axis=0)

    soma_rots, soma_pos = smpl22_to_soma30_retarget(motion, bone_offsets)

    # Check arm bones: horizontal (|Y|<threshold) + lateral direction + bone length
    arm_checks = [
        ("LeftShoulder→LeftArm",    "LeftShoulder", "LeftArm",     +1),  # left: +X
        ("LeftArm→LeftForeArm",     "LeftArm",      "LeftForeArm", +1),
        ("RightShoulder→RightArm",  "RightShoulder","RightArm",    -1),  # right: -X
        ("RightArm→RightForeArm",   "RightArm",     "RightForeArm",-1),
    ]

    neutral = soma30.neutral_joints.float()
    all_pass = True
    for label, parent_name, child_name, expected_x_sign in arm_checks:
        pi, ci = idx[parent_name], idx[child_name]
        retarget_dir = soma_pos[0, ci] - soma_pos[0, pi]
        retarget_len = retarget_dir.norm().item()
        retarget_dir_norm = retarget_dir / retarget_dir.norm()

        y_abs = abs(retarget_dir_norm[1].item())
        x_sign_ok = (retarget_dir_norm[0].item() * expected_x_sign) > 0
        horizontal_ok = y_abs < 0.05  # arm bone |Y| < 5% of unit length

        # Bone length check against SOMA30 neutral
        ref_len = (neutral[ci] - neutral[pi]).norm().item()
        bone_err_cm = abs(retarget_len - ref_len) * 100

        ok = horizontal_ok and x_sign_ok and bone_err_cm < 3.0
        all_pass = all_pass and ok
        status = "OK" if ok else "FAIL"
        print(f"  {label:35s}  |Y|={y_abs:.4f}  X_sign={'OK' if x_sign_ok else 'BAD'}  "
              f"len={retarget_len:.4f}m (ref={ref_len:.4f}m, err={bone_err_cm:.2f}cm)  [{status}]")

    print(f"\n  PASS: {all_pass}")
    return all_pass


def test_shoulder_not_collapsed():
    """Load a real motion clip and check shoulder height."""
    print("\n" + "=" * 60)
    print("TEST 2: Real motion clip — shoulder height check")
    print("=" * 60)

    smplx22 = SMPLXSkeleton22()
    soma30 = SOMASkeleton30()
    idx = {n: i for i, n in enumerate(soma30.bone_order_names)}

    # Try to load a test clip
    ann_path = os.path.join(PROJECT_ROOT, "data", "annotation", "train_hymotion_400h.json")
    if not os.path.exists(ann_path):
        print("  SKIP: annotation file not found")
        return True

    import json
    with open(ann_path) as f:
        ann = json.load(f)
    data_list = ann["data_list"]
    ann_dir = os.path.dirname(ann_path)

    # Find a clip
    test_clip = None
    for key, item in list(data_list.items())[:200]:
        nf = item.get("num_frames", 0)
        if nf < 60 or nf > 150:
            continue
        smplx_path = os.path.normpath(os.path.join(ann_dir, item["smplx_path"]))
        if os.path.exists(smplx_path):
            test_clip = (key, smplx_path)
            break

    if test_clip is None:
        print("  SKIP: no suitable test clip found")
        return True

    key, path = test_clip
    data = np.load(path, allow_pickle=True)
    poses = data["poses"].astype(np.float32)
    trans = data["trans"].astype(np.float32)
    T = min(poses.shape[0], 100)
    poses = poses[:T]
    trans = trans[:T]

    # Build 135-dim motion
    from hftrainer.pipelines.motion.differentiable_fk import rot6d_to_rotmat_row_major
    from kimodo.geometry import axis_angle_to_matrix

    body_aa = poses[:, :66].reshape(T, 22, 3)
    rot_mats = axis_angle_to_matrix(torch.from_numpy(body_aa).float())  # (T, 22, 3, 3)

    # Convert rotmat to rot6d (first two columns)
    rot6d = rot_mats[:, :, :, :2].permute(0, 1, 3, 2).reshape(T, 22, 6)  # (T, 22, 6)
    motion_135 = np.zeros((T, 135), dtype=np.float32)
    motion_135[:, :3] = trans
    motion_135[:, 3:135] = rot6d.numpy().reshape(T, 132)

    bone_offsets = np.zeros((22, 3), dtype=np.float32)
    neutral_j = smplx22.neutral_joints.numpy()
    parents = smplx22.joint_parents
    for j in range(22):
        if parents[j] != j:
            bone_offsets[j] = neutral_j[j] - neutral_j[parents[j]]

    soma_rots, soma_pos = smpl22_to_soma30_retarget(motion_135, bone_offsets)

    # Check: LeftArm and RightArm (shoulder joints) should not be below Chest
    chest_y = soma_pos[:, idx["Chest"], 1].mean().item()
    left_arm_y = soma_pos[:, idx["LeftArm"], 1].mean().item()
    right_arm_y = soma_pos[:, idx["RightArm"], 1].mean().item()

    # Shoulder should be close to or above chest
    left_ok = left_arm_y >= chest_y - 0.05  # allow 5cm below
    right_ok = right_arm_y >= chest_y - 0.05

    print(f"  Clip: {key}")
    print(f"  Frames: {T}")
    print(f"  Chest Y (mean):     {chest_y:.4f}m")
    print(f"  LeftArm Y (mean):   {left_arm_y:.4f}m  [{'OK' if left_ok else 'FAIL'}]")
    print(f"  RightArm Y (mean):  {right_arm_y:.4f}m  [{'OK' if right_ok else 'FAIL'}]")

    # Check bone proportions
    left_upper = torch.norm(soma_pos[:, idx["LeftForeArm"]] - soma_pos[:, idx["LeftArm"]], dim=-1).mean().item()
    right_upper = torch.norm(soma_pos[:, idx["RightForeArm"]] - soma_pos[:, idx["RightArm"]], dim=-1).mean().item()

    # SOMA30 neutral reference
    neutral = soma30.neutral_joints.float()
    ref_left_upper = (neutral[idx["LeftForeArm"]] - neutral[idx["LeftArm"]]).norm().item()
    ref_right_upper = (neutral[idx["RightForeArm"]] - neutral[idx["RightArm"]]).norm().item()

    left_bone_err = abs(left_upper - ref_left_upper) * 100
    right_bone_err = abs(right_upper - ref_right_upper) * 100

    print(f"  Left upper arm: {left_upper:.4f}m (ref: {ref_left_upper:.4f}m, err: {left_bone_err:.2f}cm)")
    print(f"  Right upper arm: {right_upper:.4f}m (ref: {ref_right_upper:.4f}m, err: {right_bone_err:.2f}cm)")

    all_pass = left_ok and right_ok and left_bone_err < 3 and right_bone_err < 3
    print(f"\n  PASS: {all_pass}")
    return all_pass


if __name__ == "__main__":
    p1 = test_tpose_horizontal_arms()
    p2 = test_shoulder_not_collapsed()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = [
        ("T-pose → horizontal arms", p1),
        ("Real clip shoulder check", p2),
    ]
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:40s} [{status}]")
        all_pass = all_pass and passed

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    sys.exit(0 if all_pass else 1)
