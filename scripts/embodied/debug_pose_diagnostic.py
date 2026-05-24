#!/usr/bin/env python3
"""Diagnostic: isolate which joint/value causes R leg to go upward in MuJoCo.

Tests:
1. Decode motion_135 frame 0, print all joint axis-angles
2. Start from T-pose, add one joint at a time, check body xpos
3. Compare L vs R joints for symmetry issues
4. Full pose check: set all joints and verify body positions
"""
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.embodied.run_smpl_physics_sim import (
    decode_motion_135, yup_to_zup, smpl_to_qpos,
    SMPL_JOINT_NAMES, MUJOCO_BODY_NAMES, SMPL_2_MUJOCO, MUJOCO_2_SMPL,
    _MUJOCO_TO_SMPL_NAME,
)


def test_reorder_mapping():
    """Print and verify the joint reorder mapping."""
    print("=" * 60)
    print("TEST 1: Joint Reorder Mapping Verification")
    print("=" * 60)

    smpl_nonroot = SMPL_JOINT_NAMES[1:]
    mj_nonroot = MUJOCO_BODY_NAMES[1:]

    print(f"\nSMPL_2_MUJOCO = {SMPL_2_MUJOCO}")
    print(f"MUJOCO_2_SMPL = {MUJOCO_2_SMPL}")

    print(f"\nMapping (MuJoCo body idx -> SMPL non-root joint idx):")
    for mj_idx, smpl_idx in enumerate(SMPL_2_MUJOCO):
        mj_name = mj_nonroot[mj_idx]
        smpl_name = smpl_nonroot[smpl_idx]
        smpl_mapped = _MUJOCO_TO_SMPL_NAME.get(mj_name, mj_name)
        match = "✓" if smpl_mapped == smpl_name else "✗ MISMATCH"
        print(f"  MJ[{mj_idx:2d}] {mj_name:15s} -> SMPL[{smpl_idx:2d}] {smpl_name:15s} {match}")

    print(f"\nReverse mapping (SMPL non-root idx -> MuJoCo body idx):")
    for smpl_idx, mj_idx in enumerate(MUJOCO_2_SMPL):
        smpl_name = smpl_nonroot[smpl_idx]
        mj_name = mj_nonroot[mj_idx]
        print(f"  SMPL[{smpl_idx:2d}] {smpl_name:15s} -> MJ[{mj_idx:2d}] {mj_name:15s}")

    # Verify round-trip
    for i in range(23):
        assert MUJOCO_2_SMPL[SMPL_2_MUJOCO[i]] == i, f"Round-trip failed at {i}"
        assert SMPL_2_MUJOCO[MUJOCO_2_SMPL[i]] == i, f"Reverse round-trip failed at {i}"
    print("\n✓ Round-trip verification passed")


def test_decode_frame0(npz_path):
    """Decode motion_135 and print frame 0 joint values."""
    print("\n" + "=" * 60)
    print("TEST 2: Decode motion_135 frame 0")
    print("=" * 60)

    smpl_pose_yup, transl_yup, fps = decode_motion_135(npz_path)
    T = smpl_pose_yup.shape[0]
    print(f"Motion: {T} frames @ {fps}fps")
    print(f"Translation frame 0 (Y-up): {transl_yup[0]}")

    # Print all joint axis-angles at frame 0
    joint_aa = smpl_pose_yup[0].reshape(24, 3)
    print(f"\nJoint axis-angles at frame 0 (Y-up, before coord transform):")
    for i, name in enumerate(SMPL_JOINT_NAMES):
        aa = joint_aa[i]
        angle_deg = np.linalg.norm(aa) * 180 / np.pi
        if angle_deg > 0.1:
            print(f"  [{i:2d}] {name:15s}: aa=[{aa[0]:+.4f}, {aa[1]:+.4f}, {aa[2]:+.4f}]  "
                  f"angle={angle_deg:.1f}°")
        else:
            print(f"  [{i:2d}] {name:15s}: ~zero ({angle_deg:.2f}°)")

    # Y-up -> Z-up
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose_yup, transl_yup)
    print(f"\nTranslation frame 0 (Z-up): {transl_zup[0]}")

    joint_aa_zup = smpl_pose_zup[0].reshape(24, 3)
    print(f"\nJoint axis-angles at frame 0 (Z-up):")
    print(f"  Root: [{joint_aa_zup[0, 0]:+.4f}, {joint_aa_zup[0, 1]:+.4f}, {joint_aa_zup[0, 2]:+.4f}]")

    # Check L vs R symmetry
    pairs = [(1, 2, "Hip"), (4, 5, "Knee"), (7, 8, "Ankle"), (10, 11, "Foot")]
    print(f"\n  L/R Joint Symmetry Check (non-root, Z-up):")
    for l_idx, r_idx, name in pairs:
        l_aa = joint_aa_zup[l_idx]
        r_aa = joint_aa_zup[r_idx]
        print(f"  L_{name:8s}[{l_idx:2d}]: [{l_aa[0]:+.4f}, {l_aa[1]:+.4f}, {l_aa[2]:+.4f}]  "
              f"angle={np.linalg.norm(l_aa)*180/np.pi:.1f}°")
        print(f"  R_{name:8s}[{r_idx:2d}]: [{r_aa[0]:+.4f}, {r_aa[1]:+.4f}, {r_aa[2]:+.4f}]  "
              f"angle={np.linalg.norm(r_aa)*180/np.pi:.1f}°")

    return smpl_pose_zup, transl_zup


def test_one_joint_at_a_time(xml_path, smpl_pose_zup, transl_zup):
    """Set one joint at a time on T-pose, check body positions."""
    import mujoco

    print("\n" + "=" * 60)
    print("TEST 3: One-Joint-At-A-Time Body Position Check")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    body_pos_1 = model.body_pos[1].copy()

    # T-pose reference: all zeros
    t_pose = np.zeros((1, 72), dtype=np.float32)
    t_transl = transl_zup[:1].copy()  # Use actual translation
    t_qpos = smpl_to_qpos(t_pose, t_transl, body_pos_1)

    # Full pose
    full_qpos = smpl_to_qpos(smpl_pose_zup[:1], transl_zup[:1], body_pos_1)

    # Print T-pose body positions
    data.qpos[:] = t_qpos[0]
    mujoco.mj_forward(model, data)

    print(f"\n--- T-pose body positions ---")
    print(f"{'Body':<15s} {'x':>8s} {'y':>8s} {'z':>8s}")
    t_xpos = {}
    for i, name in enumerate(MUJOCO_BODY_NAMES):
        if i == 0:
            continue  # skip world body
        # body index 0 is world, body 1 is Pelvis
        pos = data.xpos[i + 1].copy()  # +1 because world body is 0
        t_xpos[name] = pos
        print(f"  {name:<15s} {pos[0]:+8.4f} {pos[1]:+8.4f} {pos[2]:+8.4f}")

    # Actually, let me use model body names directly
    print(f"\n--- Full pose body positions (frame 0) ---")
    data.qpos[:] = full_qpos[0]
    mujoco.mj_forward(model, data)

    print(f"{'Body':<15s} {'x':>8s} {'y':>8s} {'z':>8s}  {'Δz from T-pose':>14s}")
    full_xpos = {}
    for bid in range(model.nbody):
        bname = model.body(bid).name
        pos = data.xpos[bid].copy()
        full_xpos[bname] = pos
        if bname == "world":
            continue
        dz = pos[2] - (t_xpos.get(bname, [0,0,0])[2] if bname in t_xpos else 0)
        flag = " ⚠️" if bname in ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"] and pos[2] > 0.9 else ""
        print(f"  {bname:<15s} {pos[0]:+8.4f} {pos[1]:+8.4f} {pos[2]:+8.4f}  {dz:+8.4f}{flag}")

    # Now test one body joint at a time
    print(f"\n--- One-joint-at-a-time test ---")
    print(f"Setting each non-root SMPL joint individually on T-pose base")
    print(f"Only showing joints that move leg bodies significantly\n")

    joint_aa_frame0 = smpl_pose_zup[0].reshape(24, 3)

    for smpl_joint_idx in range(1, 22):  # Skip root (0) and L/R_Hand (22,23)
        # Create pose with only this one joint set
        single_pose = np.zeros((1, 72), dtype=np.float32)
        single_pose[0, :3] = smpl_pose_zup[0, :3]  # Keep root orientation
        single_pose[0, smpl_joint_idx*3:(smpl_joint_idx+1)*3] = joint_aa_frame0[smpl_joint_idx]

        single_qpos = smpl_to_qpos(single_pose, transl_zup[:1], body_pos_1)
        data.qpos[:] = single_qpos[0]
        mujoco.mj_forward(model, data)

        # Check if any leg body moved significantly
        max_dz = 0
        affected = []
        for bid in range(model.nbody):
            bname = model.body(bid).name
            if bname == "world":
                continue
            if bname in t_xpos:
                dz = data.xpos[bid][2] - t_xpos[bname][2]
                if abs(dz) > 0.01:
                    affected.append(f"{bname}(Δz={dz:+.3f})")
                    max_dz = max(max_dz, abs(dz))

        jname = SMPL_JOINT_NAMES[smpl_joint_idx]
        aa = joint_aa_frame0[smpl_joint_idx]
        angle_deg = np.linalg.norm(aa) * 180 / np.pi

        if max_dz > 0.01 or angle_deg > 5:
            print(f"  SMPL[{smpl_joint_idx:2d}] {jname:15s}: "
                  f"aa=[{aa[0]:+.4f},{aa[1]:+.4f},{aa[2]:+.4f}] ({angle_deg:.1f}°)")
            if affected:
                print(f"    Affected bodies: {', '.join(affected)}")
            else:
                print(f"    No significant body movement")

    # Detailed check of euler values going into qpos
    print(f"\n--- Euler values in qpos for leg joints ---")
    print(f"{'MJ Body':<15s} {'qpos_idx':>10s}  {'euler_x':>8s} {'euler_y':>8s} {'euler_z':>8s}  {'from SMPL joint'}")

    for mj_body_idx in range(23):  # non-root
        mj_name = MUJOCO_BODY_NAMES[mj_body_idx + 1]
        if not any(x in mj_name for x in ["Hip", "Knee", "Ankle", "Toe"]):
            continue

        qpos_start = 7 + mj_body_idx * 3
        euler = full_qpos[0, qpos_start:qpos_start+3]

        smpl_idx = SMPL_2_MUJOCO[mj_body_idx]
        smpl_name = SMPL_JOINT_NAMES[smpl_idx + 1]  # +1 for root offset
        smpl_aa = joint_aa_frame0[smpl_idx + 1]

        print(f"  {mj_name:<15s} [{qpos_start:2d}:{qpos_start+3:2d}]  "
              f"{euler[0]:+8.4f} {euler[1]:+8.4f} {euler[2]:+8.4f}  "
              f"<- SMPL[{smpl_idx+1:2d}] {smpl_name} "
              f"aa=[{smpl_aa[0]:+.4f},{smpl_aa[1]:+.4f},{smpl_aa[2]:+.4f}]")


def test_euler_roundtrip(smpl_pose_zup):
    """Test that euler conversion is round-trip consistent."""
    print("\n" + "=" * 60)
    print("TEST 4: Euler Conversion Round-Trip")
    print("=" * 60)

    joint_aa = smpl_pose_zup[0].reshape(24, 3)

    for smpl_idx in range(1, 22):
        name = SMPL_JOINT_NAMES[smpl_idx]
        aa = joint_aa[smpl_idx]
        angle_deg = np.linalg.norm(aa) * 180 / np.pi
        if angle_deg < 1.0:
            continue

        # Forward: aa -> rotation matrix -> XYZ euler
        R = sRot.from_rotvec(aa)
        euler_xyz = R.as_euler("XYZ")  # [x, y, z]

        # Reverse: XYZ euler -> rotation matrix -> aa
        R2 = sRot.from_euler("XYZ", euler_xyz)
        aa2 = R2.as_rotvec()

        err = np.linalg.norm(aa - aa2)
        status = "✓" if err < 1e-6 else f"✗ err={err:.6f}"

        if err > 1e-6:
            print(f"  [{smpl_idx:2d}] {name:15s}: aa={aa}  euler_xyz={euler_xyz}  "
                  f"aa_back={aa2}  {status}")


def test_comparison_with_phc_convention(xml_path, smpl_pose_zup, transl_zup):
    """Compare XYZ vs ZYX convention body positions side by side."""
    import mujoco

    print("\n" + "=" * 60)
    print("TEST 5: XYZ vs ZYX Euler Convention Comparison")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    body_pos_1 = model.body_pos[1].copy()

    T = 1
    joint_aa = smpl_pose_zup[0].reshape(24, 3)

    # Method A: XYZ (our method — physically correct per test)
    qpos_xyz = smpl_to_qpos(smpl_pose_zup[:1], transl_zup[:1], body_pos_1)

    # Method B: ZYX (PHC's method)
    qpos_zyx = np.zeros((1, 76), dtype=np.float64)
    qpos_zyx[0, :3] = transl_zup[0].astype(np.float64) + body_pos_1
    root_quat_xyzw = sRot.from_rotvec(joint_aa[0]).as_quat()
    qpos_zyx[0, 3:7] = root_quat_xyzw[[3, 0, 1, 2]]

    body_aa = joint_aa[1:].reshape(-1, 3)
    body_euler_zyx = sRot.from_rotvec(body_aa).as_euler("ZYX")  # [z, y, x]
    body_euler_zyx = body_euler_zyx.reshape(1, 23, 3)
    body_euler_zyx_mj = body_euler_zyx[:, SMPL_2_MUJOCO]
    qpos_zyx[0, 7:] = body_euler_zyx_mj.reshape(69)

    # Print qpos differences for leg joints
    print(f"\n--- qpos differences (leg joints only) ---")
    print(f"{'MJ Body':<15s}  {'XYZ':>24s}  {'ZYX':>24s}  {'diff':>24s}")
    for mj_body_idx in range(23):
        mj_name = MUJOCO_BODY_NAMES[mj_body_idx + 1]
        if not any(x in mj_name for x in ["Hip", "Knee", "Ankle", "Toe"]):
            continue
        s = 7 + mj_body_idx * 3
        xyz = qpos_xyz[0, s:s+3]
        zyx = qpos_zyx[0, s:s+3]
        d = xyz - zyx
        print(f"  {mj_name:<15s}  [{xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f}]  "
              f"[{zyx[0]:+.4f},{zyx[1]:+.4f},{zyx[2]:+.4f}]  "
              f"[{d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f}]")

    # Body positions with each convention
    for label, qpos in [("XYZ (ours)", qpos_xyz), ("ZYX (PHC)", qpos_zyx)]:
        data.qpos[:] = qpos[0]
        mujoco.mj_forward(model, data)
        print(f"\n--- Body positions with {label} ---")
        print(f"  {'Body':<15s} {'x':>8s} {'y':>8s} {'z':>8s}")
        for bid in range(model.nbody):
            bname = model.body(bid).name
            if bname == "world":
                continue
            if not any(x in bname for x in ["Pelvis", "Hip", "Knee", "Ankle", "Toe"]):
                continue
            pos = data.xpos[bid]
            flag = " ⚠️" if "R_" in bname and pos[2] > 0.9 else ""
            print(f"  {bname:<15s} {pos[0]:+8.4f} {pos[1]:+8.4f} {pos[2]:+8.4f}{flag}")


def test_raw_motion_joint_positions(npz_path):
    """Check if the NPZ has joint positions to compare against."""
    print("\n" + "=" * 60)
    print("TEST 6: Raw NPZ Contents")
    print("=" * 60)

    data = np.load(npz_path, allow_pickle=True)
    print(f"NPZ keys: {list(data.keys())}")
    for k in data.keys():
        v = data[k]
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                  f"range=[{v.min():.4f}, {v.max():.4f}]")
        else:
            print(f"  {k}: {v}")

    # If there are joint positions, print frame 0
    if 'positions' in data:
        pos = data['positions']  # (T, 22, 3)
        print(f"\nJoint positions frame 0 (Y-up):")
        for i in range(min(22, pos.shape[1])):
            p = pos[0, i]
            print(f"  [{i:2d}] {SMPL_JOINT_NAMES[i]:15s}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")

    # Print motion_135 frame 0 raw values
    motion = data['motion_135']
    transl = motion[0, :3]
    print(f"\nFrame 0 translation (raw): [{transl[0]:.4f}, {transl[1]:.4f}, {transl[2]:.4f}]")

    # Print rot6d per joint
    rot6d = motion[0, 3:].reshape(22, 6)
    print(f"\nFrame 0 rot6d per joint:")
    for i in range(22):
        r = rot6d[i]
        # Check if near identity: row-major identity rot6d = [1,0, 0,1, 0,0]
        is_identity = np.allclose(r, [1, 0, 0, 1, 0, 0], atol=0.1)
        tag = " (≈identity)" if is_identity else ""
        print(f"  [{i:2d}] {SMPL_JOINT_NAMES[i]:15s}: [{r[0]:+.4f},{r[1]:+.4f},{r[2]:+.4f},"
              f"{r[3]:+.4f},{r[4]:+.4f},{r[5]:+.4f}]{tag}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="Path to motion_135 NPZ")
    parser.add_argument("--xml", type=str, required=True, help="Path to smpl_humanoid.xml")
    args = parser.parse_args()

    # Test 1: Verify mapping
    test_reorder_mapping()

    # Test 6: Raw NPZ contents (first, to understand data)
    test_raw_motion_joint_positions(args.npz)

    # Test 2: Decode and print frame 0
    smpl_pose_zup, transl_zup = test_decode_frame0(args.npz)

    # Test 4: Euler round-trip
    test_euler_roundtrip(smpl_pose_zup)

    # Test 3: One joint at a time (requires mujoco)
    test_one_joint_at_a_time(args.xml, smpl_pose_zup, transl_zup)

    # Test 5: XYZ vs ZYX comparison
    test_comparison_with_phc_convention(args.xml, smpl_pose_zup, transl_zup)
