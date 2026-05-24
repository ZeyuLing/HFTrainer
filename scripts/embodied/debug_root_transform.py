#!/usr/bin/env python3
"""Focused diagnostic for Bug #4: R leg pointing upward after Y-up→Z-up transform.

Tests:
1. T-pose in Z-up (all joints zero, identity root) — baseline body positions
2. T-pose with Y-up→Z-up transformed root — should still look like T-pose
3. Frame 0 data WITHOUT coordinate transform (like PHC does) — compare body positions
4. Frame 0 data WITH coordinate transform — current approach
5. Test with Y-up gravity (model.opt.gravity = [0, -9.81, 0]) + no coord transform
6. Roundtrip test: SMPL → qpos → forward → read xpos → check plausibility

This isolates whether the problem is:
  A) The Y-up→Z-up root transform itself
  B) The Euler decomposition
  C) The joint reorder mapping
  D) Something about the MuJoCo model's rest pose
"""

import numpy as np
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as sRot

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import from our pipeline script
from scripts.embodied.run_smpl_physics_sim import (
    SMPL_JOINT_NAMES, MUJOCO_BODY_NAMES, SMPL_2_MUJOCO, MUJOCO_2_SMPL,
    rot6d_to_rotmat, decode_motion_135, yup_to_zup,
    smpl_to_qpos, _YUP_TO_ZUP, _ZUP_TO_YUP,
)


def load_model(xml_path):
    """Load MuJoCo model (without PD configuration — just for forward kinematics)."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def set_qpos_and_forward(model, data, qpos):
    """Set qpos and run forward kinematics, return body positions."""
    import mujoco
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    # Read body xpos (world positions)
    xpos = data.xpos.copy()  # (nbody, 3)
    return xpos


def print_body_positions(xpos, label="", highlight_bodies=None):
    """Print body positions in a readable format."""
    print(f"\n  --- Body Positions: {label} ---")
    for i, name in enumerate(MUJOCO_BODY_NAMES):
        x, y, z = xpos[i + 1]  # skip world body (index 0)
        marker = " <<<" if highlight_bodies and name in highlight_bodies else ""
        print(f"    [{i:2d}] {name:15s}  x={x:8.4f}  y={y:8.4f}  z={z:8.4f}{marker}")


def check_plausibility_zup(xpos, label=""):
    """Check if body positions are plausible in Z-up coordinate system."""
    issues = []
    pelvis = xpos[1]   # Pelvis = body 1
    # In Z-up: z = height, y = forward, x = right
    pelvis_h = pelvis[2]  # height

    for i, name in enumerate(MUJOCO_BODY_NAMES):
        if i == 0:
            continue  # skip Pelvis (itself)
        pos = xpos[i + 1]  # +1 because world body is index 0
        h = pos[2]

        # Feet/ankles/toes should be BELOW pelvis
        if name in ["L_Ankle", "R_Ankle", "L_Knee", "R_Knee", "L_Toe", "R_Toe"]:
            if h > pelvis_h:
                issues.append(f"  ISSUE: {name} z={h:.4f} ABOVE pelvis z={pelvis_h:.4f}")

        # Head should be ABOVE pelvis
        if name in ["Head", "Neck", "Chest"]:
            if h < pelvis_h - 0.1:  # small margin
                issues.append(f"  ISSUE: {name} z={h:.4f} BELOW pelvis z={pelvis_h:.4f}")

    if issues:
        print(f"\n  Plausibility check ({label}): FAILED")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  Plausibility check ({label}): PASSED")
    return len(issues) == 0


def check_plausibility_yup(xpos, label=""):
    """Check if body positions are plausible in Y-up coordinate system."""
    issues = []
    pelvis = xpos[1]
    pelvis_h = pelvis[1]  # Y = height in Y-up

    for i, name in enumerate(MUJOCO_BODY_NAMES):
        if i == 0:
            continue
        pos = xpos[i + 1]
        h = pos[1]  # Y = height

        if name in ["L_Ankle", "R_Ankle", "L_Knee", "R_Knee", "L_Toe", "R_Toe"]:
            if h > pelvis_h:
                issues.append(f"  ISSUE: {name} y={h:.4f} ABOVE pelvis y={pelvis_h:.4f}")

        if name in ["Head", "Neck", "Chest"]:
            if h < pelvis_h - 0.1:
                issues.append(f"  ISSUE: {name} y={h:.4f} BELOW pelvis y={pelvis_h:.4f}")

    if issues:
        print(f"\n  Plausibility check ({label}): FAILED")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  Plausibility check ({label}): PASSED")
    return len(issues) == 0


def test_tpose_zup(model, data):
    """Test 1: T-pose with identity root in Z-up.

    All joint angles zero, root at reasonable height.
    This is the baseline — should always produce valid humanoid.
    """
    print("\n" + "=" * 70)
    print("TEST 1: T-pose (identity root, Z-up)")
    print("=" * 70)

    body_pos_1 = model.body_pos[1].copy()
    print(f"  body_pos[1] (Pelvis offset): {body_pos_1}")

    qpos = np.zeros(model.nq)
    # Root position: standing at ~1m height in Z-up
    qpos[0] = 0.0   # x
    qpos[1] = 0.0   # y
    qpos[2] = 1.0   # z (height in Z-up)
    # Root quaternion: identity (no rotation)
    qpos[3] = 1.0   # w
    qpos[4:7] = 0.0  # xyz

    xpos = set_qpos_and_forward(model, data, qpos)
    print_body_positions(xpos, "T-pose Z-up (identity root)")
    check_plausibility_zup(xpos, "T-pose Z-up")

    # Check model gravity
    print(f"\n  Model gravity: {model.opt.gravity}")

    return xpos


def test_tpose_with_transform(model, data):
    """Test 2: T-pose root orientation = Rx(+90°) (what Y-up→Z-up does to identity root).

    If the original root is identity (Y-up T-pose), after Y-up→Z-up transform:
      R_root_zup = Rx(+90°) @ I = Rx(+90°)

    This should STILL produce a valid standing pose, just rotated from Y-up to Z-up.
    """
    print("\n" + "=" * 70)
    print("TEST 2: T-pose with Rx(+90°) root rotation (Y-up→Z-up transformed identity)")
    print("=" * 70)

    # Rx(+90°) as axis-angle: [pi/2, 0, 0] (rotation of 90° around X)
    Rx_90_aa = np.array([np.pi / 2, 0, 0])
    Rx_90_quat_xyzw = sRot.from_rotvec(Rx_90_aa).as_quat()
    Rx_90_quat_wxyz = Rx_90_quat_xyzw[[3, 0, 1, 2]]

    print(f"  Rx(+90°) axis-angle: {Rx_90_aa}")
    print(f"  Rx(+90°) quat wxyz:  {Rx_90_quat_wxyz}")

    qpos = np.zeros(model.nq)
    qpos[2] = 1.0   # height
    qpos[3:7] = Rx_90_quat_wxyz

    xpos = set_qpos_and_forward(model, data, qpos)
    print_body_positions(xpos, "T-pose + Rx(+90°) root")
    check_plausibility_zup(xpos, "T-pose + Rx(+90°)")

    return xpos


def test_frame0_no_transform(model, data, npz_path):
    """Test 3: Frame 0 data WITHOUT Y-up→Z-up transform (like PHC does).

    Feed Y-up SMPL data directly to MuJoCo (no coordinate transform on root).
    The model has Z-up gravity, so body will be oriented wrong relative to gravity,
    but we want to see the body positions to understand what PHC's baseline looks like.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Frame 0 — NO coordinate transform (Y-up data → MuJoCo directly)")
    print("=" * 70)

    smpl_pose, transl, fps = decode_motion_135(npz_path)
    body_pos_1 = model.body_pos[1].copy()

    print(f"  Frame 0 root_orient (Y-up, aa): {smpl_pose[0, :3]}")
    print(f"  Frame 0 transl (Y-up):          {transl[0]}")
    root_angle = np.linalg.norm(smpl_pose[0, :3])
    print(f"  Root rotation magnitude: {np.degrees(root_angle):.1f}°")

    # Convert directly (no Y-up→Z-up transform)
    ref_qpos = smpl_to_qpos(smpl_pose[:1], transl[:1], body_pos_1)
    print(f"  qpos root pos:  {ref_qpos[0, :3]}")
    print(f"  qpos root quat: {ref_qpos[0, 3:7]}")

    xpos = set_qpos_and_forward(model, data, ref_qpos[0])
    print_body_positions(xpos, "Frame 0 NO transform (Y-up direct)",
                         highlight_bodies=["R_Hip", "R_Knee", "R_Ankle", "R_Toe"])

    # Check plausibility in Z-up (even though data is Y-up — will likely fail)
    check_plausibility_zup(xpos, "Frame 0 NO transform (checking Z-up plausibility)")

    # Also check Y-up plausibility (height = Y coordinate)
    # But MuJoCo gravity is Z-down, so the "height" axis is ambiguous
    # Let's just report both
    check_plausibility_yup(xpos, "Frame 0 NO transform (checking Y-up plausibility)")

    return xpos, ref_qpos[0]


def test_frame0_with_transform(model, data, npz_path):
    """Test 4: Frame 0 data WITH Y-up→Z-up transform (our current approach)."""
    print("\n" + "=" * 70)
    print("TEST 4: Frame 0 — WITH Y-up→Z-up coordinate transform")
    print("=" * 70)

    smpl_pose, transl, fps = decode_motion_135(npz_path)
    body_pos_1 = model.body_pos[1].copy()

    # Apply Y-up→Z-up
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose[:1], transl[:1])

    print(f"  Frame 0 root_orient (Z-up, aa): {smpl_pose_zup[0, :3]}")
    print(f"  Frame 0 transl (Z-up):          {transl_zup[0]}")
    root_angle_zup = np.linalg.norm(smpl_pose_zup[0, :3])
    print(f"  Root rotation magnitude (Z-up): {np.degrees(root_angle_zup):.1f}°")

    ref_qpos = smpl_to_qpos(smpl_pose_zup[:1], transl_zup[:1], body_pos_1)
    print(f"  qpos root pos:  {ref_qpos[0, :3]}")
    print(f"  qpos root quat: {ref_qpos[0, 3:7]}")

    xpos = set_qpos_and_forward(model, data, ref_qpos[0])
    print_body_positions(xpos, "Frame 0 WITH Y-up→Z-up transform",
                         highlight_bodies=["R_Hip", "R_Knee", "R_Ankle", "R_Toe"])
    check_plausibility_zup(xpos, "Frame 0 WITH transform (Z-up)")

    return xpos, ref_qpos[0]


def test_yup_gravity_no_transform(model, data, npz_path):
    """Test 5: Change gravity to Y-up and use data without transform.

    This simulates the "just change gravity" approach. We change:
      model.opt.gravity = [0, -9.81, 0]  (Y-down)
    And feed Y-up data directly.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Y-up gravity + NO coordinate transform")
    print("=" * 70)

    # Save original gravity
    orig_gravity = model.opt.gravity.copy()

    # Change gravity to Y-up (gravity in -Y direction)
    model.opt.gravity[:] = [0, -9.81, 0]
    print(f"  Changed gravity to: {model.opt.gravity}")

    smpl_pose, transl, fps = decode_motion_135(npz_path)
    body_pos_1 = model.body_pos[1].copy()

    # No coordinate transform
    ref_qpos = smpl_to_qpos(smpl_pose[:1], transl[:1], body_pos_1)

    xpos = set_qpos_and_forward(model, data, ref_qpos[0])
    print_body_positions(xpos, "Y-up gravity, no transform",
                         highlight_bodies=["R_Hip", "R_Knee", "R_Ankle", "R_Toe"])
    check_plausibility_yup(xpos, "Y-up gravity (checking Y-up plausibility)")

    # Restore gravity
    model.opt.gravity[:] = orig_gravity

    return xpos


def test_euler_roundtrip_single_joint(model, data):
    """Test 6: Single joint Euler roundtrip via MuJoCo forward kinematics.

    For each body, set ONLY that body's joint angles to a known value (45° around X),
    then check that body's xpos makes sense.
    """
    print("\n" + "=" * 70)
    print("TEST 6: Single-joint Euler roundtrip via MuJoCo FK")
    print("=" * 70)

    body_pos_1 = model.body_pos[1].copy()

    # Start from T-pose
    base_qpos = np.zeros(model.nq)
    base_qpos[2] = 1.0  # height
    base_qpos[3] = 1.0  # identity quaternion

    # Get T-pose positions for comparison
    tpose_xpos = set_qpos_and_forward(model, data, base_qpos)

    # Test: rotate L_Hip (MuJoCo body 1) by 45° around X
    # L_Hip hinge joints start at qpos index 7 (first non-root body)
    # 3 joints per body: [x, y, z]
    test_angle = np.radians(45)

    # L_Hip is MuJoCo body index 1 -> joints at qpos[7:10]
    test_qpos = base_qpos.copy()
    test_qpos[7] = test_angle  # L_Hip_x rotation

    xpos = set_qpos_and_forward(model, data, test_qpos)
    print(f"\n  L_Hip rotated 45° around X:")
    for name_idx, name in [(1, "L_Hip"), (2, "L_Knee"), (3, "L_Ankle"), (4, "L_Toe")]:
        tpose_pos = tpose_xpos[name_idx + 1]
        new_pos = xpos[name_idx + 1]
        delta = new_pos - tpose_pos
        print(f"    {name:15s}  tpose={tpose_pos}  now={new_pos}  delta={delta}")

    # Test: rotate R_Hip by 45° around X
    # R_Hip is MuJoCo body index 5 -> joints at qpos[7 + 4*3 : 7 + 5*3] = qpos[19:22]
    test_qpos = base_qpos.copy()
    test_qpos[19] = test_angle  # R_Hip_x rotation

    xpos = set_qpos_and_forward(model, data, test_qpos)
    print(f"\n  R_Hip rotated 45° around X:")
    for name_idx, name in [(5, "R_Hip"), (6, "R_Knee"), (7, "R_Ankle"), (8, "R_Toe")]:
        tpose_pos = tpose_xpos[name_idx + 1]
        new_pos = xpos[name_idx + 1]
        delta = new_pos - tpose_pos
        print(f"    {name:15s}  tpose={tpose_pos}  now={new_pos}  delta={delta}")


def test_smpl_aa_to_qpos_roundtrip(model, data):
    """Test 7: Take a known SMPL axis-angle, convert to qpos, check body positions.

    Use a simple known pose: standing with left arm raised 90°.
    """
    print("\n" + "=" * 70)
    print("TEST 7: Known SMPL pose → qpos → MuJoCo FK → check positions")
    print("=" * 70)

    body_pos_1 = model.body_pos[1].copy()

    # Create a simple known SMPL pose (all zeros = T-pose)
    smpl_pose = np.zeros((1, 72), dtype=np.float32)
    transl = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)  # already Z-up, height=1

    # L_Shoulder (SMPL index 16): raise left arm by rotating -90° around X
    # This should move the left arm upward in Z-up
    smpl_pose[0, 16 * 3] = -np.pi / 2  # L_Shoulder X rotation = -90°

    # Convert to qpos (no coordinate transform — pose is already in "Z-up" for this test)
    ref_qpos = smpl_to_qpos(smpl_pose, transl, body_pos_1)

    # Check which qpos slots got the L_Shoulder angles
    # L_Shoulder is SMPL joint 16 (0-indexed non-root: 15)
    # SMPL_2_MUJOCO maps this to MuJoCo order
    mj_idx = SMPL_2_MUJOCO[15]  # SMPL non-root index 15 -> MuJoCo non-root index
    print(f"  L_Shoulder: SMPL non-root idx=15, MuJoCo non-root idx={mj_idx}")
    print(f"  L_Shoulder MuJoCo body name: {MUJOCO_BODY_NAMES[mj_idx + 1]}")
    print(f"  qpos slots: [{7 + mj_idx * 3}:{7 + mj_idx * 3 + 3}] = "
          f"{ref_qpos[0, 7 + mj_idx * 3:7 + mj_idx * 3 + 3]}")

    xpos = set_qpos_and_forward(model, data, ref_qpos[0])

    print(f"\n  Key body positions (Z-up, L arm raised):")
    for name in ["Pelvis", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
                  "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]:
        idx = MUJOCO_BODY_NAMES.index(name) + 1  # +1 for world body
        print(f"    {name:15s}  x={xpos[idx, 0]:8.4f}  y={xpos[idx, 1]:8.4f}  z={xpos[idx, 2]:8.4f}")


def test_per_body_isolation(model, data, npz_path):
    """Test 8: Set frame 0 root + one body at a time, check each body's effect.

    Unlike the old diagnostic which accumulated all joints, this sets
    root + exactly ONE body's joints and checks the result.
    """
    print("\n" + "=" * 70)
    print("TEST 8: Frame 0 — root + one body at a time (isolated)")
    print("=" * 70)

    smpl_pose, transl, fps = decode_motion_135(npz_path)
    body_pos_1 = model.body_pos[1].copy()

    # Apply Y-up→Z-up
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose[:1], transl[:1])

    # Get full qpos for reference
    full_qpos = smpl_to_qpos(smpl_pose_zup[:1], transl_zup[:1], body_pos_1)

    # Start from root-only (all body joints zero)
    root_qpos = np.zeros(model.nq)
    root_qpos[:7] = full_qpos[0, :7]  # root trans + quat only

    root_xpos = set_qpos_and_forward(model, data, root_qpos)
    print(f"\n  ROOT ONLY (all body joints = 0):")
    print_body_positions(root_xpos, "Root only")
    check_plausibility_zup(root_xpos, "Root only")

    # For key bodies, add just that body's joints
    key_bodies = ["L_Hip", "R_Hip", "L_Knee", "R_Knee", "Torso"]
    for body_name in key_bodies:
        mj_body_idx = MUJOCO_BODY_NAMES.index(body_name)  # includes root at 0
        mj_nonroot_idx = mj_body_idx - 1  # 0-indexed non-root

        # Get the joint angles for this body from full qpos
        joint_start = 7 + mj_nonroot_idx * 3
        joint_end = joint_start + 3
        joint_angles = full_qpos[0, joint_start:joint_end]

        # Create qpos with root + just this body
        test_qpos = root_qpos.copy()
        test_qpos[joint_start:joint_end] = joint_angles

        # Also find what SMPL joint this maps to
        smpl_nonroot_idx = MUJOCO_2_SMPL[mj_nonroot_idx]
        smpl_joint_name = SMPL_JOINT_NAMES[smpl_nonroot_idx + 1]
        smpl_aa = smpl_pose_zup[0, (smpl_nonroot_idx + 1) * 3:(smpl_nonroot_idx + 1) * 3 + 3]
        smpl_aa_mag = np.degrees(np.linalg.norm(smpl_aa))

        xpos = set_qpos_and_forward(model, data, test_qpos)
        print(f"\n  ROOT + {body_name} (SMPL: {smpl_joint_name}):")
        print(f"    SMPL aa: {smpl_aa}, magnitude: {smpl_aa_mag:.1f}°")
        print(f"    MuJoCo XYZ euler: {joint_angles}")

        # Show only the affected chain
        if body_name in ["L_Hip", "L_Knee"]:
            chain = ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"]
        elif body_name in ["R_Hip", "R_Knee"]:
            chain = ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"]
        else:
            chain = [body_name]

        for name in chain:
            idx = MUJOCO_BODY_NAMES.index(name) + 1
            root_pos = root_xpos[idx]
            new_pos = xpos[idx]
            delta = new_pos - root_pos
            print(f"    {name:15s}  root_only={root_pos}  "
                  f"with_{body_name}={new_pos}  delta={delta}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Focused root transform diagnostic")
    parser.add_argument("--xml-path", type=str, required=True,
                        help="Path to smpl_humanoid.xml")
    parser.add_argument("--npz-file", type=str, required=True,
                        help="Path to a motion_135 NPZ file")
    args = parser.parse_args()

    model, data = load_model(args.xml_path)
    print(f"Model: {model.nbody} bodies, nq={model.nq}, nv={model.nv}")
    print(f"Gravity: {model.opt.gravity}")
    print(f"body_pos[1]: {model.body_pos[1]}")

    # Print model body hierarchy
    print(f"\n  Model body hierarchy:")
    for i in range(model.nbody):
        name = model.body(i).name
        parent_id = model.body_parentid[i]
        parent_name = model.body(parent_id).name if parent_id >= 0 else "WORLD"
        pos = model.body_pos[i]
        print(f"    [{i:2d}] {name:15s}  parent={parent_name:15s}  "
              f"local_pos=({pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f})")

    # Run all tests
    test_tpose_zup(model, data)
    test_tpose_with_transform(model, data)
    test_frame0_no_transform(model, data, args.npz_file)
    test_frame0_with_transform(model, data, args.npz_file)
    test_yup_gravity_no_transform(model, data, args.npz_file)
    test_euler_roundtrip_single_joint(model, data)
    test_smpl_aa_to_qpos_roundtrip(model, data)
    test_per_body_isolation(model, data, args.npz_file)


if __name__ == "__main__":
    main()
