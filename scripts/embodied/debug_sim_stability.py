#!/usr/bin/env python3
"""Debug physics simulation stability — diagnose why mj_step explodes.

Tests:
1. Forward kinematics only (mj_forward) — does the pose produce valid contact?
2. Single substep — when does NaN first appear?
3. Contact forces at initial state
4. Actuator forces at initial state
5. Gravity-only test (no actuators)
"""
import numpy as np
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as sRot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.embodied.run_smpl_physics_sim import (
    MUJOCO_BODY_NAMES, decode_motion_135, yup_to_zup, smpl_to_qpos,
    load_mujoco_model, PD_GAINS_PER_BODY,
)


def main():
    import argparse
    import mujoco

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-path", type=str, required=True)
    parser.add_argument("--npz-file", type=str, required=True)
    args = parser.parse_args()

    smpl_pose, transl, fps = decode_motion_135(args.npz_file)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    # =========================================================
    # Test 1: Raw MuJoCo model (no custom PD setup) — mj_forward only
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 1: Raw model, mj_forward only (FK check)")
    print("=" * 70)

    model_raw = mujoco.MjModel.from_xml_path(str(args.xml_path))
    data_raw = mujoco.MjData(model_raw)

    body_pos_1 = model_raw.body_pos[1].copy()
    ref_qpos = smpl_to_qpos(smpl_pose_zup[:1], transl_zup[:1], body_pos_1)

    print(f"ref_qpos[0, :7] (root): {ref_qpos[0, :7]}")
    print(f"ref_qpos[0, 7:16] (first 3 bodies): {ref_qpos[0, 7:16]}")
    print(f"Joint angles range: [{ref_qpos[0, 7:].min():.4f}, {ref_qpos[0, 7:].max():.4f}] rad")

    data_raw.qpos[:] = ref_qpos[0]
    data_raw.qvel[:] = 0.0
    mujoco.mj_forward(model_raw, data_raw)

    print(f"\nAfter mj_forward:")
    print(f"  Root pos (xpos[1]): {data_raw.xpos[1]}")
    print(f"  # contacts: {data_raw.ncon}")
    for i in range(data_raw.ncon):
        c = data_raw.contact[i]
        print(f"  Contact {i}: geom1={c.geom1} geom2={c.geom2} dist={c.dist:.6f} pos={c.pos}")

    # Print all body positions (Z component)
    print(f"\n  Body Z positions (Z-up, should be >= 0):")
    below_ground = 0
    for i in range(1, model_raw.nbody):
        name = mujoco.mj_id2name(model_raw, mujoco.mjtObj.mjOBJ_BODY, i)
        z = data_raw.xpos[i, 2]
        flag = " *** BELOW GROUND ***" if z < 0 else ""
        print(f"    [{i:2d}] {name:15s}  z={z:8.4f}{flag}")
        if z < 0:
            below_ground += 1
    print(f"  Bodies below ground: {below_ground}")

    # =========================================================
    # Test 2: Custom PD model — check actuator forces at t=0
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 2: Custom PD model — actuator forces at t=0")
    print("=" * 70)

    model, data = load_mujoco_model(str(args.xml_path))

    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    data.ctrl[:] = ref_qpos[0, 7:]  # PD target = current position
    mujoco.mj_forward(model, data)

    # Check actuator forces
    print(f"\n  Actuator forces (should be ~0 if ctrl == qpos):")
    print(f"  ctrl range: [{data.ctrl.min():.4f}, {data.ctrl.max():.4f}]")
    print(f"  qpos[7:] range: [{data.qpos[7:].min():.4f}, {data.qpos[7:].max():.4f}]")

    # Compute expected PD force: kp*(ctrl - qpos) - kd*qvel = kp*0 - kd*0 = 0
    ctrl_qpos_diff = data.ctrl - data.qpos[7:]
    print(f"  ctrl - qpos[7:] range: [{ctrl_qpos_diff.min():.6f}, {ctrl_qpos_diff.max():.6f}]")
    print(f"  actuator_force range: [{data.actuator_force.min():.6f}, {data.actuator_force.max():.6f}]")

    # Check qacc
    print(f"\n  qacc range: [{data.qacc.min():.6f}, {data.qacc.max():.6f}]")
    print(f"  qacc[:6] (root free joint): {data.qacc[:6]}")

    # Check for NaN/Inf
    has_nan = np.any(np.isnan(data.qacc)) or np.any(np.isnan(data.qpos))
    has_inf = np.any(np.isinf(data.qacc)) or np.any(np.isinf(data.qpos))
    print(f"  NaN in qacc/qpos: {has_nan}, Inf: {has_inf}")

    # =========================================================
    # Test 3: Step-by-step simulation (1 substep at a time)
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 3: Step-by-step simulation (single substeps)")
    print("=" * 70)

    model2, data2 = load_mujoco_model(str(args.xml_path))
    data2.qpos[:] = ref_qpos[0]
    data2.qvel[:] = 0.0
    data2.ctrl[:] = ref_qpos[0, 7:]
    mujoco.mj_forward(model2, data2)

    print(f"  Initial: root_pos={data2.qpos[:3]}, root_h={data2.qpos[2]:.4f}")

    for step in range(20):
        mujoco.mj_step(model2, data2)
        root_h = float(data2.qpos[2])
        has_nan = np.any(np.isnan(data2.qpos)) or np.any(np.isnan(data2.qvel))
        has_inf = np.any(np.isinf(data2.qpos)) or np.any(np.isinf(data2.qvel))
        ncon = data2.ncon

        max_qvel = float(np.max(np.abs(data2.qvel)))
        max_qacc = float(np.max(np.abs(data2.qacc))) if not np.any(np.isnan(data2.qacc)) else float('nan')

        print(f"  Step {step+1:3d}: t={data2.time:.5f}s root_h={root_h:.4f} "
              f"max_|qvel|={max_qvel:.4f} max_|qacc|={max_qacc:.4f} "
              f"ncon={ncon} NaN={has_nan} Inf={has_inf}")

        if has_nan or has_inf:
            print(f"  *** EXPLOSION at step {step+1}! ***")
            # Print which DOFs have NaN
            nan_dofs = np.where(np.isnan(data2.qvel))[0]
            inf_dofs = np.where(np.isinf(data2.qvel))[0]
            if len(nan_dofs):
                print(f"    NaN qvel DOFs: {nan_dofs}")
            if len(inf_dofs):
                print(f"    Inf qvel DOFs: {inf_dofs}")
            break

    # =========================================================
    # Test 4: T-pose (identity) simulation — is gravity alone stable?
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 4: T-pose simulation (all zeros, gravity + PD)")
    print("=" * 70)

    model3, data3 = load_mujoco_model(str(args.xml_path))

    # T-pose: all joint angles = 0, root at height 1.0
    data3.qpos[:] = 0.0
    data3.qpos[2] = 1.0    # root height
    data3.qpos[3] = 1.0    # quaternion w=1 (identity)
    data3.qvel[:] = 0.0
    data3.ctrl[:] = 0.0     # PD targets = 0 (T-pose)
    mujoco.mj_forward(model3, data3)

    print(f"  Initial T-pose: root_pos={data3.qpos[:3]}, root_h={data3.qpos[2]:.4f}")

    for step in range(30):
        mujoco.mj_step(model3, data3)
        root_h = float(data3.qpos[2])
        has_nan = np.any(np.isnan(data3.qpos))
        max_qvel = float(np.max(np.abs(data3.qvel))) if not np.any(np.isnan(data3.qvel)) else float('nan')
        ncon = data3.ncon

        print(f"  Step {step+1:3d}: t={data3.time:.5f}s root_h={root_h:.4f} "
              f"max_|qvel|={max_qvel:.4f} ncon={ncon} NaN={has_nan}")

        if has_nan:
            print(f"  *** T-pose EXPLOSION at step {step+1}! ***")
            break

    # =========================================================
    # Test 5: Check actuator-joint mapping
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 5: Actuator-joint mapping verification")
    print("=" * 70)

    model4 = mujoco.MjModel.from_xml_path(str(args.xml_path))

    print(f"  model.nu = {model4.nu} (actuators)")
    print(f"  model.nq = {model4.nq} (qpos dim)")
    print(f"  model.nv = {model4.nv} (qvel/dof dim)")

    for i in range(min(model4.nu, 30)):
        name = mujoco.mj_id2name(model4, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = model4.actuator_trnid[i]
        trntype = model4.actuator_trntype[i]
        gear = model4.actuator_gear[i]
        gainprm = model4.actuator_gainprm[i, :3]
        biasprm = model4.actuator_biasprm[i, :3]
        biastype = model4.actuator_biastype[i]

        # Get joint name
        jnt_name = mujoco.mj_id2name(model4, mujoco.mjtObj.mjOBJ_JOINT, trnid[0])
        # Get joint qpos address
        jnt_qposadr = model4.jnt_qposadr[trnid[0]]

        print(f"  Act[{i:2d}] {name:20s} -> Joint '{jnt_name}' qposadr={jnt_qposadr} "
              f"gear={gear[0]:.0f} gainprm={gainprm} biastype={biastype} biasprm={biasprm}")

    # Verify that actuator[i] maps to qpos[7+i]
    print(f"\n  Verifying actuator[i] -> qpos[7+i] mapping:")
    mapping_ok = True
    for i in range(model4.nu):
        trnid = model4.actuator_trnid[i]
        jnt_qposadr = model4.jnt_qposadr[trnid[0]]
        expected = 7 + i
        if jnt_qposadr != expected:
            print(f"    MISMATCH: actuator[{i}] -> qpos[{jnt_qposadr}], expected qpos[{expected}]")
            mapping_ok = False
    print(f"  Mapping OK: {mapping_ok}")

    # =========================================================
    # Test 6: Gravity-only test (disable actuators)
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 6: Gravity-only (no actuators) — does body just fall gently?")
    print("=" * 70)

    model5 = mujoco.MjModel.from_xml_path(str(args.xml_path))
    data5 = mujoco.MjData(model5)

    # Zero out all passive dynamics
    model5.jnt_stiffness[:] = 0.0
    model5.dof_damping[:] = 0.0
    model5.dof_frictionloss[:] = 0.0

    # Disable actuators by setting gain to 0
    for i in range(model5.nu):
        model5.actuator_gainprm[i, 0] = 0.0
        model5.actuator_biasprm[i, :] = 0.0
        model5.actuator_gear[i, :] = np.array([1, 0, 0, 0, 0, 0])

    # Set initial pose
    data5.qpos[:] = ref_qpos[0]
    data5.qvel[:] = 0.0
    data5.ctrl[:] = 0.0
    mujoco.mj_forward(model5, data5)

    print(f"  Initial: root_h={data5.qpos[2]:.4f}, contacts={data5.ncon}")

    for step in range(20):
        mujoco.mj_step(model5, data5)
        root_h = float(data5.qpos[2])
        has_nan = np.any(np.isnan(data5.qpos))
        max_qvel = float(np.max(np.abs(data5.qvel))) if not np.any(np.isnan(data5.qvel)) else float('nan')
        ncon = data5.ncon

        print(f"  Step {step+1:3d}: t={data5.time:.5f}s root_h={root_h:.4f} "
              f"max_|qvel|={max_qvel:.4f} ncon={ncon} NaN={has_nan}")

        if has_nan:
            print(f"  *** Gravity-only EXPLOSION at step {step+1}! ***")
            break


if __name__ == "__main__":
    main()
