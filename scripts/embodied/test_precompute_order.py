#!/usr/bin/env python3
"""Test hypothesis: does precompute ordering affect survival steps?

The key difference between test_init_diff (148 steps) and lockstep (118 steps):
  - test_init_diff: precompute_maxcoords on RAW ref_qpos, then adj_body_pos[:,:,2] += shift
  - lockstep: ref_qpos[:, 2] += shift FIRST, then precompute_maxcoords on SHIFTED

Theoretically these should produce identical results (since FK is linear in root Z).
This test verifies numerically AND runs both approaches to compare survival steps.

Additionally: directly imports and calls test_init_diff.run_simulation() to confirm
its actual behavior matches what we think.
"""

import numpy as np
import mujoco
import sys
import os
import yaml
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))

# Paths
MJCF_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
YAML_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.yaml"
NPZ_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4/data/npz/v4_walk_005.npz"
ONNX_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.onnx"


def main():
    from test_physics_configs import load_model_with_config
    from run_smpl_rl_tracker import decode_motion_135, yup_to_zup, smpl_to_qpos
    from test_init_diff import precompute_maxcoords, run_simulation

    with open(YAML_PATH) as f:
        yaml_meta = yaml.safe_load(f)

    # Load motion
    smpl_pose, transl, fps = decode_motion_135(NPZ_PATH)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)
    dt_ref = 1.0 / fps

    # Load ONNX
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    print("=" * 80)
    print("  TEST: Precompute ordering vs test_init_diff direct call")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════════
    # APPROACH 1: Exactly as test_init_diff does it (direct function call)
    # (precompute on RAW, run_simulation internally does height shift)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  APPROACH 1: Direct call to test_init_diff.run_simulation()")
    print("  (precompute on RAW ref_qpos, run_simulation handles height shift)")
    print("─" * 80)

    model_1, data_1, _ = load_model_with_config("D_euler_with_margin")
    body_pos_1 = model_1.body_pos[1].copy()
    ref_qpos_1 = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)

    # Precompute on RAW ref_qpos (same as test_init_diff line 306)
    bp_1, br_1, bv_1, bav_1 = precompute_maxcoords(
        model_1, data_1, ref_qpos_1, dt_ref, use_float32=False)

    print(f"  ref_qpos_1[0, 2] (RAW root Z): {ref_qpos_1[0, 2]:.8f}")
    print(f"  body_pos_1[0, 0, 2] (RAW root body Z): {bp_1[0, 0, 2]:.8f}")

    # Call run_simulation directly (it computes height shift internally)
    result_1 = run_simulation(
        model_1, data_1, ref_qpos_1, bp_1, br_1, bv_1, bav_1,
        dt_ref, session, set_ctrl_init=False, label="Direct_test_init_diff")

    # ═══════════════════════════════════════════════════════════════════════════
    # APPROACH 2: Lockstep approach (shift first, then precompute)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  APPROACH 2: Lockstep approach (shift ref_qpos first, then precompute)")
    print("─" * 80)

    model_2, data_2, _ = load_model_with_config("D_euler_with_margin")
    body_pos_1_2 = model_2.body_pos[1].copy()
    ref_qpos_2 = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1_2)

    # Compute height shift first (same as lockstep lines 178-216)
    data_2.qpos[:] = ref_qpos_2[0]
    data_2.qvel[:] = 0.0
    mujoco.mj_forward(model_2, data_2)

    left_ids = set()
    right_ids = set()
    for bid in range(1, model_2.nbody):
        bname = mujoco.mj_id2name(model_2, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in ("L_Ankle", "L_Toe"):
            left_ids.add(bid)
        elif bname in ("R_Ankle", "R_Toe"):
            right_ids.add(bid)

    def _lowest_z(body_id_set, model, data):
        min_z = float("inf")
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] not in body_id_set:
                continue
            gtype = int(model.geom_type[gid])
            gsize = model.geom_size[gid]
            gxpos = data.geom_xpos[gid]
            gxmat = data.geom_xmat[gid].reshape(3, 3)
            if gtype == 5:
                z_ext = abs(gxmat[2, 2]) * gsize[1] + gsize[0]
                bottom = gxpos[2] - z_ext
            elif gtype == 3:
                bottom = gxpos[2] - gsize[0]
            elif gtype == 6:
                z_ext = sum(abs(gxmat[2, j]) * gsize[j] for j in range(3))
                bottom = gxpos[2] - z_ext
            else:
                bottom = gxpos[2]
            min_z = min(min_z, bottom)
        return min_z

    left_min = _lowest_z(left_ids, model_2, data_2)
    right_min = _lowest_z(right_ids, model_2, data_2)
    height_shift = 0.0 - min(left_min, right_min)
    print(f"  height_shift = {height_shift:+.8f}")

    # Apply shift BEFORE precompute (lockstep approach)
    ref_qpos_2[:, 2] += height_shift
    print(f"  ref_qpos_2[0, 2] (SHIFTED root Z): {ref_qpos_2[0, 2]:.8f}")

    # Precompute on SHIFTED ref_qpos
    bp_2, br_2, bv_2, bav_2 = precompute_maxcoords(
        model_2, data_2, ref_qpos_2, dt_ref, use_float32=False)

    print(f"  body_pos_2[0, 0, 2] (SHIFTED root body Z): {bp_2[0, 0, 2]:.8f}")

    # Now run simulation using the shifted data directly
    # We pass ref_qpos_2 (already shifted) and bp_2 (already shifted)
    # run_simulation will try to compute height shift AGAIN from ref_qpos_2[0]...
    # That's the key: run_simulation always re-computes height_shift internally!
    # If ref_qpos is already shifted, the height_shift computed inside will be ~0.
    result_2_via_run_sim = run_simulation(
        model_2, data_2, ref_qpos_2, bp_2, br_2, bv_2, bav_2,
        dt_ref, session, set_ctrl_init=False, label="Lockstep_via_run_sim")

    # ═══════════════════════════════════════════════════════════════════════════
    # APPROACH 3: Lockstep approach but bypass run_simulation's internal shift
    # (manual sim loop, same as lockstep_compare.py)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  APPROACH 3: Manual sim loop (lockstep style, no double-shift)")
    print("─" * 80)

    model_3, data_3, _ = load_model_with_config("D_euler_with_margin")
    body_pos_1_3 = model_3.body_pos[1].copy()
    ref_qpos_3 = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1_3)

    # Compute and apply shift
    data_3.qpos[:] = ref_qpos_3[0]
    data_3.qvel[:] = 0.0
    mujoco.mj_forward(model_3, data_3)
    left_min_3 = _lowest_z(left_ids, model_3, data_3)
    right_min_3 = _lowest_z(right_ids, model_3, data_3)
    height_shift_3 = 0.0 - min(left_min_3, right_min_3)
    ref_qpos_3[:, 2] += height_shift_3

    # Precompute on shifted
    bp_3, br_3, bv_3, bav_3 = precompute_maxcoords(
        model_3, data_3, ref_qpos_3, dt_ref, use_float32=False)

    # Set initial state (no internal height fix since already shifted)
    data_3.qpos[:] = ref_qpos_3[0]
    data_3.qvel[:] = 0.0
    mujoco.mj_forward(model_3, data_3)

    # Manual sim loop (same as lockstep_compare.py)
    CONTROL_DT = 0.02
    DECIMATION = 20
    MAX_STEPS = 200
    FALL_THRESHOLD = 0.3
    num_ref_frames = ref_qpos_3.shape[0]
    sim_time = 0.0
    prev_actions = np.zeros(69, dtype=np.float32)
    out_names = [o.name for o in session.get_outputs()]

    def _extract_sim_state(model, data):
        num_bodies = 24
        sim_pos = data.xpos[1:num_bodies + 1].copy()
        sim_rot_wxyz = data.xquat[1:num_bodies + 1].copy()
        sim_rot = sim_rot_wxyz[..., [1, 2, 3, 0]]  # wxyz -> xyzw
        sim_vel = np.zeros((num_bodies, 3))
        sim_ang_vel = np.zeros((num_bodies, 3))
        for i in range(num_bodies):
            bid = i + 1
            lin_vel = data.cvel[bid, 3:6].copy()
            ang_vel = data.cvel[bid, 0:3].copy()
            xmat = data.xmat[bid].reshape(3, 3)
            body_ipos = model.body_ipos[bid]
            offset = xmat @ body_ipos
            lin_vel_com = lin_vel + np.cross(ang_vel, offset)
            sim_vel[i] = lin_vel_com
            sim_ang_vel[i] = ang_vel
        return sim_pos, sim_rot, sim_vel, sim_ang_vel

    result_3 = MAX_STEPS
    for step in range(MAX_STEPS):
        sim_pos, sim_rot, sim_vel, sim_ang_vel = _extract_sim_state(model_3, data_3)

        ref_time = sim_time + CONTROL_DT
        ref_frame_idx = min(int(ref_time / dt_ref), num_ref_frames - 1)

        inputs = {
            "current_rigid_body_ang_vel": sim_ang_vel[np.newaxis].astype(np.float32),
            "current_rigid_body_pos": sim_pos[np.newaxis].astype(np.float32),
            "current_rigid_body_rot": sim_rot[np.newaxis].astype(np.float32),
            "current_rigid_body_vel": sim_vel[np.newaxis].astype(np.float32),
            "ground_heights": np.zeros((1,), dtype=np.float32),
            "historical_actions": prev_actions[np.newaxis, np.newaxis].astype(np.float32),
            "mimic_future_ang_vel": bav_3[ref_frame_idx:ref_frame_idx+1][np.newaxis].astype(np.float32),
            "mimic_future_pos": bp_3[ref_frame_idx:ref_frame_idx+1][np.newaxis].astype(np.float32),
            "mimic_future_rot": br_3[ref_frame_idx:ref_frame_idx+1][np.newaxis].astype(np.float32),
            "mimic_future_vel": bv_3[ref_frame_idx:ref_frame_idx+1][np.newaxis].astype(np.float32),
        }

        outputs = session.run(out_names, inputs)
        out_dict = {n: v for n, v in zip(out_names, outputs)}

        jpt = out_dict["joint_pos_targets"].squeeze()
        prev_actions = out_dict["actions"].squeeze().copy()

        if "stiffness_targets" in out_dict:
            stiff = out_dict["stiffness_targets"].squeeze()
            damp = out_dict["damping_targets"].squeeze()
            for i in range(model_3.nu):
                model_3.actuator_gainprm[i, 0] = float(stiff[i])
                model_3.actuator_biasprm[i, 1] = -float(stiff[i])
                model_3.actuator_biasprm[i, 2] = -float(damp[i])

        data_3.ctrl[:] = jpt
        for _ in range(DECIMATION):
            mujoco.mj_step(model_3, data_3)
        sim_time += CONTROL_DT

        root_h = data_3.qpos[2]
        if root_h < FALL_THRESHOLD:
            print(f"  [Manual_lockstep] FELL at step {step}! root_h={root_h:.4f}")
            result_3 = step
            break
        if np.any(np.isnan(data_3.qpos)):
            print(f"  [Manual_lockstep] NaN at step {step}!")
            result_3 = step
            break
    else:
        print(f"  [Manual_lockstep] SURVIVED {MAX_STEPS} steps! Final root_h={data_3.qpos[2]:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # NUMERIC COMPARISON: RAW precompute vs SHIFTED precompute
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("  NUMERIC COMPARISON: Precompute on RAW vs SHIFTED")
    print("─" * 80)

    # Compare body_pos: bp_1 (RAW) + shift vs bp_2 (SHIFTED)
    bp_1_shifted = bp_1.copy()
    bp_1_shifted[:, :, 2] += height_shift
    bp_diff = np.abs(bp_1_shifted - bp_2).max()
    print(f"  body_pos (RAW+shift vs SHIFTED): max diff = {bp_diff:.2e}")

    # Compare body_rot
    br_diff = np.abs(br_1 - br_2).max()
    print(f"  body_rot (RAW vs SHIFTED): max diff = {br_diff:.2e}")

    # Compare body_vel (this is the critical one)
    bv_diff = np.abs(bv_1 - bv_2).max()
    print(f"  body_vel (RAW vs SHIFTED): max diff = {bv_diff:.2e}")
    if bv_diff > 1e-10:
        frame_diffs = np.abs(bv_1 - bv_2).max(axis=(1, 2))
        worst_frame = np.argmax(frame_diffs)
        print(f"    Worst frame: {worst_frame}, diff={frame_diffs[worst_frame]:.2e}")
        print(f"    bv_1[{worst_frame}, 0] = {bv_1[worst_frame, 0]}")
        print(f"    bv_2[{worst_frame}, 0] = {bv_2[worst_frame, 0]}")

    # Compare body_ang_vel
    bav_diff = np.abs(bav_1 - bav_2).max()
    print(f"  body_ang_vel (RAW vs SHIFTED): max diff = {bav_diff:.2e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Approach 1 (test_init_diff.run_simulation, precompute RAW): step {result_1}")
    print(f"  Approach 2 (shift first, then run_simulation): step {result_2_via_run_sim}")
    print(f"  Approach 3 (manual lockstep, shift first, no run_simulation): step {result_3}")
    print()

    if result_1 > result_3 + 10:
        print("  !! CONFIRMED: Precompute ordering matters!")
        print(f"  !! Gap: {result_1 - result_3} steps")
        print("  !! test_init_diff's approach (precompute RAW, shift later) is BETTER")
    elif result_1 == result_3 or abs(result_1 - result_3) <= 2:
        print("  Precompute ordering does NOT matter (same survival)")
        print("  The 30-step gap must come from something else.")
    else:
        print(f"  Small difference: {abs(result_1 - result_3)} steps")

    if result_2_via_run_sim != result_3:
        print(f"\n  !! run_simulation's internal height fix caused a difference!")
        print(f"  !! Approach 2 ({result_2_via_run_sim}) vs Approach 3 ({result_3})")
        print("  !! This means run_simulation applies height_shift AGAIN to already-shifted data")
        print("  !! (double-shift bug if input is pre-shifted)")


if __name__ == "__main__":
    main()
