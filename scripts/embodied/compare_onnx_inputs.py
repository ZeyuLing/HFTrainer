#!/usr/bin/env python3
"""Compare ONNX inputs at step 0 between test_physics_configs and run_smpl_rl_tracker.

This script runs BOTH code paths on the same NPZ and compares the EXACT numeric
values passed to the ONNX policy at step 0. Any difference explains why one works
and the other doesn't.

Strategy: Import from each script exactly, reproducing the same initialization
logic each script uses.
"""

import numpy as np
import mujoco
import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(__file__))

# Paths
MJCF_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
YAML_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.yaml"
NPZ_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4/data/npz/v4_walk_005.npz"
ONNX_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.onnx"

with open(YAML_PATH) as f:
    YAML_META = yaml.safe_load(f)

PHYSICS_DT = 0.001
CONTROL_DT = 0.02
DECIMATION = 20


def get_inputs_from_test_physics_configs():
    """Reproduce test_physics_configs.py step 0 ONNX inputs.

    Uses exact same functions as test_physics_configs.py:
      load_model_with_config, precompute_maxcoords, extract_sim_state.
    Ground height fix: same bilateral foot grounding as test_physics_configs main().
    """
    from test_physics_configs import (
        MUJOCO_BODY_NAMES, precompute_maxcoords, extract_sim_state,
        load_model_with_config, quat_mul_wxyz
    )
    from run_smpl_rl_tracker import decode_motion_135, yup_to_zup, smpl_to_qpos

    smpl_pose, transl, fps = decode_motion_135(NPZ_PATH)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    # Load model with Config D (working config)
    model, data, desc = load_model_with_config("D_euler_with_margin")
    body_pos_1 = model.body_pos[1].copy()

    # Convert to qpos
    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)

    # Ground height fix: same logic as test_physics_configs.py main() (line 487-531)
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    left_foot_ids = set()
    right_foot_ids = set()
    for bid in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in ("L_Ankle", "L_Toe"):
            left_foot_ids.add(bid)
        elif bname in ("R_Ankle", "R_Toe"):
            right_foot_ids.add(bid)

    def _lowest_geom_z(body_id_set, model, data):
        min_z = float("inf")
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] not in body_id_set:
                continue
            gtype = int(model.geom_type[gid])
            gsize = model.geom_size[gid]
            gxpos = data.geom_xpos[gid]
            gxmat = data.geom_xmat[gid].reshape(3, 3)
            if gtype == 5:  # capsule
                z_ext = abs(gxmat[2, 2]) * gsize[1] + gsize[0]
                bottom = gxpos[2] - z_ext
            elif gtype == 3:  # sphere
                bottom = gxpos[2] - gsize[0]
            elif gtype == 6:  # box
                z_ext = sum(abs(gxmat[2, j]) * gsize[j] for j in range(3))
                bottom = gxpos[2] - z_ext
            else:
                bottom = gxpos[2]
            min_z = min(min_z, bottom)
        return min_z

    left_min = _lowest_geom_z(left_foot_ids, model, data)
    right_min = _lowest_geom_z(right_foot_ids, model, data)
    grounding_ref_z = min(left_min, right_min)
    height_shift = 0.0 - grounding_ref_z
    ref_qpos[:, 2] += height_shift
    print(f"  [test_physics] height_shift = {height_shift:+.6f}m")
    print(f"  [test_physics] root_h after = {ref_qpos[0, 2]:.6f}m")

    dt_ref = 1.0 / fps

    # Precompute maxcoords
    body_pos, body_rot, body_vel, body_ang_vel = precompute_maxcoords(
        model, data, ref_qpos, dt_ref)

    # Set initial pose
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Extract sim state
    sim_pos, sim_rot, sim_vel, sim_ang_vel = extract_sim_state(model, data)

    # Future reference (same as test_physics_configs)
    ref_time = 0.0 + CONTROL_DT
    ref_frame_idx = min(int(ref_time / dt_ref), ref_qpos.shape[0] - 1)
    print(f"  [test_physics] future ref_frame_idx = {ref_frame_idx} (time={ref_time:.4f}, dt_ref={dt_ref:.6f})")

    future_pos = body_pos[ref_frame_idx:ref_frame_idx+1]
    future_rot = body_rot[ref_frame_idx:ref_frame_idx+1]
    future_vel = body_vel[ref_frame_idx:ref_frame_idx+1]
    future_ang_vel = body_ang_vel[ref_frame_idx:ref_frame_idx+1]

    inputs = {
        "current_rigid_body_ang_vel": sim_ang_vel[np.newaxis].astype(np.float32),
        "current_rigid_body_pos": sim_pos[np.newaxis].astype(np.float32),
        "current_rigid_body_rot": sim_rot[np.newaxis].astype(np.float32),
        "current_rigid_body_vel": sim_vel[np.newaxis].astype(np.float32),
        "ground_heights": np.zeros((1,), dtype=np.float32),
        "historical_actions": np.zeros((1, 1, 69), dtype=np.float32),
        "mimic_future_ang_vel": future_ang_vel[np.newaxis].astype(np.float32),
        "mimic_future_pos": future_pos[np.newaxis].astype(np.float32),
        "mimic_future_rot": future_rot[np.newaxis].astype(np.float32),
        "mimic_future_vel": future_vel[np.newaxis].astype(np.float32),
    }
    return inputs, ref_qpos, body_pos, body_rot, body_vel, body_ang_vel


def get_inputs_from_run_smpl_rl_tracker():
    """Reproduce run_smpl_rl_tracker.py step 0 ONNX inputs.

    Uses exact same functions: load_mujoco_model, precompute_reference_maxcoords,
    extract_sim_state. Ground height fix: bilateral foot grounding (same as run_smpl).
    """
    from run_smpl_rl_tracker import (
        decode_motion_135, yup_to_zup, smpl_to_qpos,
        load_mujoco_model,
        precompute_reference_maxcoords, extract_sim_state,
    )

    smpl_pose, transl, fps = decode_motion_135(NPZ_PATH)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    stiffness = YAML_META["control"]["stiffness"]
    damping = YAML_META["control"]["damping"]

    # Load model (same physics as Config D — Euler + margin=0.02)
    model, data = load_mujoco_model(MJCF_PATH, stiffness, damping, PHYSICS_DT)
    body_pos_1 = model.body_pos[1].copy()

    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)

    # Fix height: same bilateral foot grounding as run_smpl_rl_tracker main()
    # (lines 1250-1380 in run_smpl_rl_tracker.py)
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    left_foot_body_ids = set()
    right_foot_body_ids = set()
    for bid in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in ("L_Ankle", "L_Toe"):
            left_foot_body_ids.add(bid)
        elif bname in ("R_Ankle", "R_Toe"):
            right_foot_body_ids.add(bid)

    def _compute_lowest_geom_z(body_id_set, model, data):
        min_z = float("inf")
        min_gname = ""
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] not in body_id_set:
                continue
            gtype = int(model.geom_type[gid])
            gsize = model.geom_size[gid]
            gxpos = data.geom_xpos[gid]
            gxmat = data.geom_xmat[gid].reshape(3, 3)
            if gtype == 5:  # capsule
                z_ext = abs(gxmat[2, 2]) * gsize[1] + gsize[0]
                bottom = gxpos[2] - z_ext
            elif gtype == 3:  # sphere
                bottom = gxpos[2] - gsize[0]
            elif gtype == 6:  # box
                z_ext = sum(abs(gxmat[2, j]) * gsize[j] for j in range(3))
                bottom = gxpos[2] - z_ext
            else:
                bottom = gxpos[2]
            if bottom < min_z:
                min_z = bottom
                min_gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"g{gid}"
        return min_z, min_gname

    left_min_z, left_gname = _compute_lowest_geom_z(left_foot_body_ids, model, data)
    right_min_z, right_gname = _compute_lowest_geom_z(right_foot_body_ids, model, data)

    TARGET_GEOM_SURFACE_Z = 0.0
    FOOT_SWING_THRESHOLD = 0.08
    foot_height_diff = abs(left_min_z - right_min_z)
    grounding_ref_z = min(left_min_z, right_min_z)
    height_shift = TARGET_GEOM_SURFACE_Z - grounding_ref_z

    if abs(height_shift) > 0.0001:
        ref_qpos[:, 2] += height_shift

    print(f"  [run_smpl] height_shift = {height_shift:+.6f}m")
    print(f"  [run_smpl] root_h after = {ref_qpos[0, 2]:.6f}m")

    # Apply REF_RESPAWN_OFFSET = 0.0 (matching current run_smpl)
    ref_qpos[:, 2] += 0.0

    dt_ref = 1.0 / fps

    # Precompute reference maxcoords
    ref_data = precompute_reference_maxcoords(model, data, ref_qpos, dt_ref)

    # Set initial pose (same as run_smpl_rl_tracker lines 863-887)
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    data.ctrl[:] = ref_qpos[0, 7:]
    mujoco.mj_forward(model, data)

    # Extract sim state (same as run_smpl_rl_tracker)
    cur_state = extract_sim_state(model, data, 24)

    # Future reference (same as run_smpl_rl_tracker simulation loop)
    sim_time = 0.0
    future_step_indices = YAML_META["motion"]["future_step_indices"]
    future_dt_seconds = YAML_META["motion"]["future_dt_seconds"]
    T_ref = ref_qpos.shape[0]

    future_states = []
    for fi, fdt in zip(future_step_indices, future_dt_seconds):
        future_time = sim_time + fi * fdt
        ref_frame_idx = min(int(future_time / dt_ref), T_ref - 1)
        future_ref = {k: v[ref_frame_idx].copy() for k, v in ref_data.items()}
        future_states.append(future_ref)

    print(f"  [run_smpl] future_step_indices = {future_step_indices}")
    print(f"  [run_smpl] future_dt_seconds = {future_dt_seconds}")
    # Show the actual frame indices
    for fi, fdt in zip(future_step_indices, future_dt_seconds):
        future_time = sim_time + fi * fdt
        ref_frame_idx = min(int(future_time / dt_ref), T_ref - 1)
        print(f"  [run_smpl] future: fi={fi}, fdt={fdt}, time={future_time:.4f}, frame={ref_frame_idx}")

    future_body_pos = np.stack([fs["body_pos"] for fs in future_states], axis=0)
    future_body_rot = np.stack([fs["body_rot"] for fs in future_states], axis=0)
    future_body_vel = np.stack([fs["body_vel"] for fs in future_states], axis=0)
    future_body_ang_vel = np.stack([fs["body_ang_vel"] for fs in future_states], axis=0)

    # Build inputs with ONNX names (same as test_physics_configs naming)
    inputs = {
        "current_rigid_body_ang_vel": cur_state["body_ang_vel"][None].astype(np.float32),
        "current_rigid_body_pos": cur_state["body_pos"][None].astype(np.float32),
        "current_rigid_body_rot": cur_state["body_rot"][None].astype(np.float32),
        "current_rigid_body_vel": cur_state["body_vel"][None].astype(np.float32),
        "ground_heights": np.zeros(1, dtype=np.float32),
        "historical_actions": np.zeros((1, 1, 69), dtype=np.float32),
        "mimic_future_ang_vel": future_body_ang_vel[None].astype(np.float32),
        "mimic_future_pos": future_body_pos[None].astype(np.float32),
        "mimic_future_rot": future_body_rot[None].astype(np.float32),
        "mimic_future_vel": future_body_vel[None].astype(np.float32),
    }
    return inputs, ref_qpos, ref_data["body_pos"], ref_data["body_rot"], ref_data["body_vel"], ref_data["body_ang_vel"]


def main():
    print("=" * 70)
    print("  Comparing ONNX inputs: test_physics_configs vs run_smpl_rl_tracker")
    print("=" * 70)

    print("\n[1] Loading test_physics_configs path...")
    inputs_test, qpos_test, bp_test, br_test, bv_test, bav_test = get_inputs_from_test_physics_configs()
    print("\n[2] Loading run_smpl_rl_tracker path...")
    inputs_run, qpos_run, bp_run, br_run, bv_run, bav_run = get_inputs_from_run_smpl_rl_tracker()

    print("\n--- Reference qpos comparison ---")
    qpos_diff = np.abs(qpos_test - qpos_run).max()
    print(f"  max |qpos_test - qpos_run| = {qpos_diff:.8f}")
    if qpos_diff > 1e-6:
        frame_diffs = np.abs(qpos_test - qpos_run).max(axis=1)
        worst_frame = np.argmax(frame_diffs)
        print(f"  Worst frame: {worst_frame}, diff={frame_diffs[worst_frame]:.8f}")
        print(f"  qpos_test[{worst_frame}][:10] = {qpos_test[worst_frame][:10]}")
        print(f"  qpos_run [{worst_frame}][:10] = {qpos_run[worst_frame][:10]}")

    print("\n--- Reference body_pos comparison (all frames) ---")
    bp_diff = np.abs(bp_test - bp_run).max()
    print(f"  max |body_pos_test - body_pos_run| = {bp_diff:.8f}")

    print("\n--- Reference body_rot comparison (all frames) ---")
    br_diff = np.abs(br_test - br_run).max()
    print(f"  max |body_rot_test - body_rot_run| = {br_diff:.8f}")

    print("\n--- Reference body_vel comparison (all frames) ---")
    bv_diff = np.abs(bv_test - bv_run)
    print(f"  max |body_vel_test - body_vel_run| = {bv_diff.max():.8f}")
    print(f"  mean = {bv_diff.mean():.8f}")
    frame_vel_diffs = bv_diff.max(axis=(1, 2))
    worst_frame = np.argmax(frame_vel_diffs)
    print(f"  Worst frame: {worst_frame}, diff={frame_vel_diffs[worst_frame]:.8f}")
    if bv_diff.max() > 1e-4:
        print(f"  body_vel_test[0, 0] = {bv_test[0, 0]}")
        print(f"  body_vel_run [0, 0] = {bv_run[0, 0]}")
        print(f"  body_vel_test[1, 0] = {bv_test[1, 0]}")
        print(f"  body_vel_run [1, 0] = {bv_run[1, 0]}")

    print("\n--- Reference body_ang_vel comparison (all frames) ---")
    bav_diff = np.abs(bav_test - bav_run)
    print(f"  max |body_ang_vel_test - body_ang_vel_run| = {bav_diff.max():.8f}")
    print(f"  mean = {bav_diff.mean():.8f}")
    worst_frame = np.argmax(bav_diff.max(axis=(1, 2)))
    print(f"  Worst frame: {worst_frame}, diff={bav_diff.max(axis=(1,2))[worst_frame]:.8f}")
    if bav_diff.max() > 1e-4:
        print(f"  body_ang_vel_test[0, 0] = {bav_test[0, 0]}")
        print(f"  body_ang_vel_run [0, 0] = {bav_run[0, 0]}")
        print(f"  body_ang_vel_test[1, 0] = {bav_test[1, 0]}")
        print(f"  body_ang_vel_run [1, 0] = {bav_run[1, 0]}")

    print("\n--- ONNX input comparison (step 0) ---")
    for key in sorted(inputs_test.keys()):
        if key not in inputs_run:
            print(f"  {key}: MISSING in run_smpl!")
            continue
        arr_test = inputs_test[key]
        arr_run = inputs_run[key]
        if arr_test.shape != arr_run.shape:
            print(f"  {key}: SHAPE MISMATCH! test={arr_test.shape}, run={arr_run.shape}")
            continue
        diff = np.abs(arr_test - arr_run)
        max_diff = diff.max()
        mean_diff = diff.mean()
        if max_diff > 1e-6:
            print(f"  {key}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f} *** DIFFERENT ***")
            flat_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"    worst at {flat_idx}: test={arr_test[flat_idx]:.8f}, run={arr_run[flat_idx]:.8f}")
        else:
            print(f"  {key}: max_diff={max_diff:.10f} (MATCH)")

    # Now run ONNX with BOTH sets of inputs and compare outputs
    import onnxruntime as ort
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]

    outputs_test = session.run(out_names, inputs_test)
    outputs_run = session.run(out_names, inputs_run)

    print("\n--- ONNX output comparison (step 0) ---")
    for name, out_t, out_r in zip(out_names, outputs_test, outputs_run):
        diff = np.abs(out_t - out_r)
        max_diff = diff.max()
        print(f"  {name}: max_diff={max_diff:.8f}")
        if name == "joint_pos_targets":
            print(f"    test[:10] = {out_t.squeeze()[:10]}")
            print(f"    run [:10] = {out_r.squeeze()[:10]}")

    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    any_diff = False
    for key in sorted(inputs_test.keys()):
        if key in inputs_run:
            max_diff = np.abs(inputs_test[key] - inputs_run[key]).max()
            if max_diff > 1e-4:
                any_diff = True
                print(f"  SIGNIFICANT DIFF in '{key}': max={max_diff:.6f}")
    if not any_diff:
        print("  All ONNX inputs are IDENTICAL (within 1e-4)")
        print("  If outputs differ, the problem is NOT in step-0 inputs")
        print("  → Check: physics config differences, ctrl application, dynamic gains usage")


if __name__ == "__main__":
    main()
