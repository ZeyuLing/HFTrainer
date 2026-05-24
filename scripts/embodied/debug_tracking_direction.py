"""Diagnostic: Does the ONNX policy output targets that point toward the reference?

Runs the RL tracker for a few steps and checks:
1. Whether joint_pos_targets point toward the next reference DOF positions
2. Whether the PD error is in the correct direction
3. What magnitude of torque the PD controller produces
4. Whether the reference is changing over time (not stuck)
"""

import sys
import os
import numpy as np
import mujoco
import yaml
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_smpl_rl_tracker import (
    precompute_reference_maxcoords,
    get_reference_at_time,
    extract_sim_state,
    extract_body_com_offsets,
    apply_heading_offset_np,
    compute_yaw_offset_np,
    mujoco_wxyz_to_xyzw,
    load_mujoco_model,
    decode_motion_135,
    yup_to_zup,
    smpl_to_qpos,
)


def main():
    base_dir = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
    onnx_path = f"{base_dir}/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.onnx"
    yaml_path = f"{base_dir}/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.yaml"
    mjcf_path = f"{base_dir}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"

    # Load a test motion
    motion_dir = f"{base_dir}/output/embodied_t2m_v4/data/npz"
    npz_files = [f for f in os.listdir(motion_dir) if f.endswith('.npz')]
    if not npz_files:
        print("ERROR: No NPZ files found")
        return

    test_file = os.path.join(motion_dir, npz_files[0])
    print(f"Using test motion: {test_file}")

    # Decode using the same function as run_smpl_rl_tracker.process_single_motion
    smpl_pose, transl, motion_fps = decode_motion_135(test_file)
    print(f"Decoded: {smpl_pose.shape[0]} frames @ {motion_fps}fps")

    # Y-up to Z-up
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    # Load MuJoCo model for conversion
    model_tmp = mujoco.MjModel.from_xml_path(mjcf_path)
    body_pos_1 = model_tmp.body_pos[1].copy()

    # SMPL -> qpos (same as process_single_motion)
    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)
    print(f"ref_qpos shape: {ref_qpos.shape}")

    # Ground height fix (simplified: place lowest foot geom at floor)
    data_tmp = mujoco.MjData(model_tmp)
    data_tmp.qpos[:] = ref_qpos[0]
    data_tmp.qvel[:] = 0.0
    mujoco.mj_forward(model_tmp, data_tmp)

    # Find lowest geom Z at frame 0
    min_geom_z = float('inf')
    for gid in range(model_tmp.ngeom):
        bid = model_tmp.geom_bodyid[gid]
        if bid < 1:
            continue  # skip world body
        gtype = int(model_tmp.geom_type[gid])
        gsize = model_tmp.geom_size[gid]
        gxpos = data_tmp.geom_xpos[gid]
        gxmat = data_tmp.geom_xmat[gid].reshape(3, 3)
        if gtype == 5:  # capsule
            radius = gsize[0]
            half_len = gsize[1]
            z_ext = abs(gxmat[2, 2]) * half_len + radius
            bottom_z = gxpos[2] - z_ext
        elif gtype == 3:  # sphere
            bottom_z = gxpos[2] - gsize[0]
        elif gtype == 6:  # box
            half_extents = gsize[:3]
            z_ext = (abs(gxmat[2, 0]) * half_extents[0] +
                     abs(gxmat[2, 1]) * half_extents[1] +
                     abs(gxmat[2, 2]) * half_extents[2])
            bottom_z = gxpos[2] - z_ext
        else:
            bottom_z = gxpos[2]
        min_geom_z = min(min_geom_z, bottom_z)

    height_shift = 0.0 - min_geom_z  # Place lowest geom at floor
    ref_qpos[:, 2] += height_shift
    print(f"Ground height shift: {height_shift:+.4f}m")
    print(f"Root height after fix: [{ref_qpos[:, 2].min():.4f}, {ref_qpos[:, 2].max():.4f}]")
    del model_tmp, data_tmp

    # Load YAML
    with open(yaml_path) as f:
        yaml_meta = yaml.safe_load(f)

    timing = yaml_meta["timing"]
    control = yaml_meta["control"]
    runtime = yaml_meta["_runtime"]

    control_dt = timing["control_dt"]  # 0.02
    decimation = timing["decimation"]  # 20
    physics_dt = timing["physics_dt"]  # 0.001
    num_bodies = 24
    num_dofs = 69
    dt_ref = 1.0 / motion_fps
    T_ref = ref_qpos.shape[0]
    future_step_indices = yaml_meta["motion"]["future_step_indices"]
    future_dt_seconds = yaml_meta["motion"]["future_dt_seconds"]
    onnx_name_to_key = runtime["onnx_name_to_in_key"]
    stiffness = control["stiffness"]
    damping_ctrl = control["damping"]

    # Load MuJoCo model with proper actuator setup
    model, data = load_mujoco_model(mjcf_path, stiffness, damping_ctrl, physics_dt)
    body_com_offsets = extract_body_com_offsets(model, num_bodies)

    # Precompute reference
    ref_data = precompute_reference_maxcoords(model, data, ref_qpos, dt_ref)

    # Initialize from reference
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    data.ctrl[:] = ref_qpos[0, 7:]
    mujoco.mj_forward(model, data)

    # Load ONNX
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual_in_names = [inp.name for inp in session.get_inputs()]
    actual_out_names = [out.name for out in session.get_outputs()]

    # Heading offset
    sim_state_0 = extract_sim_state(model, data, num_bodies, body_com_offsets)
    heading_offset = compute_yaw_offset_np(
        sim_state_0["body_rot"][0], ref_data["body_rot"][0, 0])
    print(f"Heading offset: {heading_offset}")

    # Run simulation
    prev_actions = np.zeros(num_dofs, dtype=np.float32)

    N_STEPS = 30  # Run 30 steps (0.6s at 50Hz)

    print("\n" + "="*100)
    print("STEP-BY-STEP TRACKING DIAGNOSTIC")
    print("="*100)
    print(f"{'step':>4} | {'cos(tgt,ref)':>12} | {'|tgt-cur|':>10} | {'|ref-cur|':>10} | "
          f"{'|tgt|':>8} | {'|cur|':>8} | {'|ref_fut|':>8} | {'root_err':>8} | {'mpjpe':>8}")
    print("-"*100)

    for step_idx in range(N_STEPS):
        sim_time = step_idx * control_dt

        # Current state
        cur_state = extract_sim_state(model, data, num_bodies, body_com_offsets)
        cur_dof = data.qpos[7:].copy()

        # Reference at current time
        ref_now = get_reference_at_time(ref_data, sim_time, dt_ref, T_ref)

        # Future reference
        future_states = []
        for fi, fdt in zip(future_step_indices, future_dt_seconds):
            future_time = sim_time + fi * fdt
            future_ref = get_reference_at_time(ref_data, future_time, dt_ref, T_ref)
            future_ref["body_rot"] = apply_heading_offset_np(
                heading_offset, future_ref["body_rot"])
            future_states.append(future_ref)

        future_body_pos = np.stack([fs["body_pos"] for fs in future_states], axis=0)
        future_body_rot = np.stack([fs["body_rot"] for fs in future_states], axis=0)
        future_body_vel = np.stack([fs["body_vel"] for fs in future_states], axis=0)
        future_body_ang_vel = np.stack([fs["body_ang_vel"] for fs in future_states], axis=0)

        # Reference DOF at future time (what we want to track)
        ref_future_dof = get_reference_at_time(
            ref_data, sim_time + future_dt_seconds[0], dt_ref, T_ref)["dof_pos"]

        # Build ONNX inputs
        key_to_array = {
            "current.rigid_body_pos": cur_state["body_pos"][None],
            "current.rigid_body_rot": cur_state["body_rot"][None],
            "current.rigid_body_vel": cur_state["body_vel"][None],
            "current.rigid_body_ang_vel": cur_state["body_ang_vel"][None],
            "ground_heights": np.zeros(1, dtype=np.float32),
            "historical.actions": prev_actions[None, None],
            "mimic.future_pos": future_body_pos[None],
            "mimic.future_rot": future_body_rot[None],
            "mimic.future_vel": future_body_vel[None],
            "mimic.future_ang_vel": future_body_ang_vel[None],
        }

        onnx_inputs = {}
        for onnx_name, sem_key in onnx_name_to_key.items():
            if sem_key in key_to_array:
                onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)

        # ONNX inference
        ort_out = session.run(actual_out_names, onnx_inputs)
        out_dict = {name: val for name, val in zip(actual_out_names, ort_out)}
        joint_pos_targets = out_dict["joint_pos_targets"].squeeze().copy()

        # ====== CORE DIAGNOSTIC ======
        # Direction from current to target (what PD controller will do)
        delta_target = joint_pos_targets - cur_dof  # PD error direction
        # Direction from current to reference (what we WANT to happen)
        delta_ref = ref_future_dof - cur_dof  # Desired direction

        # Cosine similarity: are they pointing the same way?
        norm_tgt = np.linalg.norm(delta_target)
        norm_ref = np.linalg.norm(delta_ref)
        if norm_tgt > 1e-8 and norm_ref > 1e-8:
            cos_sim = np.dot(delta_target, delta_ref) / (norm_tgt * norm_ref)
        else:
            cos_sim = 0.0

        # Position tracking error (MPJPE)
        mpjpe = np.sqrt(((cur_state["body_pos"] - ref_now["body_pos"])**2).sum(-1)).mean()

        # Root position error
        root_err = np.linalg.norm(cur_state["body_pos"][0] - ref_now["body_pos"][0])

        print(f"{step_idx:4d} | {cos_sim:12.4f} | {norm_tgt:10.4f} | {norm_ref:10.4f} | "
              f"{np.linalg.norm(joint_pos_targets):8.4f} | {np.linalg.norm(cur_dof):8.4f} | "
              f"{np.linalg.norm(ref_future_dof):8.4f} | {root_err:8.4f} | {mpjpe:8.4f}")

        # Extra diagnostics at specific steps
        if step_idx in [0, 5, 15, 29]:
            print(f"       --- Step {step_idx} details ---")
            print(f"       joint_pos_targets[:6] = {joint_pos_targets[:6]}")
            print(f"       cur_dof[:6]           = {cur_dof[:6]}")
            print(f"       ref_future_dof[:6]    = {ref_future_dof[:6]}")
            print(f"       delta_target[:6]      = {delta_target[:6]}")
            print(f"       delta_ref[:6]         = {delta_ref[:6]}")

            # Check per-DOF agreement (sign match)
            sign_match = np.sign(delta_target) == np.sign(delta_ref)
            # Only count DOFs where ref is actually moving (|delta_ref| > threshold)
            moving_dofs = np.abs(delta_ref) > 0.01  # > 0.01 rad
            if moving_dofs.sum() > 0:
                match_rate = sign_match[moving_dofs].mean()
                print(f"       Sign match rate (moving DOFs): {match_rate:.2%} "
                      f"({moving_dofs.sum()} DOFs moving)")

            # Root body state
            print(f"       cur_root_pos = {cur_state['body_pos'][0]}")
            print(f"       ref_root_pos = {ref_now['body_pos'][0]}")
            print(f"       cur_root_rot(xyzw) = {cur_state['body_rot'][0]}")
            print(f"       ref_root_rot(xyzw) = {ref_now['body_rot'][0]}")

            # Stiffness/damping values
            if "stiffness_targets" in out_dict:
                st = out_dict["stiffness_targets"].squeeze()
                dt_ = out_dict["damping_targets"].squeeze()
                print(f"       stiffness[:3] = {st[:3]} (kp)")
                print(f"       damping[:3]   = {dt_[:3]} (kd)")
                # Expected torque: kp * delta_target
                expected_torque = st * delta_target
                print(f"       expected_torque[:6] = {expected_torque[:6]}")
                print(f"       expected_torque L2 = {np.linalg.norm(expected_torque):.2f}")
            print()

        # Apply dynamic PD gains
        if "stiffness_targets" in out_dict and "damping_targets" in out_dict:
            stiff_out = out_dict["stiffness_targets"].squeeze()
            damp_out = out_dict["damping_targets"].squeeze()
            for i in range(model.nu):
                kp = float(stiff_out[i])
                kd = float(damp_out[i])
                model.actuator_gainprm[i, 0] = kp
                model.actuator_biasprm[i, 1] = -kp
                model.actuator_biasprm[i, 2] = -kd

        prev_actions = out_dict["actions"].squeeze().copy()

        # Apply control and step
        data.ctrl[:] = joint_pos_targets
        for _ in range(decimation):
            mujoco.mj_step(model, data)

    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    # Final state comparison
    final_state = extract_sim_state(model, data, num_bodies, body_com_offsets)
    final_ref = get_reference_at_time(ref_data, N_STEPS * control_dt, dt_ref, T_ref)
    final_mpjpe = np.sqrt(((final_state["body_pos"] - final_ref["body_pos"])**2).sum(-1)).mean()

    # Compare DOF positions at the end
    final_dof = data.qpos[7:]
    ref_final_dof = get_reference_at_time(ref_data, N_STEPS * control_dt, dt_ref, T_ref)["dof_pos"]
    dof_err = np.abs(final_dof - ref_final_dof)

    print(f"Final MPJPE after {N_STEPS} steps: {final_mpjpe:.4f} m")
    print(f"Final DOF error: mean={dof_err.mean():.4f} rad, max={dof_err.max():.4f} rad")
    print(f"Final DOF error per joint group:")

    joint_names = yaml_meta["joint_names"]
    # Group by limb
    groups = {
        "L_Leg": [0,1,2,3,4,5,6,7,8,9,10,11],
        "R_Leg": [12,13,14,15,16,17,18,19,20,21,22,23],
        "Torso": [24,25,26,27,28,29,30,31,32],
        "Head":  [33,34,35,36,37,38],
        "L_Arm": [39,40,41,42,43,44,45,46,47,48,49,50,51,52,53],
        "R_Arm": [54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],
    }
    for gname, indices in groups.items():
        gerr = dof_err[indices]
        print(f"  {gname:8s}: mean={gerr.mean():.4f} rad ({np.degrees(gerr.mean()):.2f}°), "
              f"max={gerr.max():.4f} rad ({np.degrees(gerr.max()):.2f}°)")


if __name__ == "__main__":
    main()
