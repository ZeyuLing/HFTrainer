"""Diagnostic: Test double-heading-correction hypothesis and action module behavior.

Key tests:
1. Feed ONNX with future_reference = current_state (zero error)
   → Policy should output joint_pos_targets ≈ current DOF positions
2. Compare with heading_offset applied vs NOT applied to future reference
3. Check if the actions→joint_pos_targets scaling matches expected BUILT_IN_PD formula

If test 1 fails (targets ≠ current DOF even with zero error), the issue is in how
current state is represented to the ONNX, not in the heading correction.
"""

import sys
import os
import numpy as np
import mujoco
import yaml

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


def run_onnx_inference(session, onnx_name_to_key, actual_in_names, actual_out_names,
                       cur_state, future_body_pos, future_body_rot,
                       future_body_vel, future_body_ang_vel, prev_actions, num_dofs=69):
    """Run a single ONNX inference with given inputs."""
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

    ort_out = session.run(actual_out_names, onnx_inputs)
    return {name: val for name, val in zip(actual_out_names, ort_out)}


def main():
    base_dir = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
    onnx_path = f"{base_dir}/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.onnx"
    yaml_path = f"{base_dir}/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.yaml"
    mjcf_path = f"{base_dir}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"

    # Load a test motion
    motion_dir = f"{base_dir}/output/embodied_t2m_v4/data/npz"
    npz_files = sorted([f for f in os.listdir(motion_dir) if f.endswith('.npz')])
    if not npz_files:
        print("ERROR: No NPZ files found")
        return

    test_file = os.path.join(motion_dir, npz_files[0])
    print(f"Using test motion: {test_file}")

    # Decode and convert
    smpl_pose, transl, motion_fps = decode_motion_135(test_file)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    model_tmp = mujoco.MjModel.from_xml_path(mjcf_path)
    body_pos_1 = model_tmp.body_pos[1].copy()
    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)

    # Ground height fix
    data_tmp = mujoco.MjData(model_tmp)
    data_tmp.qpos[:] = ref_qpos[0]
    data_tmp.qvel[:] = 0.0
    mujoco.mj_forward(model_tmp, data_tmp)

    min_geom_z = float('inf')
    for gid in range(model_tmp.ngeom):
        bid = model_tmp.geom_bodyid[gid]
        if bid < 1:
            continue
        gtype = int(model_tmp.geom_type[gid])
        gsize = model_tmp.geom_size[gid]
        gxpos = data_tmp.geom_xpos[gid]
        gxmat = data_tmp.geom_xmat[gid].reshape(3, 3)
        if gtype == 5:
            radius = gsize[0]
            half_len = gsize[1]
            z_ext = abs(gxmat[2, 2]) * half_len + radius
            bottom_z = gxpos[2] - z_ext
        elif gtype == 3:
            bottom_z = gxpos[2] - gsize[0]
        elif gtype == 6:
            half_extents = gsize[:3]
            z_ext = (abs(gxmat[2, 0]) * half_extents[0] +
                     abs(gxmat[2, 1]) * half_extents[1] +
                     abs(gxmat[2, 2]) * half_extents[2])
            bottom_z = gxpos[2] - z_ext
        else:
            bottom_z = gxpos[2]
        min_geom_z = min(min_geom_z, bottom_z)

    height_shift = 0.0 - min_geom_z
    ref_qpos[:, 2] += height_shift
    del model_tmp, data_tmp

    # Load YAML
    with open(yaml_path) as f:
        yaml_meta = yaml.safe_load(f)

    timing = yaml_meta["timing"]
    control = yaml_meta["control"]
    runtime = yaml_meta["_runtime"]

    control_dt = timing["control_dt"]
    physics_dt = timing["physics_dt"]
    decimation = timing["decimation"]
    num_bodies = 24
    num_dofs = 69
    dt_ref = 1.0 / motion_fps
    T_ref = ref_qpos.shape[0]
    future_step_indices = yaml_meta["motion"]["future_step_indices"]
    future_dt_seconds = yaml_meta["motion"]["future_dt_seconds"]
    onnx_name_to_key = runtime["onnx_name_to_in_key"]
    stiffness = control["stiffness"]
    damping_ctrl = control["damping"]

    # Load MuJoCo model
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

    # Current state
    cur_state = extract_sim_state(model, data, num_bodies, body_com_offsets)
    cur_dof = data.qpos[7:].copy()

    # Heading offset (should be ~identity at step 0)
    heading_offset = compute_yaw_offset_np(
        cur_state["body_rot"][0], ref_data["body_rot"][0, 0])
    print(f"\nHeading offset: {heading_offset}")
    print(f"  (should be ~[0, 0, 0, 1] since robot starts at reference)")

    # Previous actions (zeros at step 0)
    prev_actions = np.zeros(num_dofs, dtype=np.float32)

    # ====================================================================
    # TEST 1: Zero-error condition (future_ref = current_state)
    # If observation construction is correct, policy should output
    # joint_pos_targets ≈ current DOF positions ("stay in place")
    # ====================================================================
    print("\n" + "=" * 80)
    print("TEST 1: ZERO-ERROR CONDITION (future_ref = current_state)")
    print("=" * 80)
    print("  Feeding current body state as the future reference.")
    print("  Expected: joint_pos_targets ≈ current DOF (maintain pose)")

    # Use current state as both current AND future reference
    future_body_pos = cur_state["body_pos"][None]  # (1, 24, 3)
    future_body_rot = cur_state["body_rot"][None]  # (1, 24, 4)
    future_body_vel = cur_state["body_vel"][None]  # (1, 24, 3) = zeros
    future_body_ang_vel = cur_state["body_ang_vel"][None]  # (1, 24, 3) = zeros

    out1 = run_onnx_inference(session, onnx_name_to_key, actual_in_names, actual_out_names,
                              cur_state, future_body_pos, future_body_rot,
                              future_body_vel, future_body_ang_vel, prev_actions)

    jpt1 = out1["joint_pos_targets"].squeeze()
    act1 = out1["actions"].squeeze()

    print(f"\n  Results:")
    print(f"    cur_dof[:6]           = {cur_dof[:6]}")
    print(f"    joint_pos_targets[:6] = {jpt1[:6]}")
    print(f"    actions[:6]           = {act1[:6]}")
    print(f"    |jpt - cur_dof|       = {np.linalg.norm(jpt1 - cur_dof):.4f} rad")
    print(f"    |jpt - cur_dof| mean  = {np.abs(jpt1 - cur_dof).mean():.4f} rad")
    print(f"    |jpt - cur_dof| max   = {np.abs(jpt1 - cur_dof).max():.4f} rad")
    print(f"    jpt L2 norm           = {np.linalg.norm(jpt1):.4f}")
    print(f"    cur_dof L2 norm       = {np.linalg.norm(cur_dof):.4f}")
    print(f"    actions L2 norm       = {np.linalg.norm(act1):.4f}")

    # Check ratio: jpt / cur_dof (should be ~1.0 if targets match current)
    nonzero_mask = np.abs(cur_dof) > 0.01
    if nonzero_mask.sum() > 0:
        ratio = jpt1[nonzero_mask] / cur_dof[nonzero_mask]
        print(f"\n    Ratio jpt/cur_dof (nonzero DOFs):")
        print(f"      mean = {ratio.mean():.4f}")
        print(f"      std  = {ratio.std():.4f}")
        print(f"      min  = {ratio.min():.4f}")
        print(f"      max  = {ratio.max():.4f}")

    # ====================================================================
    # TEST 2: Normal future reference (with heading offset applied)
    # This is what the current code does
    # ====================================================================
    print("\n" + "=" * 80)
    print("TEST 2: NORMAL FUTURE REFERENCE (with heading_offset applied)")
    print("=" * 80)

    sim_time = 0.0
    future_time = sim_time + future_step_indices[0] * future_dt_seconds[0]
    future_ref = get_reference_at_time(ref_data, future_time, dt_ref, T_ref)

    # Apply heading offset (current code does this)
    future_ref_rot_with_heading = apply_heading_offset_np(
        heading_offset, future_ref["body_rot"])

    future_body_pos_2 = future_ref["body_pos"][None]
    future_body_rot_2 = future_ref_rot_with_heading[None]
    future_body_vel_2 = future_ref["body_vel"][None]
    future_body_ang_vel_2 = future_ref["body_ang_vel"][None]

    out2 = run_onnx_inference(session, onnx_name_to_key, actual_in_names, actual_out_names,
                              cur_state, future_body_pos_2, future_body_rot_2,
                              future_body_vel_2, future_body_ang_vel_2, prev_actions)

    jpt2 = out2["joint_pos_targets"].squeeze()
    act2 = out2["actions"].squeeze()

    # Future reference DOF
    ref_future_dof = future_ref["dof_pos"]

    print(f"\n  Results:")
    print(f"    cur_dof[:6]           = {cur_dof[:6]}")
    print(f"    ref_future_dof[:6]    = {ref_future_dof[:6]}")
    print(f"    joint_pos_targets[:6] = {jpt2[:6]}")
    print(f"    actions[:6]           = {act2[:6]}")
    print(f"    |jpt - cur_dof|       = {np.linalg.norm(jpt2 - cur_dof):.4f} rad")
    print(f"    |jpt - ref_future|    = {np.linalg.norm(jpt2 - ref_future_dof):.4f} rad")

    # ====================================================================
    # TEST 3: Future reference WITHOUT heading offset (raw world frame)
    # Test the double-application hypothesis
    # ====================================================================
    print("\n" + "=" * 80)
    print("TEST 3: FUTURE REFERENCE WITHOUT heading_offset (raw world frame)")
    print("=" * 80)
    print("  If ONNX internally normalizes by heading, raw world frame should be correct.")
    print("  If this gives BETTER targets, heading_offset is being double-applied.")

    future_body_rot_3 = future_ref["body_rot"][None]  # NO heading offset

    out3 = run_onnx_inference(session, onnx_name_to_key, actual_in_names, actual_out_names,
                              cur_state, future_body_pos_2, future_body_rot_3,
                              future_body_vel_2, future_body_ang_vel_2, prev_actions)

    jpt3 = out3["joint_pos_targets"].squeeze()
    act3 = out3["actions"].squeeze()

    print(f"\n  Results:")
    print(f"    cur_dof[:6]           = {cur_dof[:6]}")
    print(f"    ref_future_dof[:6]    = {ref_future_dof[:6]}")
    print(f"    joint_pos_targets[:6] = {jpt3[:6]}")
    print(f"    actions[:6]           = {act3[:6]}")
    print(f"    |jpt - cur_dof|       = {np.linalg.norm(jpt3 - cur_dof):.4f} rad")
    print(f"    |jpt - ref_future|    = {np.linalg.norm(jpt3 - ref_future_dof):.4f} rad")

    # ====================================================================
    # TEST 4: Check actions→joint_pos_targets relationship
    # BUILT_IN_PD formula: joint_pos_targets = offset + scale * actions
    # For symmetric limits [-π, π]: offset=0, scale=π
    # ====================================================================
    print("\n" + "=" * 80)
    print("TEST 4: ACTIONS → JOINT_POS_TARGETS RELATIONSHIP")
    print("=" * 80)
    print("  BUILT_IN_PD formula: jpt = pd_action_offset + pd_action_scale * actions")
    print("  If limits are [-π, π]: offset=0, scale=π → jpt = π * actions")

    # Check if jpt ≈ π * actions
    expected_pi_scale = np.pi * act2
    err_pi = np.linalg.norm(jpt2 - expected_pi_scale)
    print(f"\n  Test jpt = π * actions:")
    print(f"    |jpt - π*actions| = {err_pi:.4f}")
    print(f"    Ratio jpt/actions (where |actions|>0.01):")
    actions_nonzero = np.abs(act2) > 0.01
    if actions_nonzero.sum() > 0:
        ratio_act = jpt2[actions_nonzero] / act2[actions_nonzero]
        print(f"      mean = {ratio_act.mean():.4f} (π≈3.1416)")
        print(f"      std  = {ratio_act.std():.4f}")
        print(f"      min  = {ratio_act.min():.4f}")
        print(f"      max  = {ratio_act.max():.4f}")

    # Also check: is there an offset?
    # If jpt = offset + scale * actions, solve for offset and scale
    # Using least squares: [1, actions] @ [offset, scale]^T = jpt
    A = np.column_stack([np.ones(num_dofs), act2])
    result = np.linalg.lstsq(A, jpt2, rcond=None)
    offset_fit, scale_fit = result[0]
    residual = jpt2 - (offset_fit + scale_fit * act2)
    print(f"\n  Least-squares fit: jpt = {offset_fit:.4f} + {scale_fit:.4f} * actions")
    print(f"    Residual L2 = {np.linalg.norm(residual):.6f}")
    print(f"    (Expected: offset≈0, scale≈π for symmetric limits)")

    # ====================================================================
    # TEST 5: What happens with a KNOWN pose?
    # Set robot to T-pose (all zeros) and reference to T-pose
    # Policy should output targets ≈ 0 (maintain T-pose)
    # ====================================================================
    print("\n" + "=" * 80)
    print("TEST 5: T-POSE (all DOF zeros) — KNOWN BASELINE")
    print("=" * 80)
    print("  If policy sees 'I am at T-pose and should stay at T-pose',")
    print("  it should output targets ≈ 0.")

    # Set all DOFs to zero (T-pose)
    data.qpos[:] = 0.0
    data.qpos[2] = 0.93  # Reasonable standing height
    data.qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion (wxyz)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    cur_state_tpose = extract_sim_state(model, data, num_bodies, body_com_offsets)

    # Future = current (zero error, at T-pose)
    future_pos_t = cur_state_tpose["body_pos"][None]
    future_rot_t = cur_state_tpose["body_rot"][None]
    future_vel_t = cur_state_tpose["body_vel"][None]
    future_angvel_t = cur_state_tpose["body_ang_vel"][None]

    out5 = run_onnx_inference(session, onnx_name_to_key, actual_in_names, actual_out_names,
                              cur_state_tpose, future_pos_t, future_rot_t,
                              future_vel_t, future_angvel_t, prev_actions)

    jpt5 = out5["joint_pos_targets"].squeeze()
    act5 = out5["actions"].squeeze()
    cur_dof_tpose = data.qpos[7:].copy()  # Should be all zeros

    print(f"\n  Results:")
    print(f"    cur_dof[:6]           = {cur_dof_tpose[:6]}")
    print(f"    joint_pos_targets[:6] = {jpt5[:6]}")
    print(f"    actions[:6]           = {act5[:6]}")
    print(f"    |jpt|                 = {np.linalg.norm(jpt5):.4f}")
    print(f"    |actions|             = {np.linalg.norm(act5):.4f}")
    print(f"    |jpt - 0|             = {np.linalg.norm(jpt5):.4f} (should be ~0)")

    # ====================================================================
    # TEST 6: Check what ONNX input names actually map to
    # Print the exact tensor values being fed
    # ====================================================================
    print("\n" + "=" * 80)
    print("TEST 6: EXACT ONNX INPUT SHAPES AND VALUES")
    print("=" * 80)

    # Reset to reference pose for final inspection
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    data.ctrl[:] = ref_qpos[0, 7:]
    mujoco.mj_forward(model, data)
    cur_state_ref = extract_sim_state(model, data, num_bodies, body_com_offsets)

    print(f"\n  Current state (robot at ref[0]):")
    print(f"    body_pos[0] (root): {cur_state_ref['body_pos'][0]}")
    print(f"    body_rot[0] (root, xyzw): {cur_state_ref['body_rot'][0]}")
    print(f"    body_vel[0]: {cur_state_ref['body_vel'][0]}")
    print(f"    body_ang_vel[0]: {cur_state_ref['body_ang_vel'][0]}")

    print(f"\n  Future ref (t=0.02s):")
    future_ref = get_reference_at_time(ref_data, 0.02, dt_ref, T_ref)
    print(f"    body_pos[0] (root): {future_ref['body_pos'][0]}")
    print(f"    body_rot[0] (root, xyzw): {future_ref['body_rot'][0]}")
    print(f"    body_vel[0]: {future_ref['body_vel'][0]}")
    print(f"    body_ang_vel[0]: {future_ref['body_ang_vel'][0]}")

    # Difference between current and future
    pos_diff = future_ref["body_pos"] - cur_state_ref["body_pos"]
    print(f"\n  Difference (future - current):")
    print(f"    body_pos[0] diff: {pos_diff[0]}")
    print(f"    body_pos RMSD: {np.sqrt((pos_diff**2).sum(-1)).mean():.5f} m")
    print(f"    (Should be tiny: only 0.02s of motion)")

    # ====================================================================
    # TEST 7: Check ONNX with IDENTITY rotations for ALL bodies
    # This isolates whether rotation encoding is causing issues
    # ====================================================================
    print("\n" + "=" * 80)
    print("TEST 7: FEED IDENTITY ROTATIONS (both current and future)")
    print("=" * 80)
    print("  If the ONNX internal observation module correctly handles")
    print("  identity rotations, this should produce consistent outputs.")

    identity_rot = np.zeros((num_bodies, 4), dtype=np.float32)
    identity_rot[:, 3] = 1.0  # xyzw identity = [0, 0, 0, 1]

    cur_state_id = {
        "body_pos": cur_state_ref["body_pos"].copy(),
        "body_rot": identity_rot.copy(),
        "body_vel": np.zeros((num_bodies, 3), dtype=np.float32),
        "body_ang_vel": np.zeros((num_bodies, 3), dtype=np.float32),
    }

    future_pos_id = cur_state_id["body_pos"][None]
    future_rot_id = identity_rot[None]
    future_vel_id = np.zeros((1, num_bodies, 3), dtype=np.float32)
    future_angvel_id = np.zeros((1, num_bodies, 3), dtype=np.float32)

    out7 = run_onnx_inference(session, onnx_name_to_key, actual_in_names, actual_out_names,
                              cur_state_id, future_pos_id, future_rot_id,
                              future_vel_id, future_angvel_id, prev_actions)

    jpt7 = out7["joint_pos_targets"].squeeze()
    act7 = out7["actions"].squeeze()

    print(f"\n  Results (identity rotation everywhere):")
    print(f"    joint_pos_targets[:6] = {jpt7[:6]}")
    print(f"    actions[:6]           = {act7[:6]}")
    print(f"    |jpt|                 = {np.linalg.norm(jpt7):.4f}")
    print(f"    |actions|             = {np.linalg.norm(act7):.4f}")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)

    print(f"\n  Test 1 (zero error, real pose): |jpt - cur_dof| = {np.linalg.norm(jpt1 - cur_dof):.4f}")
    print(f"  Test 2 (with heading offset):   |jpt - cur_dof| = {np.linalg.norm(jpt2 - cur_dof):.4f}")
    print(f"  Test 3 (no heading offset):     |jpt - cur_dof| = {np.linalg.norm(jpt3 - cur_dof):.4f}")
    print(f"  Test 5 (T-pose, zero error):    |jpt - 0|       = {np.linalg.norm(jpt5):.4f}")
    print(f"  Test 7 (identity rots):         |jpt|           = {np.linalg.norm(jpt7):.4f}")

    print(f"\n  KEY QUESTION: Does jpt ≈ cur_dof when future_ref = current_state?")
    rel_err_1 = np.linalg.norm(jpt1 - cur_dof) / (np.linalg.norm(cur_dof) + 1e-8)
    print(f"  Relative error: {rel_err_1:.4f}")
    if rel_err_1 < 0.1:
        print(f"  ✓ YES → Observation construction is correct")
        print(f"    Issue is in the future reference (heading offset or interpolation)")
    else:
        print(f"  ✗ NO → Observation construction has a BUG")
        print(f"    The policy sees 'zero error' but still wants to change pose")
        print(f"    Likely: ONNX internal normalization sees different state than expected")

    print(f"\n  Does removing heading_offset help?")
    err_with = np.linalg.norm(jpt2 - cur_dof)
    err_without = np.linalg.norm(jpt3 - cur_dof)
    print(f"    With heading_offset:    |jpt - cur_dof| = {err_with:.4f}")
    print(f"    Without heading_offset: |jpt - cur_dof| = {err_without:.4f}")
    if err_without < err_with * 0.8:
        print(f"  ✓ YES → heading_offset is DOUBLE-APPLIED (remove it!)")
    elif err_with < err_without * 0.8:
        print(f"  ✗ NO → heading_offset is NEEDED")
    else:
        print(f"  → No significant difference (heading_offset ≈ identity at step 0)")


if __name__ == "__main__":
    main()
