"""Diagnostic: Feed ONNX policy with current==reference (zero tracking error).

If the policy is working correctly and we feed it body states where the current
state exactly matches the target (future) state, the expected output is:
  - raw actions ≈ 0 (or very small)
  - joint_pos_targets ≈ current DOF positions (since targets = offset + scale * actions)

This tests the full observation pipeline inside the ONNX:
  1. Heading normalization (root rotation → heading quat → inverse)
  2. Root-relative body positions
  3. 6D tan-norm rotation encoding
  4. Running mean/std normalization
  5. Target pose computation (relative to current heading)
  6. Actor network forward pass

If outputs are NOT near-zero, the issue is in how we construct the inputs,
not in the policy itself (since the policy was trained to produce zero actions
when already at target).

Usage:
    python3 scripts/embodied/diag_onnx_zero_error.py
"""

import sys
import os
import numpy as np
import yaml
import mujoco

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Paths
# ============================================================================
BASE = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
ONNX_PATH = os.path.join(
    BASE, "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/"
    "compiled_models/unified_pipeline.onnx")
YAML_PATH = os.path.join(
    BASE, "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/"
    "compiled_models/unified_pipeline.yaml")
MJCF_PATH = os.path.join(
    BASE, "ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml")


def mujoco_wxyz_to_xyzw(quats_wxyz):
    """Convert MuJoCo wxyz quaternions to xyzw (policy convention)."""
    if quats_wxyz.ndim == 1:
        return np.array([quats_wxyz[1], quats_wxyz[2], quats_wxyz[3], quats_wxyz[0]])
    return np.concatenate([quats_wxyz[:, 1:4], quats_wxyz[:, 0:1]], axis=1)


def load_mujoco_model(mjcf_path, stiffness, damping, physics_dt):
    """Load MuJoCo model and configure actuators."""
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    model.opt.timestep = physics_dt

    # Configure actuator PD gains
    for i in range(model.nu):
        kp = stiffness[i] if i < len(stiffness) else 200.0
        kd = damping[i] if i < len(damping) else 10.0
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kd

    return model, data


def extract_body_com_offsets(model, num_bodies):
    """Extract per-body COM offsets (same as ProtoMotions)."""
    offsets = np.zeros((num_bodies, 3), dtype=np.float32)
    for b_idx in range(num_bodies):
        body_id = b_idx + 1  # Skip world body
        geom_indices = [g for g in range(model.ngeom)
                        if model.geom_bodyid[g] == body_id]
        if geom_indices:
            geom_positions = model.geom_pos[geom_indices]
            offsets[b_idx] = geom_positions.mean(axis=0)
    return offsets


def apply_com_velocity_correction(body_vel_frame, body_ang_vel, body_rot_wxyz, body_com_offsets):
    """Convert frame-origin velocity to COM velocity."""
    num_bodies = body_vel_frame.shape[0]
    body_vel_com = body_vel_frame.copy()

    for b in range(num_bodies):
        offset_local = body_com_offsets[b]
        if np.linalg.norm(offset_local) < 1e-8:
            continue
        # Rotate offset to world frame
        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat(mujoco_wxyz_to_xyzw(body_rot_wxyz[b:b+1])).as_matrix()[0]
        offset_world = rot @ offset_local
        # v_COM = v_frame + omega × r
        omega = body_ang_vel[b]
        body_vel_com[b] = body_vel_frame[b] + np.cross(omega, offset_world)

    return body_vel_com


def main():
    print("=" * 70)
    print("DIAGNOSTIC: ONNX Zero-Error Test")
    print("If current == reference, actions should be ≈ 0")
    print("=" * 70)

    # Load YAML metadata
    with open(YAML_PATH) as f:
        yaml_meta = yaml.safe_load(f)

    robot_meta = yaml_meta["robot"]
    timing = yaml_meta["timing"]
    control = yaml_meta["control"]
    runtime = yaml_meta["_runtime"]

    num_bodies = robot_meta["num_bodies"]  # 24
    num_dofs = robot_meta["num_dofs"]      # 69
    control_dt = timing["control_dt"]      # 0.02
    physics_dt = timing["physics_dt"]      # 0.001
    stiffness = control["stiffness"]
    damping = control["damping"]
    onnx_name_to_key = runtime["onnx_name_to_in_key"]

    print(f"\nConfig: {num_bodies} bodies, {num_dofs} DOFs, "
          f"control_dt={control_dt}, physics_dt={physics_dt}")
    print(f"ONNX name → key mapping:")
    for k, v in onnx_name_to_key.items():
        print(f"  {k} → {v}")

    # Load MuJoCo
    model, data = load_mujoco_model(MJCF_PATH, stiffness, damping, physics_dt)
    body_com_offsets = extract_body_com_offsets(model, num_bodies)

    print(f"\nMuJoCo model: {model.nbody} bodies, nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # Load ONNX
    import onnxruntime as ort
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    actual_in_names = [inp.name for inp in session.get_inputs()]
    actual_out_names = [out.name for out in session.get_outputs()]

    print(f"\nONNX inputs: {actual_in_names}")
    print(f"ONNX outputs: {actual_out_names}")

    # ====================================================================
    # TEST 1: T-pose (default qpos = zeros)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Default T-pose (qpos=0 except root height)")
    print("=" * 70)

    # Set default pose (T-pose with reasonable height)
    data.qpos[:] = 0.0
    data.qpos[2] = 0.91  # Approximate standing height
    data.qpos[3] = 1.0   # wxyz quaternion w=1 (identity)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    run_zero_error_test(model, data, session, actual_in_names, actual_out_names,
                        onnx_name_to_key, num_bodies, num_dofs, body_com_offsets,
                        test_name="T-pose")

    # ====================================================================
    # TEST 2: Load from a real reference pose
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Real reference pose from sample NPZ")
    print("=" * 70)

    # Try to find a sample NPZ
    sample_npz_dirs = [
        os.path.join(BASE, "output/embodied_t2m_v4/data/npz"),
        os.path.join(BASE, "output/physflow_v2/test_rl_oracle"),
    ]

    sample_npz = None
    for d in sample_npz_dirs:
        if os.path.isdir(d):
            npz_files = [f for f in os.listdir(d) if f.endswith('.npz')]
            if npz_files:
                sample_npz = os.path.join(d, npz_files[0])
                break

    # Test 2A: Multi-step simulation to check stability
    # Start in T-pose (which passes), then do 10 control steps with "hold current pose"
    # If the policy is stable, the robot should stay still
    print(f"\n  Running 10-step hold test from T-pose...")
    data.qpos[:] = 0.0
    data.qpos[2] = 0.91
    data.qpos[3] = 1.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0  # Start with zero ctrl (DOFs at 0)
    mujoco.mj_forward(model, data)

    prev_actions_test = np.zeros(num_dofs, dtype=np.float32)
    decimation = timing["decimation"]

    for step in range(10):
        # Extract current state
        bp = data.xpos[1:num_bodies + 1].copy().astype(np.float32)
        br_wxyz = data.xquat[1:num_bodies + 1].copy()
        br = mujoco_wxyz_to_xyzw(br_wxyz).astype(np.float32)
        cv = data.cvel[1:num_bodies + 1].copy()
        bav = cv[:, 0:3].astype(np.float32)
        bvf = cv[:, 3:6].astype(np.float32)
        bv = apply_com_velocity_correction(bvf, bav, br_wxyz.astype(np.float32), body_com_offsets)

        # Feed current as target (hold pose)
        onnx_in = build_onnx_inputs(
            bp, br, bv, bav,
            bp, br, bv, bav,  # future == current
            prev_actions_test,
            onnx_name_to_key, actual_in_names
        )
        ort_out = session.run(actual_out_names, onnx_in)
        actions = ort_out[0].squeeze()
        jpt = ort_out[1].squeeze()
        stiff = ort_out[2].squeeze()
        damp = ort_out[3].squeeze()

        prev_actions_test = actions.copy()

        # Apply control
        data.ctrl[:] = jpt
        # Apply dynamic PD gains
        for i in range(model.nu):
            model.actuator_gainprm[i, 0] = stiff[i]
            model.actuator_biasprm[i, 1] = -stiff[i]
            model.actuator_biasprm[i, 2] = -damp[i]

        for _ in range(decimation):
            mujoco.mj_step(model, data)

        root_h = data.qpos[2]
        act_abs_max = np.abs(actions).max()
        dof_err = np.abs(jpt - data.qpos[7:76]).mean()
        print(f"    Step {step}: root_h={root_h:.4f}, actions_max={act_abs_max:.4f}, "
              f"dof_err_from_jpt={dof_err:.4f}")

    print(f"\n  After 10 steps: root_h={data.qpos[2]:.4f} (started at 0.91)")
    if abs(data.qpos[2] - 0.91) < 0.05:
        print(f"  ✅ Robot stays approximately in place (Δh={data.qpos[2]-0.91:.4f})")
    else:
        print(f"  ❌ Robot moved significantly (Δh={data.qpos[2]-0.91:.4f})")

    # Now the real test: set a reference pose that's DIFFERENT from current
    # and check that the policy tries to move towards it
    print(f"\n  Running 10-step tracking test (target = rotated arm)...")
    data.qpos[:] = 0.0
    data.qpos[2] = 0.91
    data.qpos[3] = 1.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    # Create a target with left shoulder rotated (DOF index for L_Shoulder)
    target_qpos = data.qpos.copy()
    target_qpos[7 + 3*3] = 1.0  # Rotate one joint by 1 radian

    # Compute target body state via FK
    data.qpos[:] = target_qpos
    mujoco.mj_forward(model, data)
    target_bp = data.xpos[1:num_bodies + 1].copy().astype(np.float32)
    target_br_wxyz = data.xquat[1:num_bodies + 1].copy()
    target_br = mujoco_wxyz_to_xyzw(target_br_wxyz).astype(np.float32)
    target_cv = data.cvel[1:num_bodies + 1].copy()
    target_bav = target_cv[:, 0:3].astype(np.float32)
    target_bvf = target_cv[:, 3:6].astype(np.float32)
    target_bv = apply_com_velocity_correction(
        target_bvf, target_bav, target_br_wxyz.astype(np.float32), body_com_offsets)

    # Reset to base pose
    data.qpos[:] = 0.0
    data.qpos[2] = 0.91
    data.qpos[3] = 1.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    prev_actions_test = np.zeros(num_dofs, dtype=np.float32)
    for step in range(10):
        bp = data.xpos[1:num_bodies + 1].copy().astype(np.float32)
        br_wxyz = data.xquat[1:num_bodies + 1].copy()
        br = mujoco_wxyz_to_xyzw(br_wxyz).astype(np.float32)
        cv = data.cvel[1:num_bodies + 1].copy()
        bav = cv[:, 0:3].astype(np.float32)
        bvf = cv[:, 3:6].astype(np.float32)
        bv = apply_com_velocity_correction(bvf, bav, br_wxyz.astype(np.float32), body_com_offsets)

        # Target is the rotated-arm pose
        onnx_in = build_onnx_inputs(
            bp, br, bv, bav,
            target_bp, target_br, target_bv, target_bav,
            prev_actions_test,
            onnx_name_to_key, actual_in_names
        )
        ort_out = session.run(actual_out_names, onnx_in)
        actions = ort_out[0].squeeze()
        jpt = ort_out[1].squeeze()
        stiff = ort_out[2].squeeze()
        damp = ort_out[3].squeeze()

        prev_actions_test = actions.copy()

        data.ctrl[:] = jpt
        for i in range(model.nu):
            model.actuator_gainprm[i, 0] = stiff[i]
            model.actuator_biasprm[i, 1] = -stiff[i]
            model.actuator_biasprm[i, 2] = -damp[i]

        for _ in range(decimation):
            mujoco.mj_step(model, data)

        # Track convergence toward target
        pos_err = np.sqrt(((bp - target_bp)**2).sum(-1)).mean()
        print(f"    Step {step}: MPJPE={pos_err:.4f}m, actions_max={np.abs(actions).max():.4f}, "
              f"root_h={data.qpos[2]:.4f}")

    # ====================================================================
    # TEST 3: Systematically vary one input to see sensitivity
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Input sensitivity analysis")
    print("=" * 70)

    # Use T-pose as baseline
    data.qpos[:] = 0.0
    data.qpos[2] = 0.91
    data.qpos[3] = 1.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Get baseline body state
    body_pos = data.xpos[1:num_bodies + 1].copy().astype(np.float32)
    body_rot_wxyz = data.xquat[1:num_bodies + 1].copy()
    body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz).astype(np.float32)

    cvel = data.cvel[1:num_bodies + 1].copy()
    body_ang_vel = cvel[:, 0:3].astype(np.float32)
    body_vel_frame = cvel[:, 3:6].astype(np.float32)
    body_vel = apply_com_velocity_correction(
        body_vel_frame, body_ang_vel, body_rot_wxyz.astype(np.float32), body_com_offsets)

    # Build baseline inputs (current == future)
    baseline_inputs = build_onnx_inputs(
        body_pos, body_rot, body_vel, body_ang_vel,
        body_pos, body_rot, body_vel, body_ang_vel,  # future == current
        np.zeros(num_dofs, dtype=np.float32),
        onnx_name_to_key, actual_in_names
    )

    # Run baseline
    ort_out = session.run(actual_out_names, baseline_inputs)
    baseline_actions = ort_out[0].squeeze()
    baseline_jpt = ort_out[1].squeeze()

    print(f"\nBaseline (current==future):")
    print(f"  actions: abs_mean={np.abs(baseline_actions).mean():.6f}, "
          f"max={np.abs(baseline_actions).max():.6f}")
    print(f"  jpt: abs_mean={np.abs(baseline_jpt).mean():.6f}, "
          f"max={np.abs(baseline_jpt).max():.6f}")

    # Now perturb each input and see sensitivity
    perturbation_tests = [
        ("future_pos +0.1m Z", "mimic.future_pos", lambda x: x + np.array([0, 0, 0.1])),
        ("future_pos +0.5m X", "mimic.future_pos", lambda x: x + np.array([0.5, 0, 0])),
        ("current_vel +1.0 Z", "current.rigid_body_vel", lambda x: x + np.array([0, 0, 1.0])),
    ]

    for name, key, perturb_fn in perturbation_tests:
        perturbed = {k: v.copy() for k, v in baseline_inputs.items()}
        for onnx_name, sem_key in onnx_name_to_key.items():
            if sem_key == key and onnx_name in perturbed:
                arr = perturbed[onnx_name]
                # Apply perturbation to all bodies
                orig_shape = arr.shape
                flat = arr.reshape(-1, 3) if arr.shape[-1] == 3 else arr
                if arr.shape[-1] == 3:
                    flat = arr.reshape(-1, 3)
                    for i in range(flat.shape[0]):
                        flat[i] = perturb_fn(flat[i])
                    perturbed[onnx_name] = flat.reshape(orig_shape)
                break

        ort_out2 = session.run(actual_out_names, perturbed)
        perturbed_actions = ort_out2[0].squeeze()
        delta = np.abs(perturbed_actions - baseline_actions).mean()
        print(f"  {name}: action_delta_mean={delta:.6f}, "
              f"actions_abs_mean={np.abs(perturbed_actions).mean():.6f}")


def build_onnx_inputs(body_pos, body_rot, body_vel, body_ang_vel,
                      future_pos, future_rot, future_vel, future_ang_vel,
                      prev_actions, onnx_name_to_key, actual_in_names):
    """Build ONNX input dict from body states."""
    key_to_array = {
        "current.rigid_body_pos": body_pos[None],         # (1, 24, 3)
        "current.rigid_body_rot": body_rot[None],         # (1, 24, 4)
        "current.rigid_body_vel": body_vel[None],         # (1, 24, 3)
        "current.rigid_body_ang_vel": body_ang_vel[None], # (1, 24, 3)
        "ground_heights": np.zeros(1, dtype=np.float32),  # (1,)
        "historical.actions": prev_actions[None, None],   # (1, 1, 69)
        "mimic.future_pos": future_pos[None, None],       # (1, 1, 24, 3)
        "mimic.future_rot": future_rot[None, None],       # (1, 1, 24, 4)
        "mimic.future_vel": future_vel[None, None],       # (1, 1, 24, 3)
        "mimic.future_ang_vel": future_ang_vel[None, None],  # (1, 1, 24, 3)
    }

    onnx_inputs = {}
    for onnx_name, sem_key in onnx_name_to_key.items():
        if sem_key in key_to_array:
            onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)

    # Verify all inputs present
    missing = [n for n in actual_in_names if n not in onnx_inputs]
    if missing:
        raise RuntimeError(f"Missing ONNX inputs: {missing}")

    return onnx_inputs


def run_zero_error_test(model, data, session, actual_in_names, actual_out_names,
                        onnx_name_to_key, num_bodies, num_dofs, body_com_offsets,
                        test_name=""):
    """Run ONNX with current==future and analyze outputs."""

    # Extract current state (same as extract_sim_state in run_smpl_rl_tracker.py)
    body_pos = data.xpos[1:num_bodies + 1].copy().astype(np.float32)
    body_rot_wxyz = data.xquat[1:num_bodies + 1].copy()
    body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz).astype(np.float32)

    cvel = data.cvel[1:num_bodies + 1].copy()
    body_ang_vel = cvel[:, 0:3].astype(np.float32)
    body_vel_frame = cvel[:, 3:6].astype(np.float32)
    body_vel = apply_com_velocity_correction(
        body_vel_frame, body_ang_vel, body_rot_wxyz.astype(np.float32), body_com_offsets)

    print(f"\n  [{test_name}] Current state:")
    print(f"    root_pos (body_pos[0]): {body_pos[0]}")
    print(f"    root_rot (xyzw): {body_rot[0]}")
    print(f"    body_vel[0]: {body_vel[0]}")
    print(f"    body_ang_vel[0]: {body_ang_vel[0]}")
    print(f"    dof_pos[:5] (qpos[7:12]): {data.qpos[7:12]}")

    # Use SAME state as future reference (zero tracking error)
    onnx_inputs = build_onnx_inputs(
        body_pos, body_rot, body_vel, body_ang_vel,
        body_pos, body_rot, body_vel, body_ang_vel,  # future == current
        np.zeros(num_dofs, dtype=np.float32),         # zero prev actions
        onnx_name_to_key, actual_in_names
    )

    # Print shapes for verification
    print(f"\n  [{test_name}] ONNX input shapes:")
    for name in actual_in_names:
        arr = onnx_inputs[name]
        print(f"    {name}: shape={arr.shape}, dtype={arr.dtype}, "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]")

    # Run inference
    ort_out = session.run(actual_out_names, onnx_inputs)
    out_dict = {name: val for name, val in zip(actual_out_names, ort_out)}

    # Analyze outputs
    print(f"\n  [{test_name}] ONNX outputs (ZERO ERROR EXPECTED):")
    for name in actual_out_names:
        arr = out_dict[name].squeeze()
        print(f"    {name}: shape={arr.shape}, "
              f"abs_mean={np.abs(arr).mean():.6f}, "
              f"abs_max={np.abs(arr).max():.6f}, "
              f"std={arr.std():.6f}")

    actions = out_dict["actions"].squeeze()
    jpt = out_dict["joint_pos_targets"].squeeze()
    current_dofs = data.qpos[7:7+num_dofs].copy()

    # Key check: jpt should be close to current DOFs
    jpt_error = jpt - current_dofs
    print(f"\n  [{test_name}] KEY METRIC: joint_pos_targets vs current DOFs:")
    print(f"    jpt_error abs_mean = {np.abs(jpt_error).mean():.6f} rad")
    print(f"    jpt_error abs_max  = {np.abs(jpt_error).max():.6f} rad")
    print(f"    jpt_error std      = {jpt_error.std():.6f}")

    if np.abs(actions).max() < 0.1:
        print(f"\n  ✅ [{test_name}] PASS: actions are near-zero (max={np.abs(actions).max():.6f})")
    elif np.abs(actions).max() < 0.5:
        print(f"\n  ⚠️  [{test_name}] MARGINAL: actions are moderate (max={np.abs(actions).max():.6f})")
    else:
        print(f"\n  ❌ [{test_name}] FAIL: actions are LARGE (max={np.abs(actions).max():.6f})")
        print(f"    This means the observation pipeline is not correctly encoding")
        print(f"    the 'already at target' state.")

        # Additional analysis: check which DOFs have large targets
        large_idx = np.where(np.abs(jpt_error) > 0.5)[0]
        if len(large_idx) > 0:
            print(f"\n    DOFs with |jpt - current| > 0.5 rad:")
            for idx in large_idx[:10]:
                # Get actuator name
                act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
                print(f"      [{idx}] {act_name or 'unnamed'}: "
                      f"current={current_dofs[idx]:.4f}, target={jpt[idx]:.4f}, "
                      f"error={jpt_error[idx]:.4f}")

    # Check stiffness/damping
    if "stiffness_targets" in out_dict:
        stiff = out_dict["stiffness_targets"].squeeze()
        damp = out_dict["damping_targets"].squeeze()
        print(f"\n  [{test_name}] PD gains:")
        print(f"    stiffness: mean={stiff.mean():.1f}, range=[{stiff.min():.1f}, {stiff.max():.1f}]")
        print(f"    damping: mean={damp.mean():.1f}, range=[{damp.min():.1f}, {damp.max():.1f}]")

    return out_dict


if __name__ == "__main__":
    main()
