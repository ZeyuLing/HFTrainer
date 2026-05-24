#!/usr/bin/env python3
"""Lockstep step-by-step comparison between test_physics_configs and run_smpl_rl_tracker.

Runs BOTH simulation loops on two INDEPENDENT but identically-configured MuJoCo models,
comparing qpos, ONNX inputs, ONNX outputs, and model parameters at EVERY step to
pinpoint exactly where (and why) trajectories first diverge.

Key: We use the SAME ONNX session for both, so any divergence MUST come from either:
  1. Different simulation state (qpos/qvel after physics stepping)
  2. Different observation construction (extract_sim_state or reference lookup)
  3. Different model parameters (actuator gains, geom properties)
  4. Different control application (what gets written to data.ctrl)

Usage:
    cd /apdcephfs/.../hf_trainer
    python3 scripts/embodied/lockstep_compare.py
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

with open(YAML_PATH) as f:
    YAML_META = yaml.safe_load(f)

CONTROL_DT = 0.02
DECIMATION = 20
MAX_STEPS = 100


# ===========================================================================
#  Shared utilities (copied verbatim to avoid any import-side-effects)
# ===========================================================================

def mujoco_wxyz_to_xyzw(quats_wxyz):
    return quats_wxyz[..., [1, 2, 3, 0]]


def _quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# ===========================================================================
#  Model loading — IDENTICAL for both (uses test_physics_configs's approach)
# ===========================================================================

def create_model():
    """Create MuJoCo model with Config D settings (the working config).

    This uses the SAME code as test_physics_configs.load_model_with_config("D_euler_with_margin")
    to ensure both model instances are TRULY identical.
    """
    from test_physics_configs import load_model_with_config
    model, data, desc = load_model_with_config("D_euler_with_margin")
    return model, data


# ===========================================================================
#  Reference precomputation — SINGLE shared implementation
# ===========================================================================

def precompute_maxcoords(model, data, ref_qpos, dt_ref):
    """Precompute body max-coords from reference qpos trajectory.

    Uses float64 throughout (the correct approach verified in test_init_diff.py).
    This is called ONCE and the result shared by both simulation paths.
    """
    T = ref_qpos.shape[0]
    num_bodies = 24

    body_pos = np.zeros((T, num_bodies, 3))  # float64
    body_rot = np.zeros((T, num_bodies, 4))  # float64, xyzw

    for t in range(T):
        data.qpos[:] = ref_qpos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos[1:num_bodies + 1].copy()
        body_rot_wxyz = data.xquat[1:num_bodies + 1].copy()
        body_rot[t] = mujoco_wxyz_to_xyzw(body_rot_wxyz)

    # Backward diff velocity
    body_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)

    for f in range(1, T):
        body_vel[f] = (body_pos[f] - body_pos[f - 1]) / dt_ref
        for j in range(num_bodies):
            q0 = body_rot[f - 1, j]  # xyzw
            q1 = body_rot[f, j]
            q0_w = np.array([q0[3], q0[0], q0[1], q0[2]])
            q1_w = np.array([q1[3], q1[0], q1[1], q1[2]])
            q0_inv = np.array([q0_w[0], -q0_w[1], -q0_w[2], -q0_w[3]])
            dq = _quat_mul_wxyz(q1_w, q0_inv)
            if dq[0] < 0:
                dq = -dq
            body_ang_vel[f, j] = 2.0 * dq[1:4] / dt_ref

    if T > 1:
        body_vel[0] = body_vel[1]
        body_ang_vel[0] = body_ang_vel[1]

    return body_pos, body_rot, body_vel, body_ang_vel


# ===========================================================================
#  extract_sim_state — TWO implementations side by side
# ===========================================================================

def extract_sim_state_test_physics(model, data):
    """Exact copy of test_physics_configs.py extract_sim_state().

    Uses named body lookup via MUJOCO_BODY_NAMES.
    Returns 4 separate arrays (float64).
    """
    MUJOCO_BODY_NAMES = [
        "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
        "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
        "Torso", "Spine", "Chest", "Neck", "Head",
        "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
        "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
    ]
    num_bodies = len(MUJOCO_BODY_NAMES)
    body_ids = []
    for name in MUJOCO_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        body_ids.append(bid)

    sim_pos = np.zeros((num_bodies, 3))
    sim_rot = np.zeros((num_bodies, 4))
    sim_vel = np.zeros((num_bodies, 3))
    sim_ang_vel = np.zeros((num_bodies, 3))

    for j, bid in enumerate(body_ids):
        sim_pos[j] = data.xpos[bid].copy()
        quat_wxyz = data.xquat[bid].copy()
        sim_rot[j] = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

        cvel = data.cvel[bid]
        ang_vel = cvel[:3].copy()
        lin_vel = cvel[3:].copy()

        com_local = model.body_ipos[bid]
        xmat = data.xmat[bid].reshape(3, 3)
        com_world = xmat @ com_local

        lin_vel_com = lin_vel + np.cross(ang_vel, com_world)

        sim_vel[j] = lin_vel_com
        sim_ang_vel[j] = ang_vel

    return sim_pos, sim_rot, sim_vel, sim_ang_vel


def extract_sim_state_run_smpl(model, data, num_bodies=24):
    """Exact copy of run_smpl_rl_tracker.py extract_sim_state().

    Uses sequential j+1 indexing.
    Returns dict with float64 arrays.
    """
    body_pos = np.zeros((num_bodies, 3))
    body_rot = np.zeros((num_bodies, 4))
    body_vel = np.zeros((num_bodies, 3))
    body_ang_vel = np.zeros((num_bodies, 3))

    for j in range(num_bodies):
        bid = j + 1

        body_pos[j] = data.xpos[bid].copy()
        quat_wxyz = data.xquat[bid].copy()
        body_rot[j] = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

        cvel = data.cvel[bid]
        ang_vel = cvel[:3].copy()
        lin_vel = cvel[3:].copy()

        com_local = model.body_ipos[bid]
        xmat = data.xmat[bid].reshape(3, 3)
        com_world = xmat @ com_local

        lin_vel_com = lin_vel + np.cross(ang_vel, com_world)

        body_vel[j] = lin_vel_com
        body_ang_vel[j] = ang_vel

    return {
        "body_pos": body_pos,
        "body_rot": body_rot,
        "body_vel": body_vel,
        "body_ang_vel": body_ang_vel,
    }


# ===========================================================================
#  Build ONNX inputs — TWO implementations side by side
# ===========================================================================

def build_onnx_inputs_test_physics(sim_pos, sim_rot, sim_vel, sim_ang_vel,
                                    future_pos, future_rot, future_vel, future_ang_vel,
                                    prev_actions):
    """Exact copy of test_physics_configs.py ONNX input construction.

    Uses hardcoded tensor names.
    future_* have shape (1, 24, dim) — sliced from the reference array.
    """
    inputs = {
        "current_rigid_body_ang_vel": sim_ang_vel[np.newaxis].astype(np.float32),
        "current_rigid_body_pos": sim_pos[np.newaxis].astype(np.float32),
        "current_rigid_body_rot": sim_rot[np.newaxis].astype(np.float32),
        "current_rigid_body_vel": sim_vel[np.newaxis].astype(np.float32),
        "ground_heights": np.zeros((1,), dtype=np.float32),
        "historical_actions": prev_actions[np.newaxis, np.newaxis].astype(np.float32),
        "mimic_future_ang_vel": future_ang_vel[np.newaxis].astype(np.float32),
        "mimic_future_pos": future_pos[np.newaxis].astype(np.float32),
        "mimic_future_rot": future_rot[np.newaxis].astype(np.float32),
        "mimic_future_vel": future_vel[np.newaxis].astype(np.float32),
    }
    return inputs


def build_onnx_inputs_run_smpl(cur_state, future_body_pos, future_body_rot,
                                future_body_vel, future_body_ang_vel,
                                prev_actions, onnx_name_to_key, actual_in_names):
    """Exact copy of run_smpl_rl_tracker.py ONNX input construction.

    Uses YAML onnx_name_to_key mapping.
    future_* have shape (1, 24, dim) — np.stack'd from future_states list.
    """
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

    return onnx_inputs


# ===========================================================================
#  Simulation loop A: test_physics_configs style
# ===========================================================================

def step_test_physics(model, data, ref_body_pos, ref_body_rot, ref_body_vel,
                      ref_body_ang_vel, dt_ref, sim_time, prev_actions, session,
                      out_names):
    """Single step of test_physics_configs loop. Returns ONNX inputs, outputs, new prev_actions."""
    num_ref_frames = ref_body_pos.shape[0]

    # Extract sim state
    sim_pos, sim_rot, sim_vel, sim_ang_vel = extract_sim_state_test_physics(model, data)

    # Future reference
    ref_time = sim_time + CONTROL_DT
    ref_frame_idx = min(int(ref_time / dt_ref), num_ref_frames - 1)

    future_pos = ref_body_pos[ref_frame_idx:ref_frame_idx+1]
    future_rot = ref_body_rot[ref_frame_idx:ref_frame_idx+1]
    future_vel = ref_body_vel[ref_frame_idx:ref_frame_idx+1]
    future_ang_vel = ref_body_ang_vel[ref_frame_idx:ref_frame_idx+1]

    # Build ONNX inputs
    inputs = build_onnx_inputs_test_physics(
        sim_pos, sim_rot, sim_vel, sim_ang_vel,
        future_pos, future_rot, future_vel, future_ang_vel,
        prev_actions)

    # Run ONNX
    outputs = session.run(out_names, inputs)
    out_dict = {name: val for name, val in zip(out_names, outputs)}

    joint_pos_targets = out_dict["joint_pos_targets"].squeeze()
    new_prev_actions = out_dict["actions"].squeeze().copy()

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

    # Set control and step physics
    data.ctrl[:] = joint_pos_targets
    for _ in range(DECIMATION):
        mujoco.mj_step(model, data)

    return inputs, out_dict, new_prev_actions


# ===========================================================================
#  Simulation loop B: run_smpl_rl_tracker style
# ===========================================================================

def step_run_smpl(model, data, ref_data_dict, dt_ref, sim_time, prev_actions,
                  session, out_names, onnx_name_to_key, actual_in_names,
                  future_step_indices, future_dt_seconds):
    """Single step of run_smpl_rl_tracker loop. Returns ONNX inputs, outputs, new prev_actions."""
    T_ref = ref_data_dict["body_pos"].shape[0]
    num_bodies = 24

    # Extract sim state (run_smpl style)
    cur_state = extract_sim_state_run_smpl(model, data, num_bodies)

    # Future reference (run_smpl style — using future_step_indices)
    future_states = []
    for fi, fdt in zip(future_step_indices, future_dt_seconds):
        future_time = sim_time + fi * fdt
        ref_frame_idx = min(int(future_time / dt_ref), T_ref - 1)
        future_ref = {k: v[ref_frame_idx].copy() for k, v in ref_data_dict.items()}
        future_states.append(future_ref)

    future_body_pos = np.stack([fs["body_pos"] for fs in future_states], axis=0)
    future_body_rot = np.stack([fs["body_rot"] for fs in future_states], axis=0)
    future_body_vel = np.stack([fs["body_vel"] for fs in future_states], axis=0)
    future_body_ang_vel = np.stack([fs["body_ang_vel"] for fs in future_states], axis=0)

    # Build ONNX inputs (run_smpl style)
    inputs = build_onnx_inputs_run_smpl(
        cur_state, future_body_pos, future_body_rot,
        future_body_vel, future_body_ang_vel,
        prev_actions, onnx_name_to_key, actual_in_names)

    # Run ONNX
    outputs = session.run(out_names, inputs)
    out_dict = {name: val for name, val in zip(out_names, outputs)}

    joint_pos_targets = out_dict["joint_pos_targets"].squeeze().copy()
    new_prev_actions = out_dict["actions"].squeeze().copy()

    # Apply dynamic PD gains (same as run_smpl)
    if "stiffness_targets" in out_dict and "damping_targets" in out_dict:
        stiff_out = out_dict["stiffness_targets"].squeeze()
        damp_out = out_dict["damping_targets"].squeeze()
        for i in range(model.nu):
            kp = float(stiff_out[i])
            kd = float(damp_out[i])
            model.actuator_gainprm[i, 0] = kp
            model.actuator_biasprm[i, 1] = -kp
            model.actuator_biasprm[i, 2] = -kd

    # Set control and step physics (with NaN guard like run_smpl)
    data.ctrl[:] = joint_pos_targets
    for sub_step in range(DECIMATION):
        mujoco.mj_step(model, data)
        if np.any(np.isnan(data.qpos[:7])):
            print(f"    [B] NaN in substep {sub_step}!")
            break

    return inputs, out_dict, new_prev_actions


# ===========================================================================
#  Main lockstep comparison
# ===========================================================================

def main():
    from run_smpl_rl_tracker import decode_motion_135, yup_to_zup, smpl_to_qpos

    print("=" * 80)
    print("  LOCKSTEP COMPARISON: test_physics_configs vs run_smpl_rl_tracker")
    print("=" * 80)

    # Load motion
    smpl_pose, transl, fps = decode_motion_135(NPZ_PATH)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    # Load ONNX session (shared)
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]
    actual_in_names = [i.name for i in session.get_inputs()]

    dt_ref = 1.0 / fps
    onnx_name_to_key = YAML_META["_runtime"]["onnx_name_to_in_key"]
    future_step_indices = YAML_META["motion"]["future_step_indices"]
    future_dt_seconds = YAML_META["motion"]["future_dt_seconds"]

    print(f"  Motion: fps={fps}, dt_ref={dt_ref:.6f}, T={smpl_pose_zup.shape[0]}")
    print(f"  future_step_indices={future_step_indices}, future_dt_seconds={future_dt_seconds}")
    print()

    # ------------------------------------------------------------------
    # Create TWO independent model instances with IDENTICAL config
    # ------------------------------------------------------------------
    print("  Creating two independent MuJoCo model instances (Config D)...")
    model_a, data_a = create_model()
    model_b, data_b = create_model()

    # Verify models are identical
    assert np.allclose(model_a.body_pos, model_b.body_pos)
    assert np.allclose(model_a.actuator_gainprm, model_b.actuator_gainprm)
    assert np.allclose(model_a.actuator_biasprm, model_b.actuator_biasprm)
    print("  Models verified IDENTICAL.")

    body_pos_1 = model_a.body_pos[1].copy()

    # Convert to qpos
    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)

    # ------------------------------------------------------------------
    # Ground height fix (same bilateral foot grounding for both)
    # ------------------------------------------------------------------
    data_a.qpos[:] = ref_qpos[0]
    data_a.qvel[:] = 0.0
    mujoco.mj_forward(model_a, data_a)

    left_foot_ids = set()
    right_foot_ids = set()
    for bid in range(1, model_a.nbody):
        bname = mujoco.mj_id2name(model_a, mujoco.mjtObj.mjOBJ_BODY, bid)
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

    left_min = _lowest_geom_z(left_foot_ids, model_a, data_a)
    right_min = _lowest_geom_z(right_foot_ids, model_a, data_a)
    grounding_ref_z = min(left_min, right_min)
    height_shift = 0.0 - grounding_ref_z
    ref_qpos[:, 2] += height_shift
    print(f"  Height shift: {height_shift:+.6f}m")
    print(f"  Root height after shift: {ref_qpos[0, 2]:.6f}m")

    # ------------------------------------------------------------------
    # Precompute reference maxcoords (ONCE, shared by both)
    # ------------------------------------------------------------------
    print("  Precomputing reference max-coords (float64)...")
    # Use model_a for FK (both models are identical for kinematics)
    body_pos, body_rot, body_vel, body_ang_vel = precompute_maxcoords(
        model_a, data_a, ref_qpos, dt_ref)

    # Also create ref_data dict for run_smpl style access
    ref_data_dict = {
        "body_pos": body_pos,
        "body_rot": body_rot,
        "body_vel": body_vel,
        "body_ang_vel": body_ang_vel,
    }

    print(f"  Reference FK done. {body_pos.shape[0]} frames.")

    # ------------------------------------------------------------------
    # Initialize BOTH simulations identically
    # ------------------------------------------------------------------
    # Model A: test_physics_configs style
    data_a.qpos[:] = ref_qpos[0]
    data_a.qvel[:] = 0.0
    mujoco.mj_forward(model_a, data_a)

    # Model B: run_smpl_rl_tracker style (SAME as above — no ctrl pre-set)
    data_b.qpos[:] = ref_qpos[0]
    data_b.qvel[:] = 0.0
    mujoco.mj_forward(model_b, data_b)

    # Verify initial states are IDENTICAL
    assert np.allclose(data_a.qpos, data_b.qpos), "Initial qpos differ!"
    assert np.allclose(data_a.qvel, data_b.qvel), "Initial qvel differ!"
    assert np.allclose(data_a.xpos, data_b.xpos), "Initial xpos differ!"
    print("  Initial states verified IDENTICAL.")
    print()

    # ------------------------------------------------------------------
    # Lockstep simulation
    # ------------------------------------------------------------------
    prev_actions_a = np.zeros(69, dtype=np.float32)
    prev_actions_b = np.zeros(69, dtype=np.float32)

    sim_time_a = 0.0  # test_physics tracks sim_time with += CONTROL_DT
    # run_smpl computes sim_time = step_idx * control_dt

    first_diverge_step = None
    diverge_detail = ""

    print(f"  {'step':>4s}  {'qpos_diff':>12s}  {'onnx_in_diff':>14s}  "
          f"{'onnx_out_diff':>14s}  {'ctrl_diff':>12s}  {'gain_diff':>12s}  "
          f"{'root_h_A':>8s}  {'root_h_B':>8s}")
    print("  " + "-" * 100)

    for step in range(MAX_STEPS):
        # Check qpos agreement BEFORE this step's inference
        qpos_diff = np.abs(data_a.qpos - data_b.qpos).max()
        qvel_diff = np.abs(data_a.qvel - data_b.qvel).max()

        # ---- Side A: test_physics_configs ----
        sim_time_a_used = sim_time_a  # What test_physics uses for ref_time

        # Side A: fall check (test_physics checks AFTER physics, but for reporting
        # we check before to stay in sync with side B)
        root_h_a = data_a.qpos[2]
        root_h_b = data_b.qpos[2]

        if root_h_a < 0.3 or root_h_b < 0.3:
            print(f"  Step {step}: FALL! root_h_A={root_h_a:.4f}, root_h_B={root_h_b:.4f}")
            break
        if np.any(np.isnan(data_a.qpos)) or np.any(np.isnan(data_b.qpos)):
            print(f"  Step {step}: NaN detected!")
            break

        # ---- Run both sides ----
        # Side A: test_physics_configs style
        inputs_a, out_dict_a, new_prev_a = step_test_physics(
            model_a, data_a, body_pos, body_rot, body_vel, body_ang_vel,
            dt_ref, sim_time_a_used, prev_actions_a, session, out_names)

        # Side B: run_smpl_rl_tracker style
        sim_time_b = step * CONTROL_DT  # run_smpl computes this way
        inputs_b, out_dict_b, new_prev_b = step_run_smpl(
            model_b, data_b, ref_data_dict, dt_ref, sim_time_b, prev_actions_b,
            session, out_names, onnx_name_to_key, actual_in_names,
            future_step_indices, future_dt_seconds)

        # ---- Compare ONNX inputs ----
        max_input_diff = 0.0
        worst_input_key = ""
        for key in sorted(inputs_a.keys()):
            if key in inputs_b:
                d = np.abs(inputs_a[key] - inputs_b[key]).max()
                if d > max_input_diff:
                    max_input_diff = d
                    worst_input_key = key

        # ---- Compare ONNX outputs ----
        max_output_diff = 0.0
        worst_output_key = ""
        for name in out_names:
            d = np.abs(out_dict_a[name] - out_dict_b[name]).max()
            if d > max_output_diff:
                max_output_diff = d
                worst_output_key = name

        # ---- Compare ctrl values ----
        ctrl_diff = np.abs(data_a.ctrl - data_b.ctrl).max()

        # ---- Compare actuator gains ----
        gain_diff = max(
            np.abs(model_a.actuator_gainprm - model_b.actuator_gainprm).max(),
            np.abs(model_a.actuator_biasprm - model_b.actuator_biasprm).max(),
        )

        # Update sim_time for side A
        sim_time_a += CONTROL_DT
        prev_actions_a = new_prev_a
        prev_actions_b = new_prev_b

        # ---- Report ----
        if step < 5 or step % 5 == 0 or max_input_diff > 1e-6 or qpos_diff > 1e-6:
            print(f"  {step:4d}  {qpos_diff:12.2e}  {max_input_diff:14.2e}  "
                  f"{max_output_diff:14.2e}  {ctrl_diff:12.2e}  {gain_diff:12.2e}  "
                  f"{root_h_a:8.4f}  {root_h_b:8.4f}")

        # ---- Detect first meaningful divergence ----
        if first_diverge_step is None and (qpos_diff > 1e-8 or max_input_diff > 1e-8):
            first_diverge_step = step
            diverge_detail = (
                f"qpos_diff={qpos_diff:.2e}, "
                f"onnx_input_diff={max_input_diff:.2e} (key={worst_input_key}), "
                f"onnx_output_diff={max_output_diff:.2e} (key={worst_output_key}), "
                f"ctrl_diff={ctrl_diff:.2e}, gain_diff={gain_diff:.2e}"
            )
            # Print detailed breakdown at divergence point
            print(f"\n  *** FIRST DIVERGENCE at step {step} ***")
            print(f"  {diverge_detail}")
            print()

            # Show detailed per-input comparison
            print("  Per-input breakdown:")
            for key in sorted(inputs_a.keys()):
                if key in inputs_b:
                    d = np.abs(inputs_a[key] - inputs_b[key])
                    max_d = d.max()
                    if max_d > 0:
                        idx = np.unravel_index(np.argmax(d), d.shape)
                        print(f"    {key:40s}: max_diff={max_d:.2e} at {idx}, "
                              f"A={inputs_a[key][idx]:.10f}, B={inputs_b[key][idx]:.10f}")

            # Show qpos diff detail
            if qpos_diff > 0:
                qpos_a = data_a.qpos
                qpos_b = data_b.qpos
                # This is qpos AFTER physics (data_a.qpos was already stepped)
                # Actually no — we compared BEFORE stepping. Let me trace...
                # Actually qpos_diff was computed BEFORE step_test_physics/step_run_smpl
                # were called, which internally step physics. So the qpos_diff reported
                # at step N represents the state ENTERING step N (after step N-1's physics).
                print(f"\n  qpos diff detail (entering this step):")
                qdiff = np.abs(data_a.qpos - data_b.qpos)
                # Wait, data_a/data_b have ALREADY been stepped now.
                # The qpos_diff variable was computed before stepping.
                # To show WHICH DOFs diverged, re-check after stepping.
                pass

            print()

        # ---- Detect significant divergence → detailed drill-down ----
        if qpos_diff > 1e-4 and first_diverge_step is not None and step == first_diverge_step + 1:
            print(f"\n  === DRILL-DOWN at step {step} (1 step after first divergence) ===")
            qpos_now_diff = np.abs(data_a.qpos - data_b.qpos)
            worst_dof = np.argmax(qpos_now_diff)
            print(f"  Worst qpos DOF after physics: idx={worst_dof}, "
                  f"diff={qpos_now_diff[worst_dof]:.2e}")
            print(f"  data_a.qpos[{worst_dof}] = {data_a.qpos[worst_dof]:.10f}")
            print(f"  data_b.qpos[{worst_dof}] = {data_b.qpos[worst_dof]:.10f}")
            print(f"  data_a.qpos[:7] = {data_a.qpos[:7]}")
            print(f"  data_b.qpos[:7] = {data_b.qpos[:7]}")

            # Compare actuator gains
            gainprm_diff = np.abs(model_a.actuator_gainprm - model_b.actuator_gainprm)
            biasprm_diff = np.abs(model_a.actuator_biasprm - model_b.actuator_biasprm)
            if gainprm_diff.max() > 0 or biasprm_diff.max() > 0:
                worst_act = np.argmax(gainprm_diff.max(axis=1))
                print(f"  Worst actuator gain diff: act[{worst_act}]")
                print(f"    model_a.gainprm = {model_a.actuator_gainprm[worst_act][:3]}")
                print(f"    model_b.gainprm = {model_b.actuator_gainprm[worst_act][:3]}")
                print(f"    model_a.biasprm = {model_a.actuator_biasprm[worst_act][:3]}")
                print(f"    model_b.biasprm = {model_b.actuator_biasprm[worst_act][:3]}")

            # prev_actions diff
            pa_diff = np.abs(prev_actions_a - prev_actions_b).max()
            print(f"  prev_actions diff: {pa_diff:.2e}")
            print(f"  === END DRILL-DOWN ===\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    if first_diverge_step is None:
        print(f"  NO DIVERGENCE detected in {MAX_STEPS} steps!")
        print("  Both loops produce IDENTICAL results.")
        print()
        print("  This means the difference MUST be in:")
        print("  - Model initialization (ground height, initial ref_qpos)")
        print("  - Floor geom properties causing different contact behavior")
        print("  - Something in the ACTUAL run_smpl_rl_tracker.py that this lockstep")
        print("    test doesn't reproduce (e.g., the ref precompute uses different model)")
    else:
        print(f"  First divergence at step {first_diverge_step}")
        print(f"  Detail: {diverge_detail}")
        print()
        print(f"  Final state: root_h_A={data_a.qpos[2]:.4f}, root_h_B={data_b.qpos[2]:.4f}")

    # Final comparison of how far each got
    print()
    print(f"  Simulation ended at step {step}")
    print(f"  root_h_A = {data_a.qpos[2]:.4f}m")
    print(f"  root_h_B = {data_b.qpos[2]:.4f}m")
    print(f"  max qpos diff at end = {np.abs(data_a.qpos - data_b.qpos).max():.2e}")


if __name__ == "__main__":
    main()
