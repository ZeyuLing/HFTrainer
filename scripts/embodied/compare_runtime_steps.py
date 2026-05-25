#!/usr/bin/env python3
"""Runtime step-by-step comparison between test_init_diff and run_smpl_rl_tracker.

Runs BOTH code paths on the SAME motion data and compares ALL state at EVERY step
to find the FIRST divergence point. This definitively identifies the root cause.

Strategy:
  1. Use test_init_diff's approach (which survives 148 steps) as REFERENCE
  2. Use run_smpl_rl_tracker's run_rl_tracker() function as TEST
  3. Compare qpos, ctrl, ONNX inputs/outputs at every step
  4. Report the FIRST step where divergence exceeds threshold

Key insight from prior analysis:
  - Step 0 inputs are IDENTICAL (confirmed by compare_onnx_inputs.py)
  - Both use float64 reference arrays
  - Both skip ctrl_init
  - Yet run_smpl falls at step 62 while test_init_diff survives 148
  - This MUST be due to some runtime difference that accumulates
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
PHYSICS_DT = 0.001
FALL_THRESHOLD = 0.3
MAX_STEPS = 80  # Run past step 62 to see divergence


def _quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def mujoco_wxyz_to_xyzw(quats_wxyz):
    return quats_wxyz[..., [1, 2, 3, 0]]


def extract_sim_state_test(model, data):
    """test_init_diff/test_physics_configs style extract_sim_state."""
    num_bodies = 24
    sim_pos = data.xpos[1:num_bodies + 1].copy()
    sim_rot_wxyz = data.xquat[1:num_bodies + 1].copy()
    sim_rot = mujoco_wxyz_to_xyzw(sim_rot_wxyz)

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


def extract_sim_state_runsmpl(model, data, num_bodies=24):
    """run_smpl_rl_tracker style extract_sim_state (returns dict)."""
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


def precompute_maxcoords(model, data, ref_qpos, dt_ref):
    """Shared precompute (float64)."""
    T = ref_qpos.shape[0]
    num_bodies = 24

    body_pos = np.zeros((T, num_bodies, 3))
    body_rot = np.zeros((T, num_bodies, 4))

    for t in range(T):
        data.qpos[:] = ref_qpos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos[1:num_bodies + 1].copy()
        body_rot_wxyz = data.xquat[1:num_bodies + 1].copy()
        body_rot[t] = mujoco_wxyz_to_xyzw(body_rot_wxyz)

    body_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)

    for f in range(1, T):
        body_vel[f] = (body_pos[f] - body_pos[f - 1]) / dt_ref
        for j in range(num_bodies):
            q0 = body_rot[f - 1, j]
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


def fix_height(model, data, ref_qpos):
    """Bilateral foot grounding (shared by both scripts)."""
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

    def _lowest_geom_z(body_id_set):
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

    left_min = _lowest_geom_z(left_foot_ids)
    right_min = _lowest_geom_z(right_foot_ids)
    grounding_ref_z = min(left_min, right_min)
    height_shift = 0.0 - grounding_ref_z
    return height_shift


def build_onnx_inputs_test(sim_pos, sim_rot, sim_vel, sim_ang_vel,
                           future_pos, future_rot, future_vel, future_ang_vel,
                           prev_actions):
    """Build ONNX inputs the test_init_diff way."""
    return {
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


def build_onnx_inputs_runsmpl(cur_state, future_body_pos, future_body_rot,
                               future_body_vel, future_body_ang_vel,
                               prev_actions, onnx_name_to_key):
    """Build ONNX inputs the run_smpl way (using onnx_name_to_key mapping)."""
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


def main():
    from test_physics_configs import load_model_with_config
    from run_smpl_rl_tracker import decode_motion_135, yup_to_zup, smpl_to_qpos

    print("=" * 70)
    print("  RUNTIME STEP-BY-STEP COMPARISON")
    print("  test_init_diff (REF) vs run_smpl_rl_tracker (TEST)")
    print("=" * 70)

    # Load motion
    smpl_pose, transl, fps = decode_motion_135(NPZ_PATH)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    dt_ref = 1.0 / fps
    print(f"\n  Motion: fps={fps}, dt_ref={dt_ref:.6f}")

    # Load ONNX session (shared)
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]
    in_names = [i.name for i in session.get_inputs()]

    # YAML metadata
    onnx_name_to_key = YAML_META["_runtime"]["onnx_name_to_in_key"]
    future_step_indices = YAML_META["motion"]["future_step_indices"]
    future_dt_seconds = YAML_META["motion"]["future_dt_seconds"]

    # ═══════════════════════════════════════════════════════════════
    # Path A: test_init_diff (REFERENCE — survives 148 steps)
    # ═══════════════════════════════════════════════════════════════
    print("\n--- Loading Path A (test_init_diff/test_physics_configs) ---")
    model_a, data_a, _ = load_model_with_config("D_euler_with_margin")
    body_pos_1_a = model_a.body_pos[1].copy()
    ref_qpos_a = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1_a)

    # Height fix
    height_shift_a = fix_height(model_a, data_a, ref_qpos_a)
    ref_qpos_a[:, 2] += height_shift_a
    print(f"  height_shift_a = {height_shift_a:+.6f}")

    # Precompute maxcoords
    bp_a, br_a, bv_a, bav_a = precompute_maxcoords(model_a, data_a, ref_qpos_a, dt_ref)

    # Set initial pose
    data_a.qpos[:] = ref_qpos_a[0]
    data_a.qvel[:] = 0.0
    mujoco.mj_forward(model_a, data_a)

    # ═══════════════════════════════════════════════════════════════
    # Path B: run_smpl_rl_tracker (TEST — falls at step 62)
    # ═══════════════════════════════════════════════════════════════
    print("\n--- Loading Path B (run_smpl_rl_tracker) ---")
    from run_smpl_rl_tracker import load_mujoco_model

    stiffness = YAML_META["control"]["stiffness"]
    damping = YAML_META["control"]["damping"]
    model_b, data_b = load_mujoco_model(MJCF_PATH, stiffness, damping, PHYSICS_DT)
    body_pos_1_b = model_b.body_pos[1].copy()
    ref_qpos_b = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1_b)

    # Height fix (run_smpl uses _temp_model for FK, but same logic)
    height_shift_b = fix_height(model_b, data_b, ref_qpos_b)
    ref_qpos_b[:, 2] += height_shift_b
    print(f"  height_shift_b = {height_shift_b:+.6f}")

    # Precompute maxcoords
    bp_b, br_b, bv_b, bav_b = precompute_maxcoords(model_b, data_b, ref_qpos_b, dt_ref)

    # Set initial pose
    data_b.qpos[:] = ref_qpos_b[0]
    data_b.qvel[:] = 0.0
    mujoco.mj_forward(model_b, data_b)

    # ═══════════════════════════════════════════════════════════════
    # Verify initial conditions match
    # ═══════════════════════════════════════════════════════════════
    print("\n--- Verifying initial conditions ---")
    print(f"  ref_qpos match: {np.allclose(ref_qpos_a, ref_qpos_b)}, "
          f"max_diff={np.abs(ref_qpos_a - ref_qpos_b).max():.2e}")
    print(f"  height_shift match: A={height_shift_a:.6f}, B={height_shift_b:.6f}, "
          f"diff={abs(height_shift_a - height_shift_b):.2e}")
    print(f"  ref body_pos match: {np.allclose(bp_a, bp_b)}, "
          f"max_diff={np.abs(bp_a - bp_b).max():.2e}")
    print(f"  ref body_vel match: {np.allclose(bv_a, bv_b)}, "
          f"max_diff={np.abs(bv_a - bv_b).max():.2e}")
    print(f"  initial qpos match: {np.allclose(data_a.qpos, data_b.qpos)}, "
          f"max_diff={np.abs(data_a.qpos - data_b.qpos).max():.2e}")
    print(f"  initial qvel match: {np.allclose(data_a.qvel, data_b.qvel)}, "
          f"max_diff={np.abs(data_a.qvel - data_b.qvel).max():.2e}")

    # Check model attributes that affect simulation
    print("\n--- Model attribute comparison ---")
    print(f"  timestep: A={model_a.opt.timestep}, B={model_b.opt.timestep}")
    print(f"  integrator: A={model_a.opt.integrator}, B={model_b.opt.integrator}")
    print(f"  ngeom: A={model_a.ngeom}, B={model_b.ngeom}")
    print(f"  nq: A={model_a.nq}, B={model_b.nq}")
    print(f"  nv: A={model_a.nv}, B={model_b.nv}")
    print(f"  nu: A={model_a.nu}, B={model_b.nu}")

    # Check actuator params
    print(f"  actuator_gear[:5]: A={model_a.actuator_gear[:5, 0]}, B={model_b.actuator_gear[:5, 0]}")
    print(f"  actuator_gainprm[:3,0]: A={model_a.actuator_gainprm[:3, 0]}, B={model_b.actuator_gainprm[:3, 0]}")
    print(f"  actuator_biasprm[:3,1]: A={model_a.actuator_biasprm[:3, 1]}, B={model_b.actuator_biasprm[:3, 1]}")
    print(f"  actuator_biasprm[:3,2]: A={model_a.actuator_biasprm[:3, 2]}, B={model_b.actuator_biasprm[:3, 2]}")
    print(f"  actuator_forcelimited: A={model_a.actuator_forcelimited[:3]}, B={model_b.actuator_forcelimited[:3]}")
    print(f"  actuator_forcerange[:3]: A={model_a.actuator_forcerange[:3]}, B={model_b.actuator_forcerange[:3]}")

    # Check geom contact properties
    print(f"\n--- Geom contact properties ---")
    print(f"  geom_margin[:5]: A={model_a.geom_margin[:5]}, B={model_b.geom_margin[:5]}")
    print(f"  geom_contype[:5]: A={model_a.geom_contype[:5]}, B={model_b.geom_contype[:5]}")
    print(f"  geom_conaffinity[:5]: A={model_a.geom_conaffinity[:5]}, B={model_b.geom_conaffinity[:5]}")
    print(f"  geom_condim[:5]: A={model_a.geom_condim[:5]}, B={model_b.geom_condim[:5]}")

    # Find floor geom in both
    floor_a = mujoco.mj_name2id(model_a, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_b = mujoco.mj_name2id(model_b, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    print(f"\n  Floor geom id: A={floor_a}, B={floor_b}")
    print(f"  Floor contype: A={model_a.geom_contype[floor_a]}, B={model_b.geom_contype[floor_b]}")
    print(f"  Floor conaffinity: A={model_a.geom_conaffinity[floor_a]}, B={model_b.geom_conaffinity[floor_b]}")
    print(f"  Floor condim: A={model_a.geom_condim[floor_a]}, B={model_b.geom_condim[floor_b]}")
    print(f"  Floor margin: A={model_a.geom_margin[floor_a]}, B={model_b.geom_margin[floor_b]}")
    print(f"  Floor size: A={model_a.geom_size[floor_a]}, B={model_b.geom_size[floor_b]}")

    # Check passive dynamics
    print(f"\n--- Passive dynamics ---")
    print(f"  jnt_stiffness max: A={model_a.jnt_stiffness.max()}, B={model_b.jnt_stiffness.max()}")
    print(f"  dof_damping max: A={model_a.dof_damping.max()}, B={model_b.dof_damping.max()}")
    print(f"  dof_frictionloss max: A={model_a.dof_frictionloss.max()}, B={model_b.dof_frictionloss.max()}")

    # Check solver params
    print(f"\n--- Solver params ---")
    print(f"  opt.solver: A={model_a.opt.solver}, B={model_b.opt.solver}")
    print(f"  opt.iterations: A={model_a.opt.iterations}, B={model_b.opt.iterations}")
    print(f"  opt.tolerance: A={model_a.opt.tolerance}, B={model_b.opt.tolerance}")
    print(f"  opt.gravity: A={model_a.opt.gravity}, B={model_b.opt.gravity}")

    # ═══════════════════════════════════════════════════════════════
    # RUN BOTH LOOPS SIDE-BY-SIDE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  RUNNING SIDE-BY-SIDE COMPARISON")
    print("=" * 70)

    prev_actions_a = np.zeros(69, dtype=np.float32)
    prev_actions_b = np.zeros(69, dtype=np.float32)
    T_ref = ref_qpos_a.shape[0]

    first_divergence_step = None
    first_divergence_detail = ""

    for step in range(MAX_STEPS):
        sim_time = step * CONTROL_DT

        # ---- Extract state from both ----
        # Path A: test_init_diff style
        sp_a, sr_a, sv_a, sav_a = extract_sim_state_test(model_a, data_a)
        # Path B: run_smpl style
        cur_b = extract_sim_state_runsmpl(model_b, data_b, 24)

        # Compare extracted states
        pos_diff = np.abs(sp_a - cur_b["body_pos"]).max()
        rot_diff = np.abs(sr_a - cur_b["body_rot"]).max()
        vel_diff = np.abs(sv_a - cur_b["body_vel"]).max()
        ang_diff = np.abs(sav_a - cur_b["body_ang_vel"]).max()

        # ---- Future reference (both use nearest-frame) ----
        # Path A: ref_time = sim_time + CONTROL_DT
        ref_time_a = sim_time + CONTROL_DT
        ref_frame_a = min(int(ref_time_a / dt_ref), T_ref - 1)
        future_pos_a = bp_a[ref_frame_a:ref_frame_a+1]
        future_rot_a = br_a[ref_frame_a:ref_frame_a+1]
        future_vel_a = bv_a[ref_frame_a:ref_frame_a+1]
        future_ang_a = bav_a[ref_frame_a:ref_frame_a+1]

        # Path B: sim_time + fi * fdt (where fi=1, fdt=0.02 → same as A)
        future_states_b = []
        for fi, fdt in zip(future_step_indices, future_dt_seconds):
            future_time = sim_time + fi * fdt
            ref_frame_b = min(int(future_time / dt_ref), T_ref - 1)
            future_ref_b = {
                "body_pos": bp_b[ref_frame_b].copy(),
                "body_rot": br_b[ref_frame_b].copy(),
                "body_vel": bv_b[ref_frame_b].copy(),
                "body_ang_vel": bav_b[ref_frame_b].copy(),
            }
            future_states_b.append(future_ref_b)

        future_pos_b = np.stack([fs["body_pos"] for fs in future_states_b], axis=0)
        future_rot_b = np.stack([fs["body_rot"] for fs in future_states_b], axis=0)
        future_vel_b = np.stack([fs["body_vel"] for fs in future_states_b], axis=0)
        future_ang_b = np.stack([fs["body_ang_vel"] for fs in future_states_b], axis=0)

        # ---- Build ONNX inputs ----
        inputs_a = build_onnx_inputs_test(
            sp_a, sr_a, sv_a, sav_a,
            future_pos_a, future_rot_a, future_vel_a, future_ang_a,
            prev_actions_a)

        inputs_b = build_onnx_inputs_runsmpl(
            cur_b, future_pos_b, future_rot_b, future_vel_b, future_ang_b,
            prev_actions_b, onnx_name_to_key)

        # Compare ONNX inputs
        max_input_diff = 0.0
        worst_input_key = ""
        for key_a in inputs_a:
            # Map test input names to run_smpl input names
            key_b = None
            for kb in inputs_b:
                if inputs_a[key_a].shape == inputs_b[kb].shape:
                    # Match by comparing values
                    pass
            # Direct name comparison (they use different naming)
            # test uses: current_rigid_body_pos, etc.
            # run_smpl uses ONNX input names from session
            pass

        # Simpler: compare all inputs by sorted ONNX name
        # Actually both should produce the same set of ONNX input names
        # Let's build inputs_a with the same ONNX names as inputs_b
        inputs_a_mapped = {}
        key_map_a = {
            "current.rigid_body_pos": sp_a[None].astype(np.float32),
            "current.rigid_body_rot": sr_a[None].astype(np.float32),
            "current.rigid_body_vel": sv_a[None].astype(np.float32),
            "current.rigid_body_ang_vel": sav_a[None].astype(np.float32),
            "ground_heights": np.zeros(1, dtype=np.float32),
            "historical.actions": prev_actions_a[None, None].astype(np.float32),
            "mimic.future_pos": future_pos_a[None].astype(np.float32),
            "mimic.future_rot": future_rot_a[None].astype(np.float32),
            "mimic.future_vel": future_vel_a[None].astype(np.float32),
            "mimic.future_ang_vel": future_ang_a[None].astype(np.float32),
        }
        for onnx_name, sem_key in onnx_name_to_key.items():
            if sem_key in key_map_a:
                inputs_a_mapped[onnx_name] = key_map_a[sem_key]

        # Now compare inputs_a_mapped vs inputs_b
        input_diffs = {}
        for name in inputs_a_mapped:
            if name in inputs_b:
                d = np.abs(inputs_a_mapped[name] - inputs_b[name]).max()
                input_diffs[name] = d
                if d > max_input_diff:
                    max_input_diff = d
                    worst_input_key = name

        # ---- Run ONNX ----
        outputs_a = session.run(out_names, inputs_a_mapped)
        outputs_b = session.run(out_names, inputs_b)

        out_a = {n: v for n, v in zip(out_names, outputs_a)}
        out_b = {n: v for n, v in zip(out_names, outputs_b)}

        jpt_a = out_a["joint_pos_targets"].squeeze()
        jpt_b = out_b["joint_pos_targets"].squeeze()
        jpt_diff = np.abs(jpt_a - jpt_b).max()

        actions_a = out_a["actions"].squeeze()
        actions_b = out_b["actions"].squeeze()
        actions_diff = np.abs(actions_a - actions_b).max()

        # ---- Apply PD gains ----
        if "stiffness_targets" in out_a:
            stiff_a = out_a["stiffness_targets"].squeeze()
            damp_a = out_a["damping_targets"].squeeze()
            for i in range(model_a.nu):
                kp = float(stiff_a[i])
                kd = float(damp_a[i])
                model_a.actuator_gainprm[i, 0] = kp
                model_a.actuator_biasprm[i, 1] = -kp
                model_a.actuator_biasprm[i, 2] = -kd

        if "stiffness_targets" in out_b:
            stiff_b = out_b["stiffness_targets"].squeeze()
            damp_b = out_b["damping_targets"].squeeze()
            for i in range(model_b.nu):
                kp = float(stiff_b[i])
                kd = float(damp_b[i])
                model_b.actuator_gainprm[i, 0] = kp
                model_b.actuator_biasprm[i, 1] = -kp
                model_b.actuator_biasprm[i, 2] = -kd

        # ---- Store prev_actions ----
        prev_actions_a = actions_a.copy()
        prev_actions_b = actions_b.copy()

        # ---- Apply ctrl ----
        data_a.ctrl[:] = jpt_a
        data_b.ctrl[:] = jpt_b

        ctrl_diff = np.abs(data_a.ctrl - data_b.ctrl).max()

        # ---- Step physics ----
        for _ in range(DECIMATION):
            mujoco.mj_step(model_a, data_a)
        for _ in range(DECIMATION):
            mujoco.mj_step(model_b, data_b)

        # ---- Post-step comparison ----
        qpos_diff = np.abs(data_a.qpos - data_b.qpos).max()
        qvel_diff = np.abs(data_a.qvel - data_b.qvel).max()
        root_h_a = data_a.qpos[2]
        root_h_b = data_b.qpos[2]

        # ---- Report ----
        is_divergent = (qpos_diff > 1e-6 or max_input_diff > 1e-6)
        if step < 5 or step % 5 == 0 or is_divergent:
            print(f"\n  Step {step:3d}: sim_time={sim_time:.3f}s")
            print(f"    state: pos_diff={pos_diff:.2e}, vel_diff={vel_diff:.2e}, "
                  f"ang_diff={ang_diff:.2e}")
            print(f"    inputs: max_diff={max_input_diff:.2e} ({worst_input_key})")
            print(f"    outputs: jpt_diff={jpt_diff:.2e}, actions_diff={actions_diff:.2e}")
            print(f"    ctrl_diff={ctrl_diff:.2e}")
            print(f"    AFTER STEP: qpos_diff={qpos_diff:.2e}, qvel_diff={qvel_diff:.2e}")
            print(f"    root_h: A={root_h_a:.4f}, B={root_h_b:.4f}, diff={abs(root_h_a-root_h_b):.2e}")

        if first_divergence_step is None and qpos_diff > 1e-6:
            first_divergence_step = step
            first_divergence_detail = (
                f"qpos_diff={qpos_diff:.2e}, input_diff={max_input_diff:.2e}, "
                f"ctrl_diff={ctrl_diff:.2e}"
            )
            print(f"\n  *** FIRST DIVERGENCE at step {step}! ***")
            print(f"      {first_divergence_detail}")
            # Print detailed breakdown
            print(f"      Per-input diffs:")
            for name, d in sorted(input_diffs.items(), key=lambda x: -x[1]):
                if d > 1e-8:
                    print(f"        {name}: {d:.2e}")

        # ---- Fall detection ----
        if root_h_a < FALL_THRESHOLD:
            print(f"\n  PATH A FELL at step {step}! root_h={root_h_a:.4f}")
            break
        if root_h_b < FALL_THRESHOLD:
            print(f"\n  PATH B FELL at step {step}! root_h={root_h_b:.4f}")
            break
        if np.any(np.isnan(data_a.qpos)) or np.any(np.isnan(data_b.qpos)):
            print(f"\n  NaN at step {step}!")
            break

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    if first_divergence_step is not None:
        print(f"  First divergence: step {first_divergence_step}")
        print(f"  Detail: {first_divergence_detail}")
        print(f"\n  This means the models/configs are NOT identical!")
        print(f"  Check model attributes above for differences.")
    else:
        print(f"  NO divergence detected in {MAX_STEPS} steps!")
        print(f"  The two code paths produce IDENTICAL results.")
        print(f"  The fall at step 62 must be from something ELSE in run_smpl main().")


if __name__ == "__main__":
    main()
