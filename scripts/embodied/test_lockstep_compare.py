#!/usr/bin/env python3
"""Lockstep comparison: run_rl_tracker vs test_init_diff at each step.

This script isolates EXACTLY where the two implementations diverge by running
both side-by-side and comparing all intermediate values at each simulation step.

Approach:
  1. Set up both simulation paths identically (same ref_qpos, same height fix)
  2. At each step, compare: ONNX inputs, outputs, ctrl, post-step qpos
  3. Report the FIRST step where meaningful divergence occurs

The two paths being compared:
  Path A: "test_init_diff Test A" logic (from test_init_diff.py)
  Path B: "run_rl_tracker" logic (from run_smpl_rl_tracker.py)
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

CONTROL_DT = 0.02
DECIMATION = 20
PHYSICS_DT = 0.001
MAX_STEPS = 150
FALL_THRESHOLD = 0.3


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


def precompute_maxcoords_test_init_style(model, data, ref_qpos, dt_ref):
    """Exact copy of test_init_diff.py precompute_maxcoords (use_float32=False)."""
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


def extract_sim_state_test_init_style(model, data):
    """Exact copy of test_init_diff.py extract_sim_state."""
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


def extract_sim_state_run_rl_style(model, data, num_bodies=24):
    """Exact copy of run_smpl_rl_tracker.py extract_sim_state."""
    body_pos = np.zeros((num_bodies, 3))
    body_rot = np.zeros((num_bodies, 4))  # xyzw
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

    return body_pos, body_rot, body_vel, body_ang_vel


def main():
    from test_physics_configs import load_model_with_config
    from run_smpl_rl_tracker import (
        decode_motion_135, yup_to_zup, smpl_to_qpos,
        load_mujoco_model, precompute_reference_maxcoords,
    )

    with open(YAML_PATH) as f:
        yaml_meta = yaml.safe_load(f)

    # Load motion
    smpl_pose, transl, fps = decode_motion_135(NPZ_PATH)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)
    dt_ref = 1.0 / fps

    # Load ONNX
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]

    stiffness = yaml_meta["control"]["stiffness"]
    damping = yaml_meta["control"]["damping"]

    print("=" * 80)
    print("  LOCKSTEP COMPARISON: test_init_diff (Path A) vs run_rl_tracker (Path B)")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # PATH A: test_init_diff style (load_model_with_config + precompute_maxcoords)
    # ═══════════════════════════════════════════════════════════════════════
    model_a, data_a, _ = load_model_with_config("D_euler_with_margin")
    body_pos_1_a = model_a.body_pos[1].copy()
    ref_qpos_a = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1_a)

    # Height fix (same bilateral foot logic as test_init_diff)
    data_a.qpos[:] = ref_qpos_a[0]
    data_a.qvel[:] = 0.0
    mujoco.mj_forward(model_a, data_a)

    left_ids_a = set()
    right_ids_a = set()
    for bid in range(1, model_a.nbody):
        bname = mujoco.mj_id2name(model_a, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in ("L_Ankle", "L_Toe"):
            left_ids_a.add(bid)
        elif bname in ("R_Ankle", "R_Toe"):
            right_ids_a.add(bid)

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

    left_min_a = _lowest_z(left_ids_a, model_a, data_a)
    right_min_a = _lowest_z(right_ids_a, model_a, data_a)
    grounding_z_a = min(left_min_a, right_min_a)
    height_shift_a = 0.0 - grounding_z_a
    ref_qpos_a[:, 2] += height_shift_a
    print(f"  Path A: height_shift = {height_shift_a:+.6f}")

    # Precompute reference (test_init_diff style)
    bp_a, br_a, bv_a, bav_a = precompute_maxcoords_test_init_style(
        model_a, data_a, ref_qpos_a, dt_ref)

    # Set initial pose (Path A: no ctrl init)
    data_a.qpos[:] = ref_qpos_a[0]
    data_a.qvel[:] = 0.0
    mujoco.mj_forward(model_a, data_a)

    # ═══════════════════════════════════════════════════════════════════════
    # PATH B: run_rl_tracker style (load_mujoco_model + precompute_reference_maxcoords)
    # ═══════════════════════════════════════════════════════════════════════
    model_b, data_b = load_mujoco_model(MJCF_PATH, stiffness, damping, PHYSICS_DT)
    body_pos_1_b = model_b.body_pos[1].copy()
    ref_qpos_b = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1_b)

    # Height fix (same logic)
    data_b.qpos[:] = ref_qpos_b[0]
    data_b.qvel[:] = 0.0
    mujoco.mj_forward(model_b, data_b)

    left_ids_b = set()
    right_ids_b = set()
    for bid in range(1, model_b.nbody):
        bname = mujoco.mj_id2name(model_b, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in ("L_Ankle", "L_Toe"):
            left_ids_b.add(bid)
        elif bname in ("R_Ankle", "R_Toe"):
            right_ids_b.add(bid)

    left_min_b = _lowest_z(left_ids_b, model_b, data_b)
    right_min_b = _lowest_z(right_ids_b, model_b, data_b)
    grounding_z_b = min(left_min_b, right_min_b)
    height_shift_b = 0.0 - grounding_z_b
    ref_qpos_b[:, 2] += height_shift_b
    print(f"  Path B: height_shift = {height_shift_b:+.6f}")

    # Precompute reference (run_rl_tracker style)
    ref_data_b = precompute_reference_maxcoords(model_b, data_b, ref_qpos_b, dt_ref)

    # Set initial pose (Path B: no ctrl init, matching run_rl_tracker's current code)
    data_b.qpos[:] = ref_qpos_b[0]
    data_b.qvel[:] = 0.0
    mujoco.mj_forward(model_b, data_b)

    # ═══════════════════════════════════════════════════════════════════════
    # Compare initial state
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  INITIAL STATE COMPARISON")
    print(f"{'─'*80}")

    qpos_diff = np.abs(ref_qpos_a - ref_qpos_b).max()
    print(f"  ref_qpos max diff: {qpos_diff:.2e}")

    # Compare precomputed references
    bp_b = ref_data_b["body_pos"]
    br_b = ref_data_b["body_rot"]
    bv_b = ref_data_b["body_vel"]
    bav_b_arr = ref_data_b["body_ang_vel"]

    print(f"  body_pos max diff: {np.abs(bp_a - bp_b).max():.2e}")
    print(f"  body_rot max diff: {np.abs(br_a - br_b).max():.2e}")
    print(f"  body_vel max diff: {np.abs(bv_a - bv_b).max():.2e}")
    print(f"  body_ang_vel max diff: {np.abs(bav_a - bav_b_arr).max():.2e}")

    if np.abs(bv_a - bv_b).max() > 1e-8 or np.abs(bav_a - bav_b_arr).max() > 1e-8:
        print("\n  !!! PRECOMPUTED REFERENCES DIFFER !!!")
        # Find where they differ
        for name, arr_a, arr_b in [("body_vel", bv_a, bv_b), ("body_ang_vel", bav_a, bav_b_arr)]:
            diff = np.abs(arr_a - arr_b)
            if diff.max() > 1e-8:
                frame_diffs = diff.max(axis=(1, 2))
                worst_frame = np.argmax(frame_diffs)
                joint_diffs = diff[worst_frame].max(axis=1)
                worst_joint = np.argmax(joint_diffs)
                print(f"  {name}: worst frame={worst_frame}, joint={worst_joint}, "
                      f"diff={diff[worst_frame, worst_joint].max():.8e}")
                print(f"    A[{worst_frame},{worst_joint}] = {arr_a[worst_frame, worst_joint]}")
                print(f"    B[{worst_frame},{worst_joint}] = {arr_b[worst_frame, worst_joint]}")

    # Compare initial data.qpos
    init_qpos_diff = np.abs(data_a.qpos - data_b.qpos).max()
    print(f"\n  Initial data.qpos max diff: {init_qpos_diff:.2e}")

    # Compare model parameters that affect simulation
    print(f"\n  Model parameter comparison:")
    print(f"    model.opt.timestep: A={model_a.opt.timestep}, B={model_b.opt.timestep}")
    print(f"    model.opt.integrator: A={model_a.opt.integrator}, B={model_b.opt.integrator}")
    print(f"    model.opt.solver: A={model_a.opt.solver}, B={model_b.opt.solver}")
    print(f"    model.opt.iterations: A={model_a.opt.iterations}, B={model_b.opt.iterations}")
    print(f"    model.opt.noslip_iterations: A={model_a.opt.noslip_iterations}, B={model_b.opt.noslip_iterations}")
    print(f"    model.opt.gravity: A={model_a.opt.gravity}, B={model_b.opt.gravity}")
    # Note: collision/cone/impratio may not exist in all mujoco versions
    for attr in ['cone', 'impratio']:
        if hasattr(model_a.opt, attr):
            print(f"    model.opt.{attr}: A={getattr(model_a.opt, attr)}, B={getattr(model_b.opt, attr)}")
    gain_diff = np.abs(model_a.actuator_gainprm[:, 0] - model_b.actuator_gainprm[:, 0]).max()
    bias1_diff = np.abs(model_a.actuator_biasprm[:, 1] - model_b.actuator_biasprm[:, 1]).max()
    bias2_diff = np.abs(model_a.actuator_biasprm[:, 2] - model_b.actuator_biasprm[:, 2]).max()
    print(f"    actuator_gainprm[:,0] max diff: {gain_diff:.2e}")
    print(f"    actuator_biasprm[:,1] max diff: {bias1_diff:.2e}")
    print(f"    actuator_biasprm[:,2] max diff: {bias2_diff:.2e}")

    # Check margin
    print(f"    geom_margin: A max={model_a.geom_margin.max():.4f}, B max={model_b.geom_margin.max():.4f}")
    margin_diff = np.abs(model_a.geom_margin - model_b.geom_margin)
    if margin_diff.max() > 0:
        print(f"    !!! geom_margin differs! max diff = {margin_diff.max():.4f}")
        for gid in range(model_a.ngeom):
            if margin_diff[gid] > 0:
                gname_a = mujoco.mj_id2name(model_a, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"g{gid}"
                print(f"      geom[{gid}] '{gname_a}': A={model_a.geom_margin[gid]:.4f}, B={model_b.geom_margin[gid]:.4f}")

    # Check contype/conaffinity differences
    ngeom_min = min(model_a.ngeom, model_b.ngeom)
    contype_diff = np.abs(model_a.geom_contype[:ngeom_min].astype(int) - model_b.geom_contype[:ngeom_min].astype(int))
    conaffinity_diff = np.abs(model_a.geom_conaffinity[:ngeom_min].astype(int) - model_b.geom_conaffinity[:ngeom_min].astype(int))
    if contype_diff.max() > 0 or conaffinity_diff.max() > 0:
        print(f"    !!! contype/conaffinity differ!")
        for gid in range(ngeom_min):
            if contype_diff[gid] > 0 or conaffinity_diff[gid] > 0:
                gname = mujoco.mj_id2name(model_a, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"g{gid}"
                print(f"      geom[{gid}] '{gname}': "
                      f"contype A={model_a.geom_contype[gid]} B={model_b.geom_contype[gid]}, "
                      f"conaffinity A={model_a.geom_conaffinity[gid]} B={model_b.geom_conaffinity[gid]}")

    # Check ngeom counts
    print(f"    ngeom: A={model_a.ngeom}, B={model_b.ngeom}")

    # Check floor geom specifically
    print(f"\n  Floor geom comparison:")
    for model_x, label in [(model_a, "A"), (model_b, "B")]:
        for gid in range(model_x.ngeom):
            gname = mujoco.mj_id2name(model_x, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname and "floor" in gname:
                gtype = model_x.geom_type[gid]
                gsize = model_x.geom_size[gid]
                gpos = model_x.geom_pos[gid]
                contype = model_x.geom_contype[gid]
                conaffinity = model_x.geom_conaffinity[gid]
                condim = model_x.geom_condim[gid]
                margin = model_x.geom_margin[gid]
                print(f"    Path {label} floor: gid={gid}, type={gtype}, size={gsize}, pos={gpos}, "
                      f"contype={contype}, conaffinity={conaffinity}, condim={condim}, margin={margin}")

    # Check forcerange
    fr_a = model_a.actuator_forcerange
    fr_b = model_b.actuator_forcerange
    fr_diff = np.abs(fr_a - fr_b).max()
    print(f"    forcerange max diff: {fr_diff:.2e}")
    print(f"    forcerange A[0]: {fr_a[0]}")
    print(f"    forcerange B[0]: {fr_b[0]}")

    # Check dof_damping, jnt_stiffness
    dd_diff = np.abs(model_a.dof_damping - model_b.dof_damping).max()
    js_diff = np.abs(model_a.jnt_stiffness - model_b.jnt_stiffness).max()
    df_diff = np.abs(model_a.dof_frictionloss - model_b.dof_frictionloss).max()
    print(f"    dof_damping max diff: {dd_diff:.2e}")
    print(f"    jnt_stiffness max diff: {js_diff:.2e}")
    print(f"    dof_frictionloss max diff: {df_diff:.2e}")

    # ═══════════════════════════════════════════════════════════════════════
    # LOCKSTEP SIMULATION
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  LOCKSTEP SIMULATION")
    print(f"{'═'*80}")

    prev_actions_a = np.zeros(69, dtype=np.float32)
    prev_actions_b = np.zeros(69, dtype=np.float32)

    num_ref_frames = ref_qpos_a.shape[0]
    sim_time = 0.0

    first_diverge_step = None

    for step in range(MAX_STEPS):
        # ─── Extract sim state ───
        sim_pos_a, sim_rot_a, sim_vel_a, sim_ang_vel_a = extract_sim_state_test_init_style(model_a, data_a)
        sim_pos_b, sim_rot_b, sim_vel_b, sim_ang_vel_b = extract_sim_state_run_rl_style(model_b, data_b)

        # Compare sim states
        pos_diff = np.abs(sim_pos_a - sim_pos_b).max()
        rot_diff = np.abs(sim_rot_a - sim_rot_b).max()
        vel_diff = np.abs(sim_vel_a - sim_vel_b).max()
        ang_diff = np.abs(sim_ang_vel_a - sim_ang_vel_b).max()

        # ─── Get future reference ───
        # Path A: test_init_diff style
        ref_time_a = sim_time + CONTROL_DT
        ref_frame_idx_a = min(int(ref_time_a / dt_ref), num_ref_frames - 1)
        future_pos_a = bp_a[ref_frame_idx_a:ref_frame_idx_a+1]
        future_rot_a = br_a[ref_frame_idx_a:ref_frame_idx_a+1]
        future_vel_a = bv_a[ref_frame_idx_a:ref_frame_idx_a+1]
        future_ang_vel_a = bav_a[ref_frame_idx_a:ref_frame_idx_a+1]

        # Path B: run_rl_tracker style (fi=1, fdt=0.02)
        future_time_b = sim_time + 1 * 0.02  # fi * fdt
        ref_frame_idx_b = min(int(future_time_b / dt_ref), num_ref_frames - 1)
        future_pos_b = ref_data_b["body_pos"][ref_frame_idx_b:ref_frame_idx_b+1]
        future_rot_b = ref_data_b["body_rot"][ref_frame_idx_b:ref_frame_idx_b+1]
        future_vel_b = ref_data_b["body_vel"][ref_frame_idx_b:ref_frame_idx_b+1]
        future_ang_vel_b = ref_data_b["body_ang_vel"][ref_frame_idx_b:ref_frame_idx_b+1]

        # Compare future references
        fut_pos_diff = np.abs(future_pos_a - future_pos_b).max()
        fut_rot_diff = np.abs(future_rot_a - future_rot_b).max()
        fut_vel_diff = np.abs(future_vel_a - future_vel_b).max()
        fut_ang_diff = np.abs(future_ang_vel_a - future_ang_vel_b).max()

        # ─── Build ONNX inputs ───
        inputs_a = {
            "current_rigid_body_ang_vel": sim_ang_vel_a[np.newaxis].astype(np.float32),
            "current_rigid_body_pos": sim_pos_a[np.newaxis].astype(np.float32),
            "current_rigid_body_rot": sim_rot_a[np.newaxis].astype(np.float32),
            "current_rigid_body_vel": sim_vel_a[np.newaxis].astype(np.float32),
            "ground_heights": np.zeros((1,), dtype=np.float32),
            "historical_actions": prev_actions_a[np.newaxis, np.newaxis].astype(np.float32),
            "mimic_future_ang_vel": future_ang_vel_a[np.newaxis].astype(np.float32),
            "mimic_future_pos": future_pos_a[np.newaxis].astype(np.float32),
            "mimic_future_rot": future_rot_a[np.newaxis].astype(np.float32),
            "mimic_future_vel": future_vel_a[np.newaxis].astype(np.float32),
        }

        inputs_b = {
            "current_rigid_body_ang_vel": sim_ang_vel_b[np.newaxis].astype(np.float32),
            "current_rigid_body_pos": sim_pos_b[np.newaxis].astype(np.float32),
            "current_rigid_body_rot": sim_rot_b[np.newaxis].astype(np.float32),
            "current_rigid_body_vel": sim_vel_b[np.newaxis].astype(np.float32),
            "ground_heights": np.zeros((1,), dtype=np.float32),
            "historical_actions": prev_actions_b[np.newaxis, np.newaxis].astype(np.float32),
            "mimic_future_ang_vel": future_ang_vel_b[np.newaxis].astype(np.float32),
            "mimic_future_pos": future_pos_b[np.newaxis].astype(np.float32),
            "mimic_future_rot": future_rot_b[np.newaxis].astype(np.float32),
            "mimic_future_vel": future_vel_b[np.newaxis].astype(np.float32),
        }

        # Compare ONNX inputs
        onnx_input_max_diff = 0.0
        for key in inputs_a:
            d = np.abs(inputs_a[key] - inputs_b[key]).max()
            onnx_input_max_diff = max(onnx_input_max_diff, d)

        # ─── Run ONNX ───
        outputs_a = session.run(out_names, inputs_a)
        outputs_b = session.run(out_names, inputs_b)
        out_dict_a = {n: v for n, v in zip(out_names, outputs_a)}
        out_dict_b = {n: v for n, v in zip(out_names, outputs_b)}

        jpt_a = out_dict_a["joint_pos_targets"].squeeze()
        jpt_b = out_dict_b["joint_pos_targets"].squeeze()
        jpt_diff = np.abs(jpt_a - jpt_b).max()

        # ─── Apply dynamic PD gains ───
        if "stiffness_targets" in out_dict_a:
            stiff_a = out_dict_a["stiffness_targets"].squeeze()
            damp_a = out_dict_a["damping_targets"].squeeze()
            for i in range(model_a.nu):
                model_a.actuator_gainprm[i, 0] = float(stiff_a[i])
                model_a.actuator_biasprm[i, 1] = -float(stiff_a[i])
                model_a.actuator_biasprm[i, 2] = -float(damp_a[i])

        if "stiffness_targets" in out_dict_b:
            stiff_b = out_dict_b["stiffness_targets"].squeeze()
            damp_b = out_dict_b["damping_targets"].squeeze()
            for i in range(model_b.nu):
                model_b.actuator_gainprm[i, 0] = float(stiff_b[i])
                model_b.actuator_biasprm[i, 1] = -float(stiff_b[i])
                model_b.actuator_biasprm[i, 2] = -float(damp_b[i])

        # ─── Set ctrl ───
        data_a.ctrl[:] = jpt_a
        data_b.ctrl[:] = jpt_b

        # Store raw actions
        prev_actions_a = out_dict_a["actions"].squeeze().copy()
        prev_actions_b = out_dict_b["actions"].squeeze().copy()

        # ─── Step physics ───
        for _ in range(DECIMATION):
            mujoco.mj_step(model_a, data_a)
        for _ in range(DECIMATION):
            mujoco.mj_step(model_b, data_b)

        sim_time += CONTROL_DT

        # ─── Post-step comparison ───
        qpos_post_diff = np.abs(data_a.qpos - data_b.qpos).max()
        root_h_a = data_a.qpos[2]
        root_h_b = data_b.qpos[2]

        # Print summary
        total_diff = max(pos_diff, rot_diff, vel_diff, ang_diff, fut_pos_diff,
                        fut_rot_diff, fut_vel_diff, fut_ang_diff, jpt_diff, qpos_post_diff)

        if step < 5 or step % 10 == 0 or total_diff > 1e-4 or root_h_a < 0.5 or root_h_b < 0.5:
            print(f"  step={step:4d}  h_A={root_h_a:.4f}  h_B={root_h_b:.4f}  "
                  f"onnx_in={onnx_input_max_diff:.2e}  jpt={jpt_diff:.2e}  "
                  f"qpos_post={qpos_post_diff:.2e}  ref_idx={ref_frame_idx_a}")

        if total_diff > 1e-4 and first_diverge_step is None:
            first_diverge_step = step
            print(f"\n  !!! FIRST SIGNIFICANT DIVERGENCE at step {step} !!!")
            print(f"      sim_pos diff:  {pos_diff:.2e}")
            print(f"      sim_rot diff:  {rot_diff:.2e}")
            print(f"      sim_vel diff:  {vel_diff:.2e}")
            print(f"      sim_ang diff:  {ang_diff:.2e}")
            print(f"      fut_pos diff:  {fut_pos_diff:.2e}")
            print(f"      fut_rot diff:  {fut_rot_diff:.2e}")
            print(f"      fut_vel diff:  {fut_vel_diff:.2e}")
            print(f"      fut_ang diff:  {fut_ang_diff:.2e}")
            print(f"      jpt diff:      {jpt_diff:.2e}")
            print(f"      qpos_post:     {qpos_post_diff:.2e}")
            # Print per-input details
            for key in sorted(inputs_a.keys()):
                d = np.abs(inputs_a[key] - inputs_b[key]).max()
                if d > 1e-6:
                    print(f"      ONNX '{key}' diff: {d:.2e}")
            print()

        # Fall detection
        if root_h_a < FALL_THRESHOLD:
            print(f"  PATH A FELL at step {step}! root_h={root_h_a:.4f}")
            if root_h_b >= FALL_THRESHOLD:
                print(f"  PATH B still alive: root_h={root_h_b:.4f}")
            break
        if root_h_b < FALL_THRESHOLD:
            print(f"  PATH B FELL at step {step}! root_h={root_h_b:.4f}")
            if root_h_a >= FALL_THRESHOLD:
                print(f"  PATH A still alive: root_h={root_h_a:.4f}")
            break
        if np.any(np.isnan(data_a.qpos)) or np.any(np.isnan(data_b.qpos)):
            print(f"  NaN at step {step}!")
            break
    else:
        print(f"\n  BOTH survived {MAX_STEPS} steps!")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  SUMMARY")
    print(f"{'═'*80}")
    if first_diverge_step is not None:
        print(f"  First significant divergence at step: {first_diverge_step}")
    else:
        print(f"  Paths remained identical (within 1e-4) throughout simulation")
    print(f"  Final root heights: A={data_a.qpos[2]:.4f}, B={data_b.qpos[2]:.4f}")


if __name__ == "__main__":
    main()
