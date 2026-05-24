#!/usr/bin/env python3
"""Test the two remaining hypotheses for why run_smpl falls at step 67.

Hypothesis 1: The `data.ctrl[:] = ref_qpos[0, 7:]` + extra mj_forward before the loop
              causes divergence (test_physics does NOT do this).

Hypothesis 2: float32 reference arrays in precompute_reference_maxcoords() accumulate
              error vs float64 in test_physics_configs.py's precompute_maxcoords().

This script runs the test_physics_configs approach but adds ONLY one change at a time
to isolate which causes the fall.

Test A: test_physics baseline (should survive 164 steps) — CONTROL
Test B: test_physics + ctrl init (add lines 884-887 from run_smpl)
Test C: test_physics + float32 refs (change precompute to float32)
Test D: test_physics + both changes (matches run_smpl exactly)
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

PHYSICS_DT = 0.001
CONTROL_DT = 0.02
DECIMATION = 20
FALL_THRESHOLD = 0.3
MAX_STEPS = 200


def _quat_mul_wxyz(q1, q2):
    """Quaternion multiply in wxyz convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def mujoco_wxyz_to_xyzw(quats_wxyz):
    """Convert wxyz quats to xyzw."""
    return quats_wxyz[..., [1, 2, 3, 0]]


def precompute_maxcoords(model, data, ref_qpos, dt_ref, use_float32=False):
    """Precompute body max-coords from reference qpos trajectory.

    If use_float32=True, uses np.float32 for storage (matching run_smpl).
    If False, uses default float64 (matching test_physics).
    """
    T = ref_qpos.shape[0]
    num_bodies = 24

    dtype = np.float32 if use_float32 else np.float64

    body_pos = np.zeros((T, num_bodies, 3), dtype=dtype)
    body_rot = np.zeros((T, num_bodies, 4), dtype=dtype)  # xyzw

    for t in range(T):
        data.qpos[:] = ref_qpos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        body_pos[t] = data.xpos[1:num_bodies + 1].copy().astype(dtype)
        body_rot_wxyz = data.xquat[1:num_bodies + 1].copy()
        body_rot[t] = mujoco_wxyz_to_xyzw(body_rot_wxyz).astype(dtype)

    # Backward diff velocity (same logic in both scripts)
    body_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)

    for f in range(1, T):
        body_vel[f] = (body_pos[f] - body_pos[f - 1]) / dt_ref

        for j in range(num_bodies):
            q0 = body_rot[f - 1, j]  # xyzw
            q1 = body_rot[f, j]
            # Convert to wxyz for quat mul
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


def extract_sim_state(model, data):
    """Extract current simulation state (matches BOTH scripts' COM correction)."""
    num_bodies = 24
    sim_pos = data.xpos[1:num_bodies + 1].copy()
    sim_rot_wxyz = data.xquat[1:num_bodies + 1].copy()
    sim_rot = mujoco_wxyz_to_xyzw(sim_rot_wxyz)

    # Velocity with COM correction (matching both scripts)
    sim_vel = np.zeros((num_bodies, 3))
    sim_ang_vel = np.zeros((num_bodies, 3))

    for i in range(num_bodies):
        bid = i + 1
        lin_vel = data.cvel[bid, 3:6].copy()
        ang_vel = data.cvel[bid, 0:3].copy()
        # COM correction: v_com = v + ω × (R @ body_ipos)
        xmat = data.xmat[bid].reshape(3, 3)
        body_ipos = model.body_ipos[bid]
        offset = xmat @ body_ipos
        lin_vel_com = lin_vel + np.cross(ang_vel, offset)
        sim_vel[i] = lin_vel_com
        sim_ang_vel[i] = ang_vel

    return sim_pos, sim_rot, sim_vel, sim_ang_vel


def run_simulation(model, data, ref_qpos, ref_body_pos, ref_body_rot, ref_body_vel,
                   ref_body_ang_vel, dt_ref, session, set_ctrl_init=False, label=""):
    """Run simulation loop matching test_physics_configs logic.

    If set_ctrl_init=True, adds the run_smpl initialization:
      data.ctrl[:] = ref_qpos[0, 7:]
      mj_forward(model, data)
    """
    # Ground height fix (same bilateral logic)
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

    left_min = _lowest_geom_z(left_foot_ids)
    right_min = _lowest_geom_z(right_foot_ids)
    grounding_ref_z = min(left_min, right_min)
    height_shift = 0.0 - grounding_ref_z
    ref_qpos_adj = ref_qpos.copy()
    ref_qpos_adj[:, 2] += height_shift

    # Set initial pose (test_physics style)
    data.qpos[:] = ref_qpos_adj[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # OPTIONAL: set ctrl init (the run_smpl difference)
    if set_ctrl_init:
        data.ctrl[:] = ref_qpos_adj[0, 7:]
        mujoco.mj_forward(model, data)

    # Precompute adjusted refs (re-FK after height adjustment for body pos)
    # We already have ref arrays — just need to adjust body_pos Z by height_shift
    adj_body_pos = ref_body_pos.copy()
    adj_body_pos[:, :, 2] += height_shift

    # Initialize
    prev_actions = np.zeros(69, dtype=np.float32)
    out_names = [o.name for o in session.get_outputs()]
    in_names = [i.name for i in session.get_inputs()]

    num_ref_frames = ref_qpos_adj.shape[0]
    sim_time = 0.0

    for step in range(MAX_STEPS):
        # Extract current sim state
        sim_pos, sim_rot, sim_vel, sim_ang_vel = extract_sim_state(model, data)

        # Future reference (same as test_physics: ref_time = sim_time + CONTROL_DT)
        ref_time = sim_time + CONTROL_DT
        ref_frame_idx = min(int(ref_time / dt_ref), num_ref_frames - 1)

        future_pos = adj_body_pos[ref_frame_idx:ref_frame_idx+1]
        future_rot = ref_body_rot[ref_frame_idx:ref_frame_idx+1]
        future_vel = ref_body_vel[ref_frame_idx:ref_frame_idx+1]
        future_ang_vel = ref_body_ang_vel[ref_frame_idx:ref_frame_idx+1]

        # Build ONNX inputs
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

        # Run ONNX
        outputs = session.run(out_names, inputs)
        out_dict = {name: val for name, val in zip(out_names, outputs)}

        joint_pos_targets = out_dict["joint_pos_targets"].squeeze()
        prev_actions = out_dict["actions"].squeeze().copy()

        # Dynamic PD gains
        if "stiffness_targets" in out_dict and "damping_targets" in out_dict:
            stiff_out = out_dict["stiffness_targets"].squeeze()
            damp_out = out_dict["damping_targets"].squeeze()
            for i in range(model.nu):
                kp = float(stiff_out[i])
                kd = float(damp_out[i])
                model.actuator_gainprm[i, 0] = kp
                model.actuator_biasprm[i, 1] = -kp
                model.actuator_biasprm[i, 2] = -kd

        # Set control
        data.ctrl[:] = joint_pos_targets

        # Step physics
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        sim_time += CONTROL_DT

        # Fall detection
        root_h = data.qpos[2]
        if root_h < FALL_THRESHOLD:
            print(f"  [{label}] FELL at step {step}! root_h={root_h:.4f}")
            return step
        if np.any(np.isnan(data.qpos)):
            print(f"  [{label}] NaN at step {step}!")
            return step

    print(f"  [{label}] SURVIVED {MAX_STEPS} steps! Final root_h={data.qpos[2]:.4f}")
    return MAX_STEPS


def main():
    from test_physics_configs import load_model_with_config
    from run_smpl_rl_tracker import decode_motion_135, yup_to_zup, smpl_to_qpos

    print("=" * 70)
    print("  HYPOTHESIS TEST: Initialization & Float Precision")
    print("=" * 70)

    # Load motion
    smpl_pose, transl, fps = decode_motion_135(NPZ_PATH)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    # Load ONNX session
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    dt_ref = 1.0 / fps
    print(f"  Motion: {NPZ_PATH}")
    print(f"  FPS: {fps}, dt_ref: {dt_ref:.6f}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # TEST A: Baseline (test_physics style: no ctrl init, float64)
    # ═══════════════════════════════════════════════════════════════
    print("─" * 70)
    print("  TEST A: Baseline (no ctrl init, float64 refs)")
    print("─" * 70)
    model_a, data_a, _ = load_model_with_config("D_euler_with_margin")
    body_pos_1 = model_a.body_pos[1].copy()
    ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)
    bp, br, bv, bav = precompute_maxcoords(model_a, data_a, ref_qpos, dt_ref, use_float32=False)
    result_a = run_simulation(model_a, data_a, ref_qpos, bp, br, bv, bav, dt_ref,
                              session, set_ctrl_init=False, label="A: baseline")

    # ═══════════════════════════════════════════════════════════════
    # TEST B: + ctrl init (add run_smpl initialization)
    # ═══════════════════════════════════════════════════════════════
    print()
    print("─" * 70)
    print("  TEST B: + ctrl init (float64 refs)")
    print("─" * 70)
    model_b, data_b, _ = load_model_with_config("D_euler_with_margin")
    bp_b, br_b, bv_b, bav_b = precompute_maxcoords(model_b, data_b, ref_qpos, dt_ref, use_float32=False)
    result_b = run_simulation(model_b, data_b, ref_qpos, bp_b, br_b, bv_b, bav_b, dt_ref,
                              session, set_ctrl_init=True, label="B: +ctrl_init")

    # ═══════════════════════════════════════════════════════════════
    # TEST C: + float32 refs (change precompute to float32)
    # ═══════════════════════════════════════════════════════════════
    print()
    print("─" * 70)
    print("  TEST C: + float32 refs (no ctrl init)")
    print("─" * 70)
    model_c, data_c, _ = load_model_with_config("D_euler_with_margin")
    bp_c, br_c, bv_c, bav_c = precompute_maxcoords(model_c, data_c, ref_qpos, dt_ref, use_float32=True)
    result_c = run_simulation(model_c, data_c, ref_qpos, bp_c, br_c, bv_c, bav_c, dt_ref,
                              session, set_ctrl_init=False, label="C: float32")

    # ═══════════════════════════════════════════════════════════════
    # TEST D: + both changes (matches run_smpl exactly)
    # ═══════════════════════════════════════════════════════════════
    print()
    print("─" * 70)
    print("  TEST D: + ctrl init + float32 refs (run_smpl equivalent)")
    print("─" * 70)
    model_d, data_d, _ = load_model_with_config("D_euler_with_margin")
    bp_d, br_d, bv_d, bav_d = precompute_maxcoords(model_d, data_d, ref_qpos, dt_ref, use_float32=True)
    result_d = run_simulation(model_d, data_d, ref_qpos, bp_d, br_d, bv_d, bav_d, dt_ref,
                              session, set_ctrl_init=True, label="D: both")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  A (baseline: no ctrl_init, float64): survived to step {result_a}")
    print(f"  B (+ctrl_init, float64):             survived to step {result_b}")
    print(f"  C (float32 refs, no ctrl_init):      survived to step {result_c}")
    print(f"  D (+ctrl_init + float32):            survived to step {result_d}")
    print()

    # Diagnosis
    if result_a > result_b:
        print("  → ctrl_init CAUSES earlier fall (B < A)")
    else:
        print("  → ctrl_init has NO impact (B >= A)")

    if result_a > result_c:
        print("  → float32 refs CAUSES earlier fall (C < A)")
    else:
        print("  → float32 has NO impact (C >= A)")

    if result_d < result_a:
        if result_b < result_a and result_c >= result_a:
            print("  → ROOT CAUSE: ctrl_init (not float32)")
        elif result_c < result_a and result_b >= result_a:
            print("  → ROOT CAUSE: float32 precision (not ctrl_init)")
        elif result_b < result_a and result_c < result_a:
            print("  → BOTH contribute to earlier fall")
        else:
            print("  → Combined effect only (neither alone is sufficient)")

    print()
    print("  Expected: A ≈ 164, D ≈ 67 (matching known behavior)")
    print("  If all equal: look elsewhere (sim loop ordering, etc.)")


if __name__ == "__main__":
    main()
