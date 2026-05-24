#!/usr/bin/env python3
"""Diagnostic: Test RL tracker with different MuJoCo physics configurations.

Hypothesis: The custom contact parameters (IMPLICITFAST integrator + solref/solimp/margin)
we added to approximate IsaacGym may be HURTING rather than helping.

ProtoMotions' own MuJoCo simulator uses:
  - Default Euler integrator (NOT implicitfast)
  - NO solref/solimp overrides (uses MuJoCo defaults)
  - NO margin overrides (margin=0.0, from MJCF)
  - Does NOT override gear (but that's a bug for SMPL; we correctly fix gear=1)

This script tests 4 configs to isolate the cause:
  A: Current (IMPLICITFAST + solref=[0.015,1.0] + solimp=[0.9,0.99,0.003] + margin=0.02)
  B: ProtoMotions defaults (Euler, no solref/solimp/margin overrides)
  C: IMPLICITFAST + no contact overrides (isolate integrator effect)
  D: Euler + margin=0.02 (isolate margin effect)

Usage:
    python3 scripts/embodied/test_physics_configs.py [--npz-file PATH]
"""

import numpy as np
import mujoco
import onnxruntime as ort
import sys
import os
import yaml
import tempfile
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(__file__))

# ===== Paths =====
MJCF_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
YAML_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.yaml"
ONNX_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.onnx"

# ===== Load YAML config =====
with open(YAML_PATH) as f:
    YAML_META = yaml.safe_load(f)

DEFAULT_STIFFNESS = YAML_META["control"]["stiffness"]
DEFAULT_DAMPING = YAML_META["control"]["damping"]
PHYSICS_DT = 0.001
CONTROL_DT = 0.02
DECIMATION = 20

MUJOCO_BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]


def patch_mjcf_xml(mjcf_path):
    """Patch MJCF to add floor and light (minimal changes)."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Remove sensors (can cause issues standalone)
    for sensor_elem in root.findall("sensor"):
        root.remove(sensor_elem)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        # Check if floor already exists
        has_floor = any(
            g.get("name") == "floor" or g.get("type") == "plane"
            for g in worldbody.findall("geom")
        )
        if not has_floor:
            floor = ET.SubElement(worldbody, "geom")
            floor.set("name", "floor")
            floor.set("type", "plane")
            floor.set("size", "50 50 1")
            floor.set("pos", "0 0 0")
            floor.set("contype", "1")
            floor.set("conaffinity", "1")
            floor.set("condim", "3")

        # Add light if missing
        if not worldbody.findall("light"):
            light = ET.SubElement(worldbody, "light")
            light.set("name", "top_light")
            light.set("pos", "0 0 3")
            light.set("dir", "0 0 -1")
            light.set("diffuse", "0.8 0.8 0.8")

    return ET.tostring(root, encoding="unicode")


def load_model_with_config(config_name: str) -> tuple:
    """Load MuJoCo model with specified physics configuration.

    Returns (model, data, config_desc)
    """
    # Write patched XML
    patched = patch_mjcf_xml(MJCF_PATH)
    asset_dir = str(Path(MJCF_PATH).parent)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=asset_dir, delete=False) as tmp:
        tmp.write(patched)
        tmp_path = tmp.name
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    data = mujoco.MjData(model)

    # Common settings for ALL configs
    model.opt.timestep = PHYSICS_DT

    # Zero passive forces (all configs need this)
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0

    # Gear=1 (all configs need this for correct PD)
    model.actuator_gear[:, 0] = 1.0

    # Configure PD actuators (same for all configs)
    for i in range(model.nu):
        kp = DEFAULT_STIFFNESS[i]
        kd = DEFAULT_DAMPING[i]
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biastype[i] = 1
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kd
        model.actuator_ctrllimited[i] = 0
        model.actuator_forcerange[i, 0] = -500.0
        model.actuator_forcerange[i, 1] = 500.0
        model.actuator_forcelimited[i] = 1

    # Disable self-collision (all configs)
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for geom_id in range(model.ngeom):
        if geom_id == floor_geom_id:
            continue
        model.geom_conaffinity[geom_id] = 0

    # ===== CONFIG-SPECIFIC SETTINGS =====
    if config_name == "A_current":
        # Current settings: IMPLICITFAST + custom contact params + margin
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for geom_id in range(model.ngeom):
            model.geom_solref[geom_id, 0] = 0.015
            model.geom_solref[geom_id, 1] = 1.0
            model.geom_solimp[geom_id, 0] = 0.9
            model.geom_solimp[geom_id, 1] = 0.99
            model.geom_solimp[geom_id, 2] = 0.003
            model.geom_margin[geom_id] = 0.02
        desc = "IMPLICITFAST + solref=[0.015,1] + solimp=[0.9,0.99,0.003] + margin=0.02"

    elif config_name == "B_protomotions_defaults":
        # ProtoMotions MuJoCo defaults: Euler, no contact overrides
        # MuJoCo defaults: integrator=Euler, solref=[0.02,1], solimp=[0.9,0.95,0.001,0.5,2], margin from MJCF (0.001)
        # We DON'T touch integrator (default=Euler), solref, solimp, or margin
        desc = "Euler (default) + MuJoCo default contact params + MJCF margin"

    elif config_name == "C_implicitfast_no_contact":
        # IMPLICITFAST integrator but NO contact param overrides
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        desc = "IMPLICITFAST + MuJoCo default contact params + MJCF margin"

    elif config_name == "D_euler_with_margin":
        # Euler integrator but WITH margin=0.02
        for geom_id in range(model.ngeom):
            model.geom_margin[geom_id] = 0.02
        desc = "Euler (default) + margin=0.02 (no solref/solimp override)"

    elif config_name == "E_implicitfast_margin_only":
        # IMPLICITFAST + margin=0.02 but no solref/solimp
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for geom_id in range(model.ngeom):
            model.geom_margin[geom_id] = 0.02
        desc = "IMPLICITFAST + margin=0.02 (no solref/solimp override)"

    elif config_name == "F_euler_soft_contacts":
        # Euler + soft contacts (solref/solimp) but NO margin
        for geom_id in range(model.ngeom):
            model.geom_solref[geom_id, 0] = 0.015
            model.geom_solref[geom_id, 1] = 1.0
            model.geom_solimp[geom_id, 0] = 0.9
            model.geom_solimp[geom_id, 1] = 0.99
            model.geom_solimp[geom_id, 2] = 0.003
        desc = "Euler + solref=[0.015,1] + solimp=[0.9,0.99,0.003] + no margin"

    else:
        raise ValueError(f"Unknown config: {config_name}")

    return model, data, desc


def load_reference_motion(npz_path):
    """Load reference motion and convert to qpos (delegates to run_smpl_rl_tracker)."""
    from run_smpl_rl_tracker import decode_motion_135, yup_to_zup, smpl_to_qpos
    smpl_pose, transl, fps = decode_motion_135(npz_path)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)
    return smpl_pose_zup, transl_zup, fps


def precompute_maxcoords(model, data, ref_qpos, dt_ref):
    """Compute FK on reference frames for RL policy inputs."""
    num_frames = ref_qpos.shape[0]
    num_bodies = len(MUJOCO_BODY_NAMES)

    body_ids = []
    for name in MUJOCO_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"Body '{name}' not found"
        body_ids.append(bid)

    body_pos = np.zeros((num_frames, num_bodies, 3))
    body_rot = np.zeros((num_frames, num_bodies, 4))  # xyzw

    for f in range(num_frames):
        data.qpos[:] = ref_qpos[f]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        for j, bid in enumerate(body_ids):
            body_pos[f, j] = data.xpos[bid].copy()
            quat_wxyz = data.xquat[bid].copy()
            body_rot[f, j] = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]  # xyzw

    # Compute velocities via finite difference
    body_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)
    for f in range(1, num_frames):
        body_vel[f] = (body_pos[f] - body_pos[f - 1]) / dt_ref
    body_vel[0] = body_vel[1] if num_frames > 1 else 0.0

    # Angular velocity from quaternion difference (simplified)
    for f in range(1, num_frames):
        for j in range(num_bodies):
            q0 = body_rot[f - 1, j]  # xyzw
            q1 = body_rot[f, j]
            # Convert to wxyz for computation
            q0_w = np.array([q0[3], q0[0], q0[1], q0[2]])
            q1_w = np.array([q1[3], q1[0], q1[1], q1[2]])
            # dq = q1 * q0_inv
            q0_inv = np.array([q0_w[0], -q0_w[1], -q0_w[2], -q0_w[3]])
            dq = quat_mul_wxyz(q1_w, q0_inv)
            # ang_vel ≈ 2 * vec(dq) / dt
            if dq[0] < 0:
                dq = -dq
            body_ang_vel[f, j] = 2.0 * dq[1:4] / dt_ref
    body_ang_vel[0] = body_ang_vel[1] if num_frames > 1 else 0.0

    return body_pos, body_rot, body_vel, body_ang_vel


def quat_mul_wxyz(q1, q2):
    """Multiply two quaternions in wxyz format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def extract_sim_state(model, data):
    """Extract current body state from simulation for RL policy."""
    num_bodies = len(MUJOCO_BODY_NAMES)
    body_ids = []
    for name in MUJOCO_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        body_ids.append(bid)

    sim_pos = np.zeros((num_bodies, 3))
    sim_rot = np.zeros((num_bodies, 4))  # xyzw
    sim_vel = np.zeros((num_bodies, 3))
    sim_ang_vel = np.zeros((num_bodies, 3))

    for j, bid in enumerate(body_ids):
        sim_pos[j] = data.xpos[bid].copy()
        quat_wxyz = data.xquat[bid].copy()
        sim_rot[j] = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

        # COM velocity correction: v_COM = v_frame + ω × r_offset
        cvel = data.cvel[bid]  # (6,) = [ang_vel(3), lin_vel(3)]
        ang_vel = cvel[:3].copy()
        lin_vel = cvel[3:].copy()

        # COM offset in world frame
        com_local = model.body_ipos[bid]
        xmat = data.xmat[bid].reshape(3, 3)
        com_world = xmat @ com_local

        # v_COM = v_frame + ω × r_COM_world
        lin_vel_com = lin_vel + np.cross(ang_vel, com_world)

        sim_vel[j] = lin_vel_com
        sim_ang_vel[j] = ang_vel

    return sim_pos, sim_rot, sim_vel, sim_ang_vel


def run_tracking_test(config_name: str, ref_qpos: np.ndarray,
                      ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel,
                      onnx_session, dt_ref: float, max_steps: int = 200):
    """Run RL tracking with specified physics config.

    Returns dict with metrics.
    """
    model, data, desc = load_model_with_config(config_name)

    print(f"\n{'='*70}")
    print(f"  Config: {config_name}")
    print(f"  {desc}")
    print(f"{'='*70}")

    # Get body IDs for this model
    body_ids = []
    for name in MUJOCO_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        body_ids.append(bid)

    # Set initial pose
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Initialize previous actions to zero
    prev_actions = np.zeros(69, dtype=np.float32)

    # ONNX input/output names
    in_names = [inp.name for inp in onnx_session.get_inputs()]
    out_names = [out.name for out in onnx_session.get_outputs()]

    num_ref_frames = ref_qpos.shape[0]
    sim_time = 0.0
    fall_step = -1

    root_heights = []
    tracking_errors = []

    print(f"  {'step':>5s}  {'root_h':>7s}  {'trk_err':>8s}  {'ncon':>5s}  "
          f"{'max_tau':>8s}  {'qvel_max':>9s}  {'ref_frame':>10s}")

    for step in range(max_steps):
        # Get current sim state
        sim_pos, sim_rot, sim_vel, sim_ang_vel = extract_sim_state(model, data)

        # Get reference state for next step (future_step_indices=[1] → 1 control step ahead)
        ref_time = sim_time + CONTROL_DT
        ref_frame_idx = min(int(ref_time / dt_ref), num_ref_frames - 1)

        future_pos = ref_body_pos[ref_frame_idx:ref_frame_idx+1]  # (1, 24, 3)
        future_rot = ref_body_rot[ref_frame_idx:ref_frame_idx+1]  # (1, 24, 4)
        future_vel = ref_body_vel[ref_frame_idx:ref_frame_idx+1]  # (1, 24, 3)
        future_ang_vel = ref_body_ang_vel[ref_frame_idx:ref_frame_idx+1]  # (1, 24, 3)

        # Build ONNX inputs (alphabetical order matching YAML)
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

        # Run ONNX inference
        outputs = onnx_session.run(out_names, inputs)
        out_dict = {name: val for name, val in zip(out_names, outputs)}

        # Extract outputs
        joint_pos_targets = out_dict["joint_pos_targets"].squeeze()
        prev_actions = out_dict["actions"].squeeze().copy()

        # Apply dynamic PD gains if available
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

        # Metrics
        root_h = data.qpos[2]
        root_heights.append(root_h)

        # Tracking error (body position)
        curr_sim_pos, _, _, _ = extract_sim_state(model, data)
        trk_err = np.mean(np.linalg.norm(curr_sim_pos - ref_body_pos[ref_frame_idx], axis=-1))
        tracking_errors.append(trk_err)

        max_tau = np.abs(data.qfrc_actuator).max()
        qvel_max = np.abs(data.qvel).max()

        if step % 10 == 0 or step < 5 or root_h < 0.5:
            print(f"  {step:5d}  {root_h:7.4f}  {trk_err:8.4f}  {data.ncon:5d}  "
                  f"{max_tau:8.1f}  {qvel_max:9.4f}  {ref_frame_idx:10d}")

        # Fall detection
        if root_h < 0.3:
            fall_step = step
            print(f"  >>> FELL at step {step}! root_h={root_h:.4f}")
            break

        if np.any(np.isnan(data.qpos)):
            fall_step = step
            print(f"  >>> NaN at step {step}!")
            break

    if fall_step < 0:
        print(f"  >>> SURVIVED {max_steps} steps! Final root_h={data.qpos[2]:.4f}")

    return {
        "config": config_name,
        "desc": desc,
        "fall_step": fall_step,
        "max_steps": max_steps,
        "survived": fall_step < 0,
        "final_root_h": float(data.qpos[2]),
        "min_root_h": float(min(root_heights)) if root_heights else 0,
        "mean_tracking_error": float(np.mean(tracking_errors)) if tracking_errors else 0,
        "root_heights": root_heights,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-file", type=str, default=None,
                        help="Path to motion NPZ file (motion_135 format)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max control steps to simulate")
    args = parser.parse_args()

    # Find a reference motion
    npz_candidates = [
        args.npz_file,
        "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4/data/npz/walk_forward.npz",
        "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4/data/npz/example_walk.npz",
    ]
    npz_path = None
    for p in npz_candidates:
        if p and os.path.exists(p):
            npz_path = p
            break

    if npz_path is None:
        print("ERROR: No reference NPZ found. Pass --npz-file PATH")
        sys.exit(1)

    print(f"Reference motion: {npz_path}")

    # Import conversion functions
    from run_smpl_rl_tracker import decode_motion_135, yup_to_zup, smpl_to_qpos

    # Decode: NPZ → SMPL axis-angle (Y-up) → Z-up → qpos
    smpl_pose, transl, fps = decode_motion_135(npz_path)
    smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)

    # Load a reference model for FK (use config B - defaults)
    model_ref, data_ref, _ = load_model_with_config("B_protomotions_defaults")
    body_pos_1 = model_ref.body_pos[1].copy()

    qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)
    print(f"Motion: {qpos.shape[0]} frames @ {fps}fps = {qpos.shape[0]/fps:.2f}s")

    # Ground offset: bilateral foot grounding (same logic as run_smpl_rl_tracker.py)
    # Place feet at floor level (Z=0) for proper ground contact
    data_ref.qpos[:] = qpos[0]
    data_ref.qvel[:] = 0.0
    mujoco.mj_forward(model_ref, data_ref)

    # Find lowest geom surface Z for left/right foot groups
    left_foot_ids = set()
    right_foot_ids = set()
    for bid in range(1, model_ref.nbody):
        bname = mujoco.mj_id2name(model_ref, mujoco.mjtObj.mjOBJ_BODY, bid)
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

    left_min = _lowest_geom_z(left_foot_ids, model_ref, data_ref)
    right_min = _lowest_geom_z(right_foot_ids, model_ref, data_ref)
    # Ground using the LOWER foot (ensure zero penetration)
    grounding_ref_z = min(left_min, right_min)
    height_shift = 0.0 - grounding_ref_z  # target = 0.0 (floor level)
    qpos[:, 2] += height_shift

    dt_ref = 1.0 / fps
    print(f"Ground offset: {height_shift:+.4f}m (L_foot={left_min:.4f}, R_foot={right_min:.4f})")
    print(f"Initial root height: {qpos[0, 2]:.4f}m")

    # Precompute reference maxcoords
    print("\nPrecomputing reference FK...")
    ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel = precompute_maxcoords(
        model_ref, data_ref, qpos, dt_ref)
    print(f"  Body pos range: Z=[{ref_body_pos[:,:,2].min():.3f}, {ref_body_pos[:,:,2].max():.3f}]")

    # Load ONNX policy
    print(f"\nLoading ONNX policy: {ONNX_PATH}")
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(ONNX_PATH, sess_opts, providers=["CPUExecutionProvider"])
    print(f"  Inputs: {[inp.name for inp in session.get_inputs()]}")
    print(f"  Outputs: {[out.name for out in session.get_outputs()]}")

    # Run all configs
    configs = [
        "A_current",
        "B_protomotions_defaults",
        "C_implicitfast_no_contact",
        "D_euler_with_margin",
        "E_implicitfast_margin_only",
        "F_euler_soft_contacts",
    ]

    results = []
    for config_name in configs:
        result = run_tracking_test(
            config_name, qpos,
            ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel,
            session, dt_ref, max_steps=args.max_steps)
        results.append(result)

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: Physics Configuration Comparison")
    print(f"{'='*70}")
    print(f"  {'Config':<35s} {'Survived?':>10s} {'Fall Step':>10s} {'Min H':>8s} {'Trk Err':>8s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in results:
        survived = "YES" if r["survived"] else "NO"
        fall = str(r["fall_step"]) if r["fall_step"] >= 0 else f">{r['max_steps']}"
        print(f"  {r['config']:<35s} {survived:>10s} {fall:>10s} "
              f"{r['min_root_h']:8.4f} {r['mean_tracking_error']:8.4f}")

    print(f"\n  Config descriptions:")
    for r in results:
        print(f"    {r['config']}: {r['desc']}")

    # Conclusion
    print(f"\n  {'='*70}")
    print(f"  ANALYSIS:")
    best = max(results, key=lambda r: r["fall_step"] if r["fall_step"] >= 0 else 9999)
    worst = min(results, key=lambda r: r["fall_step"] if r["fall_step"] >= 0 else 9999)
    print(f"    Best:  {best['config']} (fall_step={best['fall_step'] if best['fall_step']>=0 else 'NEVER'})")
    print(f"    Worst: {worst['config']} (fall_step={worst['fall_step']})")

    # Check if ProtoMotions defaults are better
    b_result = next(r for r in results if r["config"] == "B_protomotions_defaults")
    a_result = next(r for r in results if r["config"] == "A_current")
    if b_result["fall_step"] > a_result["fall_step"] or (b_result["survived"] and not a_result["survived"]):
        print(f"    >>> ProtoMotions defaults (B) are BETTER than our custom config (A)!")
        print(f"    >>> Contact parameter overrides are HURTING stability!")
    elif a_result["fall_step"] > b_result["fall_step"] or (a_result["survived"] and not b_result["survived"]):
        print(f"    >>> Our custom config (A) is BETTER than ProtoMotions defaults (B)")
        print(f"    >>> Contact parameter overrides are helping")
    else:
        print(f"    >>> Both A and B have similar performance")


if __name__ == "__main__":
    main()
