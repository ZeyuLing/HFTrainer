#!/usr/bin/env python3
"""Diagnostic: Test if SMPL humanoid is stable at rest in MuJoCo.

Sets the robot to its initial pose, applies ctrl = current dof positions
(zero PD error), and steps physics to see if it maintains the pose.

This tells us if the physics model setup is correct BEFORE we add RL tracking.
If the robot can't even maintain a static pose, the problem is in physics config.
"""

import numpy as np
import mujoco
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Paths
MJCF_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
YAML_PATH = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.yaml"

import yaml
with open(YAML_PATH) as f:
    yaml_meta = yaml.safe_load(f)

stiffness = yaml_meta["control"]["stiffness"]
damping = yaml_meta["control"]["damping"]
physics_dt = 0.001
control_dt = 0.02
decimation = 20


def setup_model(model, stiffness, damping, physics_dt, forcelimited=True, margin=0.02):
    """Configure MuJoCo model matching run_smpl_rl_tracker.py setup."""
    model.opt.timestep = physics_dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # Contact params
    for geom_id in range(model.ngeom):
        model.geom_solref[geom_id, 0] = 0.015
        model.geom_solref[geom_id, 1] = 1.0
        model.geom_solimp[geom_id, 0] = 0.9
        model.geom_solimp[geom_id, 1] = 0.99
        model.geom_solimp[geom_id, 2] = 0.003
        model.geom_margin[geom_id] = margin

    # Zero passive forces
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0

    # Actuator setup
    num_actuators = model.nu
    model.actuator_gear[:, 0] = 1.0

    for i in range(num_actuators):
        kp = stiffness[i]
        kd = damping[i]
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biastype[i] = 1
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kd
        model.actuator_ctrllimited[i] = 0
        if forcelimited:
            model.actuator_forcerange[i, 0] = -500.0
            model.actuator_forcerange[i, 1] = 500.0
            model.actuator_forcelimited[i] = 1
        else:
            model.actuator_forcelimited[i] = 0


def get_foot_contacts(model, data):
    """Count floor contacts with foot geoms."""
    contacts = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
        if "floor" in g1 or "floor" in g2:
            other = g2 if "floor" in g1 else g1
            contacts.append((other, c.dist))
    return contacts


def test_stability(forcelimited=True, margin=0.02, lower_by=0.0, label=""):
    """Test robot stability with given parameters."""
    print(f"\n{'='*70}")
    print(f"  TEST: {label}")
    print(f"  forcelimited={forcelimited}, margin={margin}, lower_by={lower_by}")
    print(f"{'='*70}")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    setup_model(model, stiffness, damping, physics_dt, forcelimited, margin)

    # Set initial pose: T-pose / default pose
    # Use default qpos from model (T-pose with root at origin)
    # Then lower feet to ground using fix_height approach
    data.qpos[:] = model.qpos0.copy()
    data.qvel[:] = 0.0

    # Place robot: set root height so lowest geom touches ground
    mujoco.mj_forward(model, data)

    # Find lowest geom Z
    min_geom_z = float('inf')
    for gid in range(model.ngeom):
        body_id = model.geom_bodyid[gid]
        if body_id == 0:  # Skip world body
            continue
        gtype = int(model.geom_type[gid])
        gxpos = data.geom_xpos[gid]
        gsize = model.geom_size[gid]
        gxmat = data.geom_xmat[gid].reshape(3, 3)

        if gtype == 5:  # capsule
            radius = gsize[0]
            half_len = gsize[1]
            z_extent = abs(gxmat[2, 2]) * half_len + radius
            bottom_z = gxpos[2] - z_extent
        elif gtype == 3:  # sphere
            bottom_z = gxpos[2] - gsize[0]
        elif gtype == 6:  # box
            half_extents = gsize[:3]
            z_extent = sum(abs(gxmat[2, j]) * half_extents[j] for j in range(3))
            bottom_z = gxpos[2] - z_extent
        else:
            bottom_z = gxpos[2]
        min_geom_z = min(min_geom_z, bottom_z)

    # Lower robot so lowest geom is at ground (or slightly penetrating)
    height_correction = -min_geom_z + 0.005 + lower_by  # 5mm above ground baseline
    # Apply negative lower_by to push INTO ground for better contact
    height_correction = -min_geom_z + 0.005 - lower_by
    data.qpos[2] += height_correction

    print(f"  Initial root height: {data.qpos[2]:.4f}m (correction: {height_correction:.4f}m)")
    print(f"  Lowest geom was at Z={min_geom_z:.4f}m")

    # Set ctrl = initial DOF positions (ZERO PD error)
    data.ctrl[:] = data.qpos[7:]

    # Forward to compute contacts
    mujoco.mj_forward(model, data)

    contacts = get_foot_contacts(model, data)
    print(f"  Initial contacts: {data.ncon} total, {len(contacts)} floor contacts")
    for name, dist in contacts[:5]:
        print(f"    {name}: dist={dist:.5f}")

    # Step physics with ZERO PD error (ctrl stays = qpos[7:])
    print(f"\n  Stepping {100} control steps ({100 * control_dt:.1f}s) with ZERO PD error...")
    print(f"  {'step':>6s}  {'root_h':>8s}  {'ncon':>5s}  {'floor_con':>10s}  {'max_force':>10s}  {'qvel_max':>10s}")

    for step in range(100):
        # Keep ctrl = current DOF positions (zero PD error)
        data.ctrl[:] = data.qpos[7:]

        for _ in range(decimation):
            mujoco.mj_step(model, data)

        root_h = data.qpos[2]
        contacts = get_foot_contacts(model, data)
        max_force = np.abs(data.qfrc_actuator).max()
        qvel_max = np.abs(data.qvel).max()

        if step % 10 == 0 or step < 5:
            contact_names = [c[0] for c in contacts[:3]]
            print(f"  {step:6d}  {root_h:8.4f}  {data.ncon:5d}  {len(contacts):10d}  "
                  f"{max_force:10.2f}  {qvel_max:10.4f}")

        if root_h < 0.3:
            print(f"  >>> FELL at step {step}! root_h={root_h:.4f}")
            break

        if np.any(np.isnan(data.qpos)):
            print(f"  >>> NaN at step {step}!")
            break
    else:
        print(f"  >>> STABLE for 100 steps! Final root_h={data.qpos[2]:.4f}")

    print(f"  Final root height: {data.qpos[2]:.4f}m")
    print(f"  Height change: {data.qpos[2] - (model.qpos0[2] + height_correction):.5f}m")

    return data.qpos[2] > 0.3


def test_with_reference_motion():
    """Test stability with the actual reference motion first frame."""
    print(f"\n{'='*70}")
    print(f"  TEST: Stability with ACTUAL reference motion pose")
    print(f"{'='*70}")

    # Load a sample motion to get the first frame qpos
    # Use the walk_forward NPZ if available
    npz_candidates = [
        "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4/data/npz/walk_forward.npz",
        "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4/data/npz/example_walk.npz",
    ]

    npz_path = None
    for p in npz_candidates:
        if os.path.exists(p):
            npz_path = p
            break

    if npz_path is None:
        # Generate a simple standing pose from default
        print("  No reference NPZ found, using default model pose")
        return

    print(f"  Loading reference: {npz_path}")

    # Import conversion functions from run_smpl_rl_tracker
    from run_smpl_rl_tracker import (
        decode_motion_135_array,
        smpl_to_qpos,
        compute_ground_offset,
    )

    npz = np.load(npz_path)
    if "motion" in npz:
        motion_135 = npz["motion"]
    elif "motion_135" in npz:
        motion_135 = npz["motion_135"]
    else:
        print(f"  Keys: {list(npz.keys())}")
        return

    print(f"  Motion shape: {motion_135.shape}")

    # Convert to qpos
    smpl_pose, transl = decode_motion_135_array(motion_135)

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    body_pos_1 = model.body_pos[1].copy()

    qpos = smpl_to_qpos(smpl_pose, transl, body_pos_1, model)

    # Fix height
    offset = compute_ground_offset(model, data, qpos)
    qpos[:, 2] -= offset

    # Setup model
    setup_model(model, stiffness, damping, physics_dt, forcelimited=True, margin=0.02)

    # Set first frame
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0.0
    data.ctrl[:] = qpos[0, 7:]
    mujoco.mj_forward(model, data)

    print(f"  Root height: {data.qpos[2]:.4f}m")
    print(f"  Initial contacts: {data.ncon}")
    contacts = get_foot_contacts(model, data)
    for name, dist in contacts[:5]:
        print(f"    {name}: dist={dist:.5f}")

    # Step with zero PD error
    print(f"\n  Stepping 100 control steps with ZERO PD error...")
    for step in range(100):
        data.ctrl[:] = data.qpos[7:]
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        root_h = data.qpos[2]
        if step % 10 == 0 or step < 5:
            contacts = get_foot_contacts(model, data)
            print(f"  step={step:4d}  root_h={root_h:.4f}  ncon={data.ncon}  "
                  f"floor_contacts={len(contacts)}  qvel_max={np.abs(data.qvel).max():.4f}")

        if root_h < 0.3:
            print(f"  >>> FELL at step {step}!")
            break
    else:
        print(f"  >>> STABLE! Final root_h={data.qpos[2]:.4f}")


def test_margin_as_spring():
    """Test if negative solref creates spring-like behavior within margin."""
    print(f"\n{'='*70}")
    print(f"  TEST: Negative solref (spring contact model)")
    print(f"  This makes MuJoCo contacts behave like IsaacGym soft contacts")
    print(f"{'='*70}")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    setup_model(model, stiffness, damping, physics_dt, forcelimited=True, margin=0.02)

    # Override contact params with negative solref (spring model)
    # In MuJoCo, negative solref[0] = -stiffness, solref[1] = -damping
    # This creates force = impedance * (stiffness * depth + damping * vel)
    # where depth = margin - dist when dist < margin
    for geom_id in range(model.ngeom):
        model.geom_solref[geom_id, 0] = -200.0   # spring stiffness
        model.geom_solref[geom_id, 1] = -50.0    # damping
        model.geom_solimp[geom_id, 0] = 0.9
        model.geom_solimp[geom_id, 1] = 0.99
        model.geom_solimp[geom_id, 2] = 0.01     # wider transition
        model.geom_margin[geom_id] = 0.02

    # Set initial pose with robot slightly above ground
    data.qpos[:] = model.qpos0.copy()
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Find lowest geom and adjust height
    min_geom_z = float('inf')
    for gid in range(model.ngeom):
        body_id = model.geom_bodyid[gid]
        if body_id == 0:
            continue
        gtype = int(model.geom_type[gid])
        gxpos = data.geom_xpos[gid]
        gsize = model.geom_size[gid]
        gxmat = data.geom_xmat[gid].reshape(3, 3)
        if gtype == 5:
            bottom_z = gxpos[2] - abs(gxmat[2, 2]) * gsize[1] - gsize[0]
        elif gtype == 3:
            bottom_z = gxpos[2] - gsize[0]
        else:
            bottom_z = gxpos[2]
        min_geom_z = min(min_geom_z, bottom_z)

    # Place with 1cm gap (inside margin zone → should get spring force)
    height_correction = -min_geom_z + 0.01  # 1cm above ground (inside 2cm margin)
    data.qpos[2] += height_correction
    data.ctrl[:] = data.qpos[7:]
    mujoco.mj_forward(model, data)

    print(f"  Root height: {data.qpos[2]:.4f}m (1cm gap inside margin)")
    print(f"  Initial contacts: {data.ncon}")
    contacts = get_foot_contacts(model, data)
    for name, dist in contacts[:5]:
        print(f"    {name}: dist={dist:.5f}")

    # Step
    print(f"\n  Stepping 100 control steps...")
    for step in range(100):
        data.ctrl[:] = data.qpos[7:]
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        root_h = data.qpos[2]
        if step % 10 == 0 or step < 5:
            contacts = get_foot_contacts(model, data)
            print(f"  step={step:4d}  root_h={root_h:.4f}  ncon={data.ncon}  "
                  f"floor_contacts={len(contacts)}")

        if root_h < 0.3:
            print(f"  >>> FELL at step {step}!")
            break
        if np.any(np.isnan(data.qpos)):
            print(f"  >>> NaN at step {step}!")
            break
    else:
        print(f"  >>> STABLE! Final root_h={data.qpos[2]:.4f}")


if __name__ == "__main__":
    # Test 1: Default pose, force limited, standard margin
    test_stability(forcelimited=True, margin=0.02, lower_by=0.0,
                   label="Force limited + margin=0.02")

    # Test 2: Default pose, NO force limit
    test_stability(forcelimited=False, margin=0.02, lower_by=0.0,
                   label="No force limit + margin=0.02")

    # Test 3: Lower robot by 1cm (more penetration)
    test_stability(forcelimited=True, margin=0.02, lower_by=0.01,
                   label="Force limited + margin=0.02 + lower 1cm")

    # Test 4: Lower robot by 2cm (ensure penetration)
    test_stability(forcelimited=True, margin=0.02, lower_by=0.02,
                   label="Force limited + margin=0.02 + lower 2cm")

    # Test 5: Spring contact model (negative solref)
    test_margin_as_spring()

    # Test 6: With actual reference motion
    test_with_reference_motion()
