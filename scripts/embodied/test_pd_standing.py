#!/usr/bin/env python3
"""Test: Can PD-controlled SMPL humanoid stand under gravity?

This test answers the root-cause question: given correct PD actuators,
ground plane, and a reference standing pose, does the humanoid maintain
its height over ~2 seconds?

Key insight from diagnostic: at step 0, ctrl=qpos[7:] so PD error=0.
The root free joint has NO actuator — standing requires ground contact
reaction forces to support the body weight (725.9 N).
"""
import sys
import numpy as np
import os
import tempfile

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco not installed"); sys.exit(1)

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
MJCF = f"{CEPH}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
NPZ = f"{CEPH}/output/embodied_t2m_v4/data/npz/v4_crouch_002.npz"

# ---- Stiffness/damping from unified_pipeline.yaml ----
stiffness = [800.0]*9 + [500.0]*3 + [800.0]*9 + [500.0]*3 + \
            [1000.0]*9 + [500.0]*6 + [500.0]*9 + [300.0]*6 + \
            [500.0]*9 + [300.0]*6
damping = [s/10 for s in stiffness]

# ---- Load reference motion to get a valid standing pose ----
from scipy.spatial.transform import Rotation as sRot

data_npz = np.load(NPZ, allow_pickle=True)
motion = data_npz['motion_135']  # (T, 135)
fps = int(data_npz.get('fps', 30))
T = motion.shape[0]

# Decode motion_135 → SMPL axis-angle
transl = motion[:, :3].copy()
rot6d = motion[:, 3:].reshape(T, 22, 6)

# rot6d → rotmat (Gram-Schmidt)
def rot6d_to_rotmat(r6d):
    """(*, 6) → (*, 3, 3)"""
    shape = r6d.shape[:-1]
    r6d = r6d.reshape(-1, 6)
    # Reorder: row-major → col-major
    r6d = r6d[:, [0, 2, 4, 1, 3, 5]]
    a1 = r6d[:, :3]
    a2 = r6d[:, 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).reshape(*shape, 3, 3)

rotmat = rot6d_to_rotmat(rot6d)  # (T, 22, 3, 3)
aa_all = sRot.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 22, 3).astype(np.float32)

smpl_pose = np.zeros((T, 72), dtype=np.float32)
smpl_pose[:, :3] = aa_all[:, 0]
smpl_pose[:, 3:66] = aa_all[:, 1:].reshape(T, -1)

# Y-up → Z-up
Rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
transl_zup = transl @ Rx.T

R_root = sRot.from_rotvec(smpl_pose[:, :3]).as_matrix()
R_root_zup = Rx[None] @ R_root
root_orient_zup = sRot.from_matrix(R_root_zup).as_rotvec().astype(np.float32)

# ---- Convert SMPL → MuJoCo qpos ----
# Load model for FK and joint ordering
model = mujoco.MjModel.from_xml_path(MJCF)
body_pos_1 = model.body_pos[1].copy()

# SMPL-to-MuJoCo joint reordering: for SMPL humanoid it's identity
ref_qpos = np.zeros((T, 76), dtype=np.float32)
for t in range(T):
    # Root translation
    ref_qpos[t, :3] = transl_zup[t] + body_pos_1
    # Root quaternion (wxyz)
    root_R = sRot.from_rotvec(root_orient_zup[t]).as_matrix()
    root_quat_xyzw = sRot.from_matrix(root_R).as_quat()
    ref_qpos[t, 3:7] = root_quat_xyzw[[3, 0, 1, 2]]  # → wxyz
    # Body joints: axis-angle → ZYX Euler
    body_aa = smpl_pose[t, 3:72].reshape(23, 3)
    body_euler = sRot.from_rotvec(body_aa).as_euler("ZYX").astype(np.float32)
    ref_qpos[t, 7:] = body_euler.flatten()

frame0 = ref_qpos[0]
print(f"Reference frame 0:")
print(f"  root_pos = {frame0[:3]}")
print(f"  root_quat(wxyz) = {frame0[3:7]}")
print(f"  joints[:10] = {frame0[7:17]}")

# ---- Load and configure model ----
# Patch XML for condim
with open(MJCF) as f:
    xml = f.read()
xml = xml.replace('condim="1"', 'condim="3"')
asset_dir = os.path.dirname(MJCF)
with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', dir=asset_dir, delete=False) as f:
    f.write(xml)
    tmp_xml = f.name

model = mujoco.MjModel.from_xml_path(tmp_xml)
os.unlink(tmp_xml)
data = mujoco.MjData(model)

# Physics timestep
model.opt.timestep = 0.001

# Zero passive forces
model.jnt_stiffness[:] = 0.0
model.dof_damping[:] = 0.0
model.dof_frictionloss[:] = 0.0

# Configure PD actuators
for i in range(model.nu):
    kp = stiffness[i]
    kd = damping[i]
    model.actuator_gear[i, 0] = 1.0
    model.actuator_gainprm[i, 0] = kp
    model.actuator_biastype[i] = 1
    model.actuator_biasprm[i, 0] = 0.0
    model.actuator_biasprm[i, 1] = -kp
    model.actuator_biasprm[i, 2] = -kd
    model.actuator_ctrllimited[i] = 0
    model.actuator_forcerange[i, 0] = -500.0
    model.actuator_forcerange[i, 1] = 500.0
    model.actuator_forcelimited[i] = 1

# Set condim=3 for all body geoms
for gid in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    if geom_name != "floor" and model.geom_bodyid[gid] > 0:
        model.geom_condim[gid] = 3

print(f"\nModel: nu={model.nu}, nq={model.nq}, nv={model.nv}")
print(f"Weight: {model.body_mass.sum() * 9.81:.1f} N")
print(f"Timestep: {model.opt.timestep}")
print(f"Solver: type={model.opt.solver}, iterations={model.opt.iterations}")
print(f"Gravity: {model.opt.gravity}")

# =====================================================================
# TEST A: PD-only, ctrl = frame0 joints (static target)
# =====================================================================
print("\n" + "=" * 70)
print("TEST A: PD-only static target (ctrl = frame0 joints)")
print("Humanoid should maintain pose under gravity via ground contact")
print("=" * 70)

data.qpos[:] = frame0
data.qvel[:] = 0.0
data.ctrl[:] = frame0[7:]
mujoco.mj_forward(model, data)

print(f"\nInitial state:")
print(f"  root_h = {data.qpos[2]:.4f}")
print(f"  ncon = {data.ncon}")
for ci in range(min(data.ncon, 8)):
    c = data.contact[ci]
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
    print(f"    {g1} <-> {g2}: dist={c.dist:.6f}, pos={c.pos}")

# Check foot positions (L_Toe=body4, R_Toe=body8)
l_toe_body = None
r_toe_body = None
for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
    if name == "L_Toe": l_toe_body = bid
    if name == "R_Toe": r_toe_body = bid

if l_toe_body and r_toe_body:
    print(f"  L_Toe pos: {data.xpos[l_toe_body]} (z should be near 0)")
    print(f"  R_Toe pos: {data.xpos[r_toe_body]} (z should be near 0)")
    print(f"  L_Ankle pos: {data.xpos[l_toe_body-1]}")
    print(f"  R_Ankle pos: {data.xpos[r_toe_body-1]}")

# Simulate 100 control steps (= 2000 physics steps = 2.0s)
control_dt = 0.02
decimation = 20
n_steps = 100

heights = []
contact_counts = []
act_force_maxes = []
qfrc_act_maxes = []
qfrc_constraint_maxes = []
qfrc_bias_z = []

for step in range(n_steps):
    heights.append(float(data.qpos[2]))

    # Ctrl stays constant (frame0 target)
    data.ctrl[:] = frame0[7:]

    # Step physics
    for _ in range(decimation):
        mujoco.mj_step(model, data)

    # Record stats
    contact_counts.append(data.ncon)
    act_force_maxes.append(float(np.abs(data.actuator_force).max()))
    qfrc_act_maxes.append(float(np.abs(data.qfrc_actuator).max()))
    qfrc_constraint_maxes.append(float(np.abs(data.qfrc_constraint).max()))
    qfrc_bias_z.append(float(data.qfrc_bias[2]))

    if step < 5 or step % 20 == 0 or step == n_steps - 1:
        # Check PD error
        pd_error = data.ctrl - data.qpos[7:]
        joint_drift = np.abs(pd_error).max()
        print(f"  step={step:3d}  h={data.qpos[2]:.4f}  ncon={data.ncon:2d}  "
              f"|act_f|_max={np.abs(data.actuator_force).max():.1f}  "
              f"|qfrc_act|_max={np.abs(data.qfrc_actuator).max():.1f}  "
              f"|qfrc_con|_max={np.abs(data.qfrc_constraint).max():.1f}  "
              f"joint_drift={joint_drift:.4f}")

    if data.qpos[2] < 0.3:
        print(f"  FALL at step {step}, h={data.qpos[2]:.4f}")
        break

print(f"\nHeight trajectory: start={heights[0]:.4f}, end={heights[-1]:.4f}, "
      f"drop={heights[0]-heights[-1]:.4f}")
print(f"Max actuator force: {max(act_force_maxes):.1f}")
print(f"Max qfrc_actuator: {max(qfrc_act_maxes):.1f}")
print(f"Max qfrc_constraint: {max(qfrc_constraint_maxes):.1f}")
print(f"Contact count range: [{min(contact_counts)}, {max(contact_counts)}]")

# =====================================================================
# TEST B: Same as A but with higher PD gains (3x stiffness)
# =====================================================================
print("\n" + "=" * 70)
print("TEST B: Higher PD gains (3x stiffness)")
print("=" * 70)

# Reconfigure with 3x stiffness
for i in range(model.nu):
    kp = stiffness[i] * 3.0
    kd = damping[i] * 3.0
    model.actuator_gainprm[i, 0] = kp
    model.actuator_biasprm[i, 1] = -kp
    model.actuator_biasprm[i, 2] = -kd

# Reset
data.qpos[:] = frame0
data.qvel[:] = 0.0
data.ctrl[:] = frame0[7:]
mujoco.mj_forward(model, data)

for step in range(n_steps):
    data.ctrl[:] = frame0[7:]
    for _ in range(decimation):
        mujoco.mj_step(model, data)

    if step < 5 or step % 20 == 0 or step == n_steps - 1:
        pd_error = data.ctrl - data.qpos[7:]
        print(f"  step={step:3d}  h={data.qpos[2]:.4f}  ncon={data.ncon:2d}  "
              f"|act_f|_max={np.abs(data.actuator_force).max():.1f}  "
              f"joint_drift={np.abs(pd_error).max():.4f}")

    if data.qpos[2] < 0.3:
        print(f"  FALL at step {step}, h={data.qpos[2]:.4f}")
        break

h_end = data.qpos[2]
print(f"Final height: {h_end:.4f}")

# =====================================================================
# TEST C: No force limiting (let PD generate unlimited torque)
# =====================================================================
print("\n" + "=" * 70)
print("TEST C: Original gains, NO force limiting")
print("=" * 70)

# Reconfigure: original gains, no force limit
for i in range(model.nu):
    kp = stiffness[i]
    kd = damping[i]
    model.actuator_gainprm[i, 0] = kp
    model.actuator_biasprm[i, 1] = -kp
    model.actuator_biasprm[i, 2] = -kd
    model.actuator_forcelimited[i] = 0  # NO limit

data.qpos[:] = frame0
data.qvel[:] = 0.0
data.ctrl[:] = frame0[7:]
mujoco.mj_forward(model, data)

for step in range(n_steps):
    data.ctrl[:] = frame0[7:]
    for _ in range(decimation):
        mujoco.mj_step(model, data)

    if step < 5 or step % 20 == 0 or step == n_steps - 1:
        print(f"  step={step:3d}  h={data.qpos[2]:.4f}  ncon={data.ncon:2d}  "
              f"|act_f|_max={np.abs(data.actuator_force).max():.1f}  "
              f"|qfrc_act|_max={np.abs(data.qfrc_actuator).max():.1f}")

    if data.qpos[2] < 0.3:
        print(f"  FALL at step {step}, h={data.qpos[2]:.4f}")
        break
    if np.any(np.isnan(data.qpos)):
        print(f"  NaN at step {step}")
        break

print(f"Final height: {data.qpos[2]:.4f}")

# =====================================================================
# TEST D: Check if reference pose has feet touching ground
# =====================================================================
print("\n" + "=" * 70)
print("TEST D: Reference pose foot-ground analysis")
print("=" * 70)

# Use a clean forward pass to check xpos
data.qpos[:] = frame0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

print("All body positions (Z coordinate = height above ground):")
for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
    z = data.xpos[bid, 2]
    marker = " ← LOW" if z < 0.05 and bid > 0 else ""
    print(f"  {name:15s}: z={z:.4f}{marker}")

# Check geom positions too
print("\nFoot-related geom positions:")
for gid in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
    if any(kw in name.lower() for kw in ["toe", "ankle", "foot", "floor"]):
        print(f"  geom '{name}': pos={data.geom_xpos[gid]}")

print("\nDone!")
