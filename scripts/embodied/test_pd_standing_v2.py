#!/usr/bin/env python3
"""Test PD-controlled SMPL humanoid standing — uses corrected transforms.

Uses:
- Correct Y-up → Z-up transform: _YUP_TO_ZUP = [[0,0,1],[1,0,0],[0,1,0]]
- "ZYX" Euler convention (matching smpl_mujoco.py / RL training convention)
- PD actuators with force limiting (±500 N·m)
"""
import sys
import numpy as np
import os
import tempfile

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco not installed"); sys.exit(1)

from scipy.spatial.transform import Rotation as sRot

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
MJCF = f"{CEPH}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
NPZ = f"{CEPH}/output/embodied_t2m_v4/data/npz/v4_crouch_002.npz"

# ---- Coordinate transform ----
_YUP_TO_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

# ---- PD gains (from unified_pipeline.yaml) ----
stiffness = [800.0]*9 + [500.0]*3 + [800.0]*9 + [500.0]*3 + \
            [1000.0]*9 + [500.0]*6 + [500.0]*9 + [300.0]*6 + \
            [500.0]*9 + [300.0]*6
damping = [s/10 for s in stiffness]

# ---- Load and decode motion ----
data_npz = np.load(NPZ, allow_pickle=True)
motion = data_npz['motion_135']
fps = int(data_npz.get('fps', 30))
T = motion.shape[0]

transl_yup = motion[:, :3].copy()
rot6d = motion[:, 3:].reshape(T, 22, 6)

def rot6d_to_rotmat(r6d):
    shape = r6d.shape[:-1]
    r6d = r6d.reshape(-1, 6)
    r6d = r6d[:, [0, 2, 4, 1, 3, 5]]
    a1 = r6d[:, :3]
    a2 = r6d[:, 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).reshape(*shape, 3, 3)

rotmat = rot6d_to_rotmat(rot6d)
aa_all = sRot.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 22, 3).astype(np.float32)

smpl_pose = np.zeros((T, 72), dtype=np.float32)
smpl_pose[:, :3] = aa_all[:, 0]
smpl_pose[:, 3:66] = aa_all[:, 1:].reshape(T, -1)

print(f"Motion: T={T}, fps={fps}")
print(f"Raw transl Y-up frame 0: x={transl_yup[0,0]:.4f}, y(height)={transl_yup[0,1]:.4f}, z={transl_yup[0,2]:.4f}")

# ---- Y-up → Z-up using main script's transform ----
transl_zup = (transl_yup.astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32)
# Transform root orient: R_zup = YUP_TO_ZUP @ R_yup
R_root_yup = sRot.from_rotvec(smpl_pose[:, :3]).as_matrix()
R_root_zup = _YUP_TO_ZUP[None] @ R_root_yup
root_orient_zup = sRot.from_matrix(R_root_zup).as_rotvec().astype(np.float32)
# Transform body joint axis-angles
body_aa_yup = smpl_pose[:, 3:66].reshape(T * 21, 3)
body_aa_zup = (body_aa_yup.astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32)
smpl_pose[:, :3] = root_orient_zup
smpl_pose[:, 3:66] = body_aa_zup.reshape(T, -1)
# joints 22-23 stay zero

print(f"Transl Z-up frame 0: x={transl_zup[0,0]:.4f}, y={transl_zup[0,1]:.4f}, z(height)={transl_zup[0,2]:.4f}")

# ---- SMPL → MuJoCo qpos with "ZYX" Euler ----
model_orig = mujoco.MjModel.from_xml_path(MJCF)
body_pos_1 = model_orig.body_pos[1].copy()
print(f"body_pos[1] (Pelvis offset): {body_pos_1}")

# SMPL_2_MUJOCO = identity for this model
ref_qpos = np.zeros((T, 76), dtype=np.float64)
joint_aa = smpl_pose.reshape(T, 24, 3)
ref_qpos[:, :3] = transl_zup.astype(np.float64) + body_pos_1
root_quat_xyzw = sRot.from_rotvec(joint_aa[:, 0]).as_quat()
ref_qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]  # -> wxyz

body_aa_all = joint_aa[:, 1:].reshape(-1, 3)
body_euler = sRot.from_rotvec(body_aa_all).as_euler("ZYX")
body_euler = body_euler.reshape(T, 23, 3)
ref_qpos[:, 7:] = body_euler.reshape(T, 69)

frame0 = ref_qpos[0].copy()
print(f"\nFrame 0 qpos:")
print(f"  root_pos = {frame0[:3]} (z should be ~1.15)")
print(f"  root_quat(wxyz) = {frame0[3:7]}")
print(f"  joints[:10] = {frame0[7:17]}")

# ---- Load model for simulation ----
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

model.opt.timestep = 0.001
model.jnt_stiffness[:] = 0.0
model.dof_damping[:] = 0.0
model.dof_frictionloss[:] = 0.0

# Configure PD actuators with force limiting
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

for gid in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    if geom_name != "floor" and model.geom_bodyid[gid] > 0:
        model.geom_condim[gid] = 3

print(f"\nModel: nu={model.nu}, nq={model.nq}, nv={model.nv}")
print(f"Weight: {model.body_mass.sum() * 9.81:.1f} N")
print(f"Timestep: {model.opt.timestep}")

# ---- FK check: initial pose ----
data.qpos[:] = frame0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

print(f"\nFK check (frame 0):")
print(f"  Pelvis: {data.xpos[1]} (z={data.xpos[1,2]:.4f})")
for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
    if name in ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]:
        print(f"  {name}: {data.xpos[bid]} (z={data.xpos[bid,2]:.4f})")
all_z = data.xpos[1:, 2]
print(f"  Min body Z: {all_z.min():.4f}, Max: {all_z.max():.4f}")
print(f"  All above ground: {'YES' if all_z.min() > -0.05 else 'NO — UNDERGROUND'}")
print(f"  Ground contacts: {data.ncon}")

# ---- Simulate: PD tracking frame0 for 2 seconds ----
print("\n" + "=" * 70)
print("PD Standing Test: ctrl = frame0 joints, 2 seconds")
print("=" * 70)

data.qpos[:] = frame0
data.qvel[:] = 0.0
data.ctrl[:] = frame0[7:]
mujoco.mj_forward(model, data)

decimation = 20  # 0.02s control dt
n_steps = 100    # 2.0s total

heights = []
for step in range(n_steps):
    heights.append(float(data.qpos[2]))
    data.ctrl[:] = frame0[7:]
    for _ in range(decimation):
        mujoco.mj_step(model, data)

    if step < 5 or step % 20 == 0 or step == n_steps - 1:
        pd_error = data.ctrl - data.qpos[7:]
        print(f"  step={step:3d}  h={data.qpos[2]:.4f}  ncon={data.ncon:2d}  "
              f"|act_f|={np.abs(data.actuator_force).max():.1f}  "
              f"drift={np.abs(pd_error).max():.4f}")

    if data.qpos[2] < 0.3:
        print(f"  FALL at step {step}, h={data.qpos[2]:.4f}")
        break

print(f"\nResult: start_h={heights[0]:.4f}, end_h={heights[-1]:.4f}, drop={heights[0]-heights[-1]:.4f}")

# ---- Simulate: PD tracking first 60 frames of motion ----
print("\n" + "=" * 70)
print("PD Motion Tracking: ctrl = ref_qpos[t], first 60 frames")
print("=" * 70)

data.qpos[:] = ref_qpos[0]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

n_track = min(60, T)
for t in range(n_track):
    data.ctrl[:] = ref_qpos[t, 7:]
    for _ in range(decimation):
        mujoco.mj_step(model, data)

    if t < 5 or t % 10 == 0 or t == n_track - 1:
        tracking_err = np.abs(data.qpos[7:] - ref_qpos[t, 7:]).mean()
        print(f"  frame={t:3d}  h={data.qpos[2]:.4f}  ncon={data.ncon:2d}  "
              f"track_err={tracking_err:.4f}  |act_f|={np.abs(data.actuator_force).max():.1f}")

    if data.qpos[2] < 0.3:
        print(f"  FALL at frame {t}, h={data.qpos[2]:.4f}")
        break

print(f"\nFinal height: {data.qpos[2]:.4f}")
print("Done!")
