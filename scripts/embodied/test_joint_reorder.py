#!/usr/bin/env python3
"""Diagnose joint reordering: SMPL vs MuJoCo body joint order.

SMPL standard: 0-Pelvis, 1-L_Hip, 2-R_Hip, 3-Spine1, 4-L_Knee, 5-R_Knee, 6-Spine2, ...
MuJoCo depth-first: 0-Pelvis, 1-L_Hip, 2-L_Knee, 3-L_Ankle, 4-L_Toe, 5-R_Hip, ...

These differ! Need correct SMPL_2_MUJOCO mapping.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as sRot

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
MJCF = f"{CEPH}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"

model = mujoco.MjModel.from_xml_path(MJCF)

# Print all bodies
print("MuJoCo body tree (depth-first):")
for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
    parent = model.body_parentid[bid]
    pname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) or f"body{parent}" if bid > 0 else "none"
    print(f"  body {bid:2d}: {name:15s} (parent={pname})")

# Print all joints
print(f"\nMuJoCo joints ({model.njnt} total, {model.nq} qpos):")
for jid in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"j{jid}"
    axis = model.jnt_axis[jid]
    qposadr = model.jnt_qposadr[jid]
    bodyid = model.jnt_bodyid[jid]
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bodyid) or f"body{bodyid}"
    jtype = model.jnt_type[jid]
    tname = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[jtype]
    print(f"  jnt {jid:2d}: {name:18s} type={tname:5s} body={bname:15s} axis={axis} qposadr={qposadr}")

# MuJoCo body joint groups (each body has 3 hinges: _x, _y, _z)
# Excluding root (free joint), group by body
print(f"\nMuJoCo body joint groups (excluding root):")
body_groups = []
for bid in range(2, model.nbody):  # skip world (0) and Pelvis (1, has free joint)
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    # Find joints for this body
    joints = []
    for jid in range(model.njnt):
        if model.jnt_bodyid[jid] == bid and model.jnt_type[jid] == 3:  # hinge
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            joints.append((jid, jname, model.jnt_qposadr[jid]))
    if joints:
        base_name = bname
        first_qpos = joints[0][2]
        mj_idx = (first_qpos - 7) // 3
        body_groups.append((mj_idx, base_name))
        print(f"  MJ slot {mj_idx:2d}: {base_name:15s} qpos[{first_qpos}:{first_qpos+3}]")

# SMPL joint order (standard 24 joints)
SMPL_JOINT_NAMES = [
    "Pelvis",       # 0 - root
    "L_Hip",        # 1
    "R_Hip",        # 2
    "Spine1",       # 3  (Torso in MuJoCo)
    "L_Knee",       # 4
    "R_Knee",       # 5
    "Spine2",       # 6  (Spine in MuJoCo)
    "L_Ankle",      # 7
    "R_Ankle",      # 8
    "Spine3",       # 9  (Chest in MuJoCo)
    "L_Foot",       # 10 (L_Toe in MuJoCo)
    "R_Foot",       # 11 (R_Toe in MuJoCo)
    "Neck",         # 12
    "L_Collar",     # 13 (L_Thorax in MuJoCo)
    "R_Collar",     # 14 (R_Thorax in MuJoCo)
    "Head",         # 15
    "L_Shoulder",   # 16
    "R_Shoulder",   # 17
    "L_Elbow",      # 18
    "R_Elbow",      # 19
    "L_Wrist",      # 20
    "R_Wrist",      # 21
    "L_Hand",       # 22
    "R_Hand",       # 23
]

# SMPL name → MuJoCo name aliases
SMPL_TO_MUJOCO_NAME = {
    "L_Hip": "L_Hip",
    "R_Hip": "R_Hip",
    "Spine1": "Torso",
    "L_Knee": "L_Knee",
    "R_Knee": "R_Knee",
    "Spine2": "Spine",
    "L_Ankle": "L_Ankle",
    "R_Ankle": "R_Ankle",
    "Spine3": "Chest",
    "L_Foot": "L_Toe",
    "R_Foot": "R_Toe",
    "Neck": "Neck",
    "L_Collar": "L_Thorax",
    "R_Collar": "R_Thorax",
    "Head": "Head",
    "L_Shoulder": "L_Shoulder",
    "R_Shoulder": "R_Shoulder",
    "L_Elbow": "L_Elbow",
    "R_Elbow": "R_Elbow",
    "L_Wrist": "L_Wrist",
    "R_Wrist": "R_Wrist",
    "L_Hand": "L_Hand",
    "R_Hand": "R_Hand",
}

# Build MuJoCo name → slot index
mj_name_to_slot = {name: idx for idx, name in body_groups}

# Build SMPL_2_MUJOCO: smpl body joint i → mujoco slot
# SMPL body joints are indices 1-23 (excluding root=0)
# In our code, body joints are indexed 0-22 (23 total)
print(f"\nSMPL → MuJoCo joint mapping:")
smpl_2_mujoco = []
for i in range(23):  # SMPL body joints 0-22 (=SMPL joints 1-23)
    smpl_name = SMPL_JOINT_NAMES[i + 1]  # +1 because root is 0
    mj_name = SMPL_TO_MUJOCO_NAME.get(smpl_name, smpl_name)
    mj_slot = mj_name_to_slot.get(mj_name, -1)
    smpl_2_mujoco.append(mj_slot)
    marker = " ← SAME" if mj_slot == i else f" ← MOVED from {i}"
    print(f"  SMPL body {i:2d} ({smpl_name:15s}) → MJ slot {mj_slot:2d} ({mj_name:15s}){marker}")

print(f"\nSMPL_2_MUJOCO = {smpl_2_mujoco}")

# Verify: is identity mapping correct?
is_identity = all(smpl_2_mujoco[i] == i for i in range(23))
print(f"Is identity mapping: {is_identity}")
if not is_identity:
    mismatches = [(i, smpl_2_mujoco[i]) for i in range(23) if smpl_2_mujoco[i] != i]
    print(f"Mismatches: {len(mismatches)} out of 23")
    for src, dst in mismatches:
        src_name = SMPL_JOINT_NAMES[src + 1]
        print(f"  SMPL {src:2d} ({src_name}) → MJ {dst}")

# Also compute inverse: MUJOCO_2_SMPL
mujoco_2_smpl = [0] * 23
for i, j in enumerate(smpl_2_mujoco):
    if 0 <= j < 23:
        mujoco_2_smpl[j] = i
print(f"\nMUJOCO_2_SMPL = {mujoco_2_smpl}")

# =====================================================================
# Now test with correct mapping
# =====================================================================
print("\n" + "=" * 70)
print("TEST: Correct SMPL_2_MUJOCO mapping with ZYX Euler (no body transform)")
print("=" * 70)

NPZ = f"{CEPH}/output/embodied_t2m_v4/data/npz/v4_crouch_002.npz"
_YUP_TO_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

data_npz = np.load(NPZ, allow_pickle=True)
motion = data_npz['motion_135']
T = motion.shape[0]
transl_yup = motion[:, :3].copy()
rot6d = motion[:, 3:].reshape(T, 22, 6)

def rot6d_to_rotmat(r6d):
    shape = r6d.shape[:-1]
    r6d = r6d.reshape(-1, 6)
    r6d = r6d[:, [0, 2, 4, 1, 3, 5]]
    a1 = r6d[:, :3]; a2 = r6d[:, 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).reshape(*shape, 3, 3)

rotmat = rot6d_to_rotmat(rot6d)
aa_all = sRot.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 22, 3).astype(np.float32)

# Build SMPL pose (Y-up)
smpl_pose = np.zeros((T, 72), dtype=np.float32)
smpl_pose[:, :3] = aa_all[:, 0]
smpl_pose[:, 3:66] = aa_all[:, 1:].reshape(T, -1)

# Transform: Y-up → Z-up (root only)
transl_zup = (transl_yup.astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32)
R_root_yup = sRot.from_rotvec(smpl_pose[:, :3]).as_matrix()
R_root_zup = _YUP_TO_ZUP[None] @ R_root_yup
root_aa_zup = sRot.from_matrix(R_root_zup).as_rotvec().astype(np.float32)

# Body joints: keep Y-up (local rotations don't change with world frame)
body_aa_yup = smpl_pose[:, 3:66].reshape(T, 21, 3)

body_pos_1 = model.body_pos[1].copy()
data = mujoco.MjData(model)

# Build qpos with CORRECT reordering + NO body transform + ZYX Euler
qpos = np.zeros((T, 76), dtype=np.float64)
qpos[:, :3] = transl_zup.astype(np.float64) + body_pos_1
root_quat_xyzw = sRot.from_rotvec(root_aa_zup).as_quat()
qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]

# Body joints with correct reordering
body_euler_smpl = sRot.from_rotvec(body_aa_yup.reshape(-1, 3)).as_euler("ZYX").reshape(T, 21, 3)
# Pad to 23 joints (joints 22, 23 = zeros)
body_euler_full = np.zeros((T, 23, 3), dtype=np.float64)
body_euler_full[:, :21] = body_euler_smpl
# Apply SMPL→MuJoCo reordering
body_euler_mj = body_euler_full[:, smpl_2_mujoco]
qpos[:, 7:] = body_euler_mj.reshape(T, 69)

data.qpos[:] = qpos[0]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

print(f"Pelvis: z={data.xpos[1, 2]:.4f}")
for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
    if name in ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle", "L_Knee", "R_Knee"]:
        print(f"  {name}: z={data.xpos[bid, 2]:.4f}")
all_z = data.xpos[1:, 2]
print(f"Min body Z: {all_z.min():.4f}, Max: {all_z.max():.4f}")
print(f"All above ground: {'YES' if all_z.min() > -0.05 else 'NO'}")
print(f"Feet near ground: {'YES' if all_z.min() < 0.15 else 'NO'}")

# Also test with body transform + correct reordering
print("\n--- With body transform (aa@M.T) + correct reordering ---")
body_aa_zup = (body_aa_yup.reshape(-1, 3).astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32).reshape(T, 21, 3)
body_euler_smpl2 = sRot.from_rotvec(body_aa_zup.reshape(-1, 3)).as_euler("ZYX").reshape(T, 21, 3)
body_euler_full2 = np.zeros((T, 23, 3), dtype=np.float64)
body_euler_full2[:, :21] = body_euler_smpl2
body_euler_mj2 = body_euler_full2[:, smpl_2_mujoco]

qpos2 = qpos.copy()
qpos2[:, 7:] = body_euler_mj2.reshape(T, 69)

data.qpos[:] = qpos2[0]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

print(f"Pelvis: z={data.xpos[1, 2]:.4f}")
for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
    if name in ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle", "L_Knee", "R_Knee"]:
        print(f"  {name}: z={data.xpos[bid, 2]:.4f}")
all_z = data.xpos[1:, 2]
print(f"Min body Z: {all_z.min():.4f}, Max: {all_z.max():.4f}")
print(f"All above ground: {'YES' if all_z.min() > -0.05 else 'NO'}")
print(f"Feet near ground: {'YES' if all_z.min() < 0.15 else 'NO'}")

print("\nDone!")
