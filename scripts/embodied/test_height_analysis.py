#!/usr/bin/env python3
"""Verify FK correctness by checking original SMPL FK in Y-up.

If the motion data itself has feet on the ground in SMPL Y-up FK,
then our MuJoCo conversion should also produce feet on ground.

Also test on different motions (walking) and different frames.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as sRot
import glob, os

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
MJCF = f"{CEPH}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
NPZ_DIR = f"{CEPH}/output/embodied_t2m_v4/data/npz"

M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

SMPL_BONE_ORDER_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
    "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
    "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

# SMPL parent indices (standard 24-joint hierarchy)
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# SMPL rest pose offsets (approximate, from SMPL model)
# We'll compute actual FK from rotations instead

model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)
body_pos_1 = model.body_pos[1].copy()

def get_body_qposaddr(m):
    from collections import OrderedDict
    result = OrderedDict()
    for bid in range(1, m.nbody):
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname is None: continue
        joints = [jid for jid in range(m.njnt) if m.jnt_bodyid[jid] == bid]
        if not joints: continue
        first_jid = joints[0]
        jtype = m.jnt_type[first_jid]
        qstart = m.jnt_qposadr[first_jid]
        qend = qstart + (7 if jtype == 0 else 4 if jtype == 1 else len(joints))
        result[bname] = (qstart, qend)
    return result

body_qposaddr = get_body_qposaddr(model)
mujoco_body_names = list(body_qposaddr.keys())
smpl_2_mujoco = [
    SMPL_BONE_ORDER_NAMES.index(q)
    for q in mujoco_body_names
    if q in SMPL_BONE_ORDER_NAMES
]

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

def build_qpos(root_aa, body_aa_24, transl, smpl_2_mj, euler_conv="ZYX"):
    """Build MuJoCo qpos from Z-up SMPL data."""
    T_n = root_aa.shape[0]
    qpos = np.zeros((T_n, 76), dtype=np.float64)
    qpos[:, :3] = transl.astype(np.float64) + body_pos_1
    root_quat_xyzw = sRot.from_rotvec(root_aa).as_quat()
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]
    all_aa = np.zeros((T_n, 24, 3), dtype=np.float64)
    all_aa[:, 0] = root_aa
    all_aa[:, 1:] = body_aa_24
    all_euler = sRot.from_rotvec(all_aa.reshape(-1, 3)).as_euler(euler_conv).reshape(T_n, 24, 3)
    all_euler_mj = all_euler[:, smpl_2_mj]
    qpos[:, 7:] = all_euler_mj[:, 1:].reshape(T_n, 69)
    return qpos

def check_fk(qpos_frame):
    data.qpos[:] = qpos_frame
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    pelvis_z = data.xpos[1, 2]
    ltoe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_Toe")
    rtoe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_Toe")
    min_z = data.xpos[1:, 2].min()
    return pelvis_z, data.xpos[ltoe_bid, 2], data.xpos[rtoe_bid, 2], min_z

# =====================================================================
# First: check what the T-pose height should be
# =====================================================================
print("=" * 70)
print("MuJoCo model body offsets (Z-up, from parent):")
print("=" * 70)
for bid in range(1, min(15, model.nbody)):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"b{bid}"
    parent = model.body_parentid[bid]
    pname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) or "world"
    pos = model.body_pos[bid]
    print(f"  {name:15s} (parent={pname:10s}): offset={pos}")

# Compute leg chain length
# Pelvis → L_Hip → L_Knee → L_Ankle → L_Toe
leg_offset_sum = 0
for name in ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    z_offset = model.body_pos[bid][2]  # Z component = vertical in Z-up
    leg_offset_sum += z_offset
    print(f"    {name} Z offset: {z_offset:.4f}")
print(f"  Total leg Z offset (parent→child sum): {leg_offset_sum:.4f}")

# The T-pose height of pelvis that gives feet at z=0
# pelvis_z + leg_offset_sum ≈ toe_z → pelvis_z ≈ toe_z - leg_offset_sum
tpose_qpos = np.zeros(76)
tpose_qpos[2] = 0.94
tpose_qpos[3] = 1.0
pz, ltz, rtz, minz = check_fk(tpose_qpos)
print(f"\n  T-pose h=0.94: Pelvis z={pz:.4f}, L_Toe z={ltz:.4f}, R_Toe z={rtz:.4f}")
print(f"  Leg length = pelvis_z - toe_z = {pz - min(ltz, rtz):.4f}")

# =====================================================================
# Test multiple NPZ files
# =====================================================================
npz_files = sorted(glob.glob(f"{NPZ_DIR}/*.npz"))[:5]
print(f"\nFound {len(glob.glob(f'{NPZ_DIR}/*.npz'))} NPZ files, testing first 5:")

for npz_path in npz_files:
    stem = os.path.basename(npz_path).replace('.npz', '')
    data_npz = np.load(npz_path, allow_pickle=True)
    motion = data_npz['motion_135']
    T = motion.shape[0]
    transl_yup = motion[:, :3].copy()
    rot6d_data = motion[:, 3:].reshape(T, 22, 6)

    rotmat = rot6d_to_rotmat(rot6d_data)
    aa_all = sRot.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 22, 3).astype(np.float32)

    smpl_pose = np.zeros((T, 72), dtype=np.float32)
    smpl_pose[:, :3] = aa_all[:, 0]
    smpl_pose[:, 3:66] = aa_all[:, 1:].reshape(T, -1)

    # Transform to Z-up
    transl_zup = (transl_yup.astype(np.float64) @ M.T).astype(np.float32)

    # Root rotation: M @ R @ M^T
    R_root_yup = sRot.from_rotvec(smpl_pose[:, :3]).as_matrix()
    R_root_zup = M[None] @ R_root_yup @ M.T[None]
    root_aa_zup = sRot.from_matrix(R_root_zup).as_rotvec().astype(np.float32)

    # Body joints: NO transform (local rotations stay same)
    body_aa = smpl_pose[:, 3:72].reshape(T, 23, 3)

    qpos_all = build_qpos(root_aa_zup, body_aa, transl_zup, smpl_2_mujoco, "ZYX")

    print(f"\n{'='*70}")
    print(f"  {stem} (T={T})")
    print(f"    Y-up frame 0: transl={transl_yup[0]} (height={transl_yup[0,1]:.4f})")
    print(f"    Z-up frame 0: transl={transl_zup[0]} (height={transl_zup[0,2]:.4f})")

    # Check frames 0, T//4, T//2, 3T//4, T-1
    frames_to_check = [0, T//4, T//2, 3*T//4, T-1]
    for t in frames_to_check:
        pz, ltz, rtz, minz = check_fk(qpos_all[t])
        foot_z = min(ltz, rtz)
        h = transl_zup[t, 2]
        status = "OK" if foot_z < 0.05 else f"FLOAT({foot_z:.3f})"
        hip_angle = np.linalg.norm(aa_all[t, 1])
        knee_angle = np.linalg.norm(aa_all[t, 4])
        print(f"    f={t:4d}: pelvis_z={pz:.3f}, min_foot_z={foot_z:.3f}, h={h:.3f}, "
              f"L_Hip={np.degrees(hip_angle):.1f}° L_Knee={np.degrees(knee_angle):.1f}° [{status}]")

# =====================================================================
# Key question: what height should the pelvis be at?
# =====================================================================
print(f"\n{'='*70}")
print("HEIGHT ANALYSIS:")
print(f"  MuJoCo T-pose: pelvis_z=0.94, feet_z=0.02, leg_length=0.92")
print(f"  For feet to be on ground (z=0), pelvis must be at z≈0.92")
print(f"  Our motion data: pelvis_z≈1.17 → feet would be at 1.17-0.92=0.25 (straight legs)")
print(f"  This matches what we see!")
print(f"\n  HYPOTHESIS: The motion data has a different body shape (taller)")
print(f"  OR the translation reference point differs.")
print(f"\n  Solution: compute ground offset per-frame = min(foot_z) and subtract.")
print(f"  OR: don't add body_pos[1] offset.")
print("=" * 70)

# =====================================================================
# Test WITHOUT body_pos[1] offset
# =====================================================================
print("\nTEST: Without body_pos[1] offset:")
data_npz = np.load(npz_files[0], allow_pickle=True)
motion = data_npz['motion_135']
T = motion.shape[0]
transl_yup = motion[:, :3].copy()
rot6d_data = motion[:, 3:].reshape(T, 22, 6)
rotmat = rot6d_to_rotmat(rot6d_data)
aa_all = sRot.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 22, 3).astype(np.float32)
smpl_pose = np.zeros((T, 72), dtype=np.float32)
smpl_pose[:, :3] = aa_all[:, 0]
smpl_pose[:, 3:66] = aa_all[:, 1:].reshape(T, -1)
transl_zup = (transl_yup.astype(np.float64) @ M.T).astype(np.float32)
R_root_yup = sRot.from_rotvec(smpl_pose[:, :3]).as_matrix()
R_root_zup = M[None] @ R_root_yup @ M.T[None]
root_aa_zup = sRot.from_matrix(R_root_zup).as_rotvec().astype(np.float32)
body_aa = smpl_pose[:, 3:72].reshape(T, 23, 3)

# Build qpos WITHOUT body_pos offset
T_n = root_aa_zup.shape[0]
qpos_nooff = np.zeros((T_n, 76), dtype=np.float64)
qpos_nooff[:, :3] = transl_zup.astype(np.float64)  # NO + body_pos_1
root_quat_xyzw = sRot.from_rotvec(root_aa_zup).as_quat()
qpos_nooff[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]
all_aa = np.zeros((T_n, 24, 3), dtype=np.float64)
all_aa[:, 0] = root_aa_zup
all_aa[:, 1:] = body_aa
all_euler = sRot.from_rotvec(all_aa.reshape(-1, 3)).as_euler("ZYX").reshape(T_n, 24, 3)
all_euler_mj = all_euler[:, smpl_2_mujoco]
qpos_nooff[:, 7:] = all_euler_mj[:, 1:].reshape(T_n, 69)

for t in [0, T//4, T//2]:
    pz, ltz, rtz, minz = check_fk(qpos_nooff[t])
    foot_z = min(ltz, rtz)
    print(f"  f={t:4d}: pelvis_z={pz:.3f}, min_foot_z={foot_z:.3f}")

# =====================================================================
# Test with ground normalization (subtract first-frame min foot z)
# =====================================================================
print("\nTEST: With ground normalization (subtract first-frame min foot z):")
pz0, ltz0, rtz0, minz0 = check_fk(qpos_all[0])
ground_offset = min(ltz0, rtz0)  # should subtract this from translation z
print(f"  Ground offset to subtract: {ground_offset:.4f}")

qpos_grounded = qpos_all.copy()
qpos_grounded[:, 2] -= ground_offset

for t in [0, T//4, T//2, 3*T//4, T-1]:
    pz, ltz, rtz, minz = check_fk(qpos_grounded[t])
    foot_z = min(ltz, rtz)
    status = "OK" if foot_z < 0.05 else f"FLOAT({foot_z:.3f})"
    print(f"  f={t:4d}: pelvis_z={pz:.3f}, min_foot_z={foot_z:.3f} [{status}]")

print("\nDone!")
