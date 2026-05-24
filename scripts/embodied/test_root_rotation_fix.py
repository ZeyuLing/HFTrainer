#!/usr/bin/env python3
"""Root rotation fix: M @ R @ M^T instead of M @ R.

The bug: root rotation Y→Z transform was M @ R_yup, which for near-identity
R_yup gives R_zup ≈ M (a 90° axis-swap rotation), causing the humanoid to
be rotated sideways.

Fix: R_zup = M @ R_yup @ M^T (proper change of basis), which maps I → I.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as sRot

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
MJCF = f"{CEPH}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
NPZ = f"{CEPH}/output/embodied_t2m_v4/data/npz/v4_crouch_002.npz"

M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

SMPL_BONE_ORDER_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
    "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
    "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)
body_pos_1 = model.body_pos[1].copy()

# =====================================================================
# Get body_qposaddr and smpl_2_mujoco (gather direction)
# =====================================================================
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

# =====================================================================
# Load motion data
# =====================================================================
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

smpl_pose_yup = np.zeros((T, 72), dtype=np.float32)
smpl_pose_yup[:, :3] = aa_all[:, 0]
smpl_pose_yup[:, 3:66] = aa_all[:, 1:].reshape(T, -1)

# =====================================================================
# Coordinate transforms
# =====================================================================
transl_zup = (transl_yup.astype(np.float64) @ M.T).astype(np.float32)
print(f"Translation Y-up: {transl_yup[0]}")
print(f"Translation Z-up: {transl_zup[0]}")
print(f"body_pos[1] = {body_pos_1}")

# Root rotation - BUG version: R_zup = M @ R_yup
R_root_yup = sRot.from_rotvec(smpl_pose_yup[:, :3]).as_matrix()
R_root_zup_BUG = M[None] @ R_root_yup
root_aa_zup_BUG = sRot.from_matrix(R_root_zup_BUG).as_rotvec().astype(np.float32)

# Root rotation - FIX version: R_zup = M @ R_yup @ M^T (proper change of basis)
R_root_zup_FIX = M[None] @ R_root_yup @ M.T[None]
root_aa_zup_FIX = sRot.from_matrix(R_root_zup_FIX).as_rotvec().astype(np.float32)

print(f"\nRoot orient Y-up:    {smpl_pose_yup[0, :3]} (angle={np.linalg.norm(smpl_pose_yup[0, :3]):.4f})")
print(f"Root orient Z-up BUG: {root_aa_zup_BUG[0]} (angle={np.linalg.norm(root_aa_zup_BUG[0]):.4f})")
print(f"Root orient Z-up FIX: {root_aa_zup_FIX[0]} (angle={np.linalg.norm(root_aa_zup_FIX[0]):.4f})")

body_aa_yup = smpl_pose_yup[:, 3:72].reshape(T, 23, 3)

# =====================================================================
# FK check helper
# =====================================================================
def check_fk(label, qpos_frame):
    data.qpos[:] = qpos_frame
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    pelvis_z = data.xpos[1, 2]
    all_z = data.xpos[1:, 2]
    foot_info = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name in ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle", "L_Knee", "R_Knee"]:
            foot_info[name] = data.xpos[bid, 2]
    print(f"\n  [{label}]")
    print(f"    Pelvis z={pelvis_z:.4f}")
    for k, v in sorted(foot_info.items()):
        print(f"    {k}: z={v:.4f}")
    print(f"    Min/Max body z: [{all_z.min():.4f}, {all_z.max():.4f}]")
    above = all_z.min() > -0.05
    feet_near = min(foot_info.get("L_Toe", 1), foot_info.get("R_Toe", 1)) < 0.15
    if above and feet_near:
        print(f"    Status: PASS (above ground + feet near ground)")
    elif above:
        print(f"    Status: FAIL (feet floating at z={min(foot_info.get('L_Toe',1), foot_info.get('R_Toe',1)):.4f})")
    else:
        print(f"    Status: FAIL (underground, min_z={all_z.min():.4f})")
    return above and feet_near

def build_qpos(root_aa, body_aa_24, transl, smpl_2_mj, euler_conv="ZYX"):
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

# =====================================================================
# TEST 0: T-pose reference
# =====================================================================
print("\n" + "=" * 70)
print("TEST 0: T-pose (identity root, zero joints, h=0.94)")
print("=" * 70)
zero_qpos = np.zeros(76)
zero_qpos[2] = 0.94
zero_qpos[3] = 1.0
check_fk("T-pose", zero_qpos)

# =====================================================================
# TEST 1: Root-only, BUG version (M @ R)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 1: Root-only (zero body joints), BUG: R_zup = M @ R_yup")
print("=" * 70)
qpos_bug_rootonly = np.zeros(76)
qpos_bug_rootonly[:3] = transl_zup[0].astype(np.float64) + body_pos_1
q_xyzw = sRot.from_rotvec(root_aa_zup_BUG[0:1]).as_quat()[0]
qpos_bug_rootonly[3:7] = q_xyzw[[3, 0, 1, 2]]
check_fk("Root-only BUG", qpos_bug_rootonly)

# =====================================================================
# TEST 2: Root-only, FIX version (M @ R @ M^T)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 2: Root-only (zero body joints), FIX: R_zup = M @ R_yup @ M^T")
print("=" * 70)
qpos_fix_rootonly = np.zeros(76)
qpos_fix_rootonly[:3] = transl_zup[0].astype(np.float64) + body_pos_1
q_xyzw = sRot.from_rotvec(root_aa_zup_FIX[0:1]).as_quat()[0]
qpos_fix_rootonly[3:7] = q_xyzw[[3, 0, 1, 2]]
check_fk("Root-only FIX", qpos_fix_rootonly)

# =====================================================================
# TEST 3: Full pose (root + body joints) with BUG root
# =====================================================================
print("\n" + "=" * 70)
print("TEST 3: Full pose, BUG root (M @ R), body joints Y-up, ZYX euler")
print("=" * 70)
qpos3 = build_qpos(root_aa_zup_BUG, body_aa_yup, transl_zup, smpl_2_mujoco, "ZYX")
check_fk("Full BUG root + body YUP + ZYX", qpos3[0])

# =====================================================================
# TEST 4: Full pose with FIX root + ZYX euler (no body transform)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 4: Full pose, FIX root (M@R@M^T), body Y-up, ZYX euler")
print("=" * 70)
qpos4 = build_qpos(root_aa_zup_FIX, body_aa_yup, transl_zup, smpl_2_mujoco, "ZYX")
check_fk("Full FIX root + body YUP + ZYX", qpos4[0])

# =====================================================================
# TEST 5: Full pose with FIX root + XYZ euler (no body transform)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 5: Full pose, FIX root (M@R@M^T), body Y-up, XYZ euler")
print("=" * 70)
qpos5 = build_qpos(root_aa_zup_FIX, body_aa_yup, transl_zup, smpl_2_mujoco, "XYZ")
check_fk("Full FIX root + body YUP + XYZ", qpos5[0])

# =====================================================================
# TEST 6: Full pose with FIX root + body transform (M@R@M^T)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 6: Full pose, FIX root, body transform (M@R@M^T), ZYX euler")
print("=" * 70)
# Proper change of basis for body joints too
body_R_yup = sRot.from_rotvec(body_aa_yup.reshape(-1, 3)).as_matrix().reshape(T, 23, 3, 3)
body_R_zup = M[None, None] @ body_R_yup @ M.T[None, None]
body_aa_zup = sRot.from_matrix(body_R_zup.reshape(-1, 3, 3)).as_rotvec().reshape(T, 23, 3).astype(np.float32)
qpos6 = build_qpos(root_aa_zup_FIX, body_aa_zup, transl_zup, smpl_2_mujoco, "ZYX")
check_fk("Full FIX root + body ZUP (M@R@M^T) + ZYX", qpos6[0])

# =====================================================================
# TEST 7: Full pose with FIX root + body transform (M@R@M^T) + XYZ
# =====================================================================
print("\n" + "=" * 70)
print("TEST 7: Full pose, FIX root, body transform (M@R@M^T), XYZ euler")
print("=" * 70)
qpos7 = build_qpos(root_aa_zup_FIX, body_aa_zup, transl_zup, smpl_2_mujoco, "XYZ")
check_fk("Full FIX root + body ZUP (M@R@M^T) + XYZ", qpos7[0])

# =====================================================================
# TEST 8: Multi-frame check for best combinations
# =====================================================================
print("\n" + "=" * 70)
print("MULTI-FRAME CHECK: 10 frames")
print("=" * 70)

tests = [
    ("T4: FIX root + body YUP + ZYX", qpos4),
    ("T5: FIX root + body YUP + XYZ", qpos5),
    ("T6: FIX root + body ZUP (M@R@M^T) + ZYX", qpos6),
    ("T7: FIX root + body ZUP (M@R@M^T) + XYZ", qpos7),
]

for label, qpos_arr in tests:
    min_zs = []
    pelvis_zs = []
    ltoe_zs = []
    rtoe_zs = []
    for t in range(min(10, T)):
        data.qpos[:] = qpos_arr[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        min_zs.append(data.xpos[1:, 2].min())
        pelvis_zs.append(data.xpos[1, 2])
        ltoe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_Toe")
        rtoe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_Toe")
        ltoe_zs.append(data.xpos[ltoe_bid, 2])
        rtoe_zs.append(data.xpos[rtoe_bid, 2])
    print(f"\n  {label}")
    print(f"    Pelvis z: mean={np.mean(pelvis_zs):.4f}, range=[{np.min(pelvis_zs):.4f}, {np.max(pelvis_zs):.4f}]")
    print(f"    L_Toe z:  mean={np.mean(ltoe_zs):.4f}, range=[{np.min(ltoe_zs):.4f}, {np.max(ltoe_zs):.4f}]")
    print(f"    R_Toe z:  mean={np.mean(rtoe_zs):.4f}, range=[{np.min(rtoe_zs):.4f}, {np.max(rtoe_zs):.4f}]")
    print(f"    Min body z: mean={np.mean(min_zs):.4f}, range=[{np.min(min_zs):.4f}, {np.max(min_zs):.4f}]")
    above = all(z > -0.05 for z in min_zs)
    near_ground = np.mean(min_zs) < 0.15
    print(f"    All above ground: {'YES' if above else 'NO'}, Avg feet near ground: {'YES' if near_ground else 'NO'}")

print("\nDone!")
