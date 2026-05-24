#!/usr/bin/env python3
"""Definitive FK test: reproduce reference smpl_mujoco.py smpl_to_qpose() exactly.

Key insight from reference code analysis:
  smpl_2_mujoco = [
      joint_names.index(q) for q in get_body_qposaddr(mj_model).keys()
  ]
  # smpl_2_mujoco[mj_slot] = smpl_joint_index  (GATHER direction)
  # Used as: euler_smpl[:, smpl_2_mujoco]  → reorders SMPL to MuJoCo

The previous test_joint_reorder.py had the INVERSE:
  smpl_2_mujoco[smpl_idx] = mj_slot  (SCATTER direction)
  But used it as gather → completely wrong reorder!

This test:
1. Computes smpl_2_mujoco exactly like reference code (gather direction)
2. Tests NO body transform (reference code expects Z-up input, does no transform)
3. Tests with body transform (our pipeline transforms body joints)
4. Uses ZYX Euler (matching reference)
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as sRot

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
MJCF = f"{CEPH}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
NPZ = f"{CEPH}/output/embodied_t2m_v4/data/npz/v4_crouch_002.npz"

_YUP_TO_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

# =====================================================================
# SMPL bone names (from ref_repo/OmniH2O/phc/phc/smpllib/smpl_parser.py)
# These use MuJoCo-style names (Torso, Spine, Chest, L_Thorax, R_Thorax)
# =====================================================================
SMPL_BONE_ORDER_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
    "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
    "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

# Load model
model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)
body_pos_1 = model.body_pos[1].copy()

print(f"Model: nbody={model.nbody}, njnt={model.njnt}, nq={model.nq}, nu={model.nu}")
print(f"body_pos[1] = {body_pos_1}")

# =====================================================================
# Compute get_body_qposaddr equivalent
# =====================================================================
def get_body_qposaddr(m):
    """Replicate uhc.khrylib.utils.get_body_qposaddr.
    Returns OrderedDict: body_name → (qpos_start, qpos_end)
    Skips world body (0) and bodies with no joints.
    """
    from collections import OrderedDict
    result = OrderedDict()
    for bid in range(1, m.nbody):  # skip world
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname is None:
            continue
        # Find joints for this body
        joints = []
        for jid in range(m.njnt):
            if m.jnt_bodyid[jid] == bid:
                joints.append(jid)
        if not joints:
            continue
        first_jid = joints[0]
        jtype = m.jnt_type[first_jid]
        qstart = m.jnt_qposadr[first_jid]
        if jtype == 0:  # free joint
            qend = qstart + 7
        elif jtype == 1:  # ball joint
            qend = qstart + 4
        else:  # hinge joints (3 per body = ball equivalent)
            qend = qstart + len(joints)
        result[bname] = (qstart, qend)
    return result

body_qposaddr = get_body_qposaddr(model)
print(f"\nMuJoCo body qpos addresses (depth-first order):")
for name, (qs, qe) in body_qposaddr.items():
    print(f"  {name:15s}: qpos[{qs}:{qe}]")

# =====================================================================
# Compute smpl_2_mujoco EXACTLY like reference code
# =====================================================================
mujoco_body_names = list(body_qposaddr.keys())
print(f"\nMuJoCo body order: {mujoco_body_names}")
print(f"SMPL bone order:   {SMPL_BONE_ORDER_NAMES}")

# Reference code: smpl_2_mujoco[mj_slot] = joint_names.index(mj_body_name)
# This is a GATHER index: euler_smpl[:, smpl_2_mujoco] reorders SMPL→MuJoCo
smpl_2_mujoco = [
    SMPL_BONE_ORDER_NAMES.index(q)
    for q in mujoco_body_names
    if q in SMPL_BONE_ORDER_NAMES
]

print(f"\nsmpl_2_mujoco (gather direction, len={len(smpl_2_mujoco)}):")
print(f"  {smpl_2_mujoco}")
print(f"\nMapping detail (MuJoCo slot → SMPL source index → name):")
for mj_slot, smpl_idx in enumerate(smpl_2_mujoco):
    mj_name = mujoco_body_names[mj_slot]
    smpl_name = SMPL_BONE_ORDER_NAMES[smpl_idx]
    match = "✓" if mj_name == smpl_name else "✗ NAME MISMATCH"
    print(f"  MJ slot {mj_slot:2d} ({mj_name:15s}) ← SMPL idx {smpl_idx:2d} ({smpl_name:15s}) {match}")

# Verify: smpl_2_mujoco includes root (Pelvis)?
# Reference code includes Pelvis but then only uses [3:] for body joints in qpos
print(f"\nFirst entry: smpl_2_mujoco[0] = {smpl_2_mujoco[0]} ({SMPL_BONE_ORDER_NAMES[smpl_2_mujoco[0]]})")
print(f"This is the root (Pelvis) — handled separately as quaternion, body joints start at index 1")

# Body-only mapping (skip root)
smpl_2_mujoco_body = smpl_2_mujoco[1:]  # skip Pelvis
print(f"\nsmpl_2_mujoco_body (23 body joints, gather direction):")
print(f"  {smpl_2_mujoco_body}")

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

# Build SMPL 72-dim pose (Y-up)
smpl_pose_yup = np.zeros((T, 72), dtype=np.float32)
smpl_pose_yup[:, :3] = aa_all[:, 0]          # root orient (3)
smpl_pose_yup[:, 3:66] = aa_all[:, 1:].reshape(T, -1)  # 21 body joints (63)
# joints 22-23 (indices 66:72) stay zero = L_Hand, R_Hand

print(f"\nMotion: T={T}")
print(f"Raw Y-up: x={transl_yup[0,0]:.4f}, y(h)={transl_yup[0,1]:.4f}, z={transl_yup[0,2]:.4f}")

# =====================================================================
# Y-up → Z-up transform
# =====================================================================
transl_zup = (transl_yup.astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32)
R_root_yup = sRot.from_rotvec(smpl_pose_yup[:, :3]).as_matrix()
R_root_zup = _YUP_TO_ZUP[None] @ R_root_yup
root_aa_zup = sRot.from_matrix(R_root_zup).as_rotvec().astype(np.float32)

body_aa_yup = smpl_pose_yup[:, 3:72].reshape(T, 23, 3)  # 23 joints (21 data + 2 zeros)

print(f"Z-up: x={transl_zup[0,0]:.4f}, y={transl_zup[0,1]:.4f}, z(h)={transl_zup[0,2]:.4f}")

# =====================================================================
# Helper: build qpos and check FK
# =====================================================================
def smpl_to_qpos_ref(root_aa, body_aa_24, transl, smpl_2_mj):
    """Reproduce reference smpl_to_qpose() exactly.

    root_aa: (T, 3) axis-angle
    body_aa_24: (T, 23, 3) axis-angle for 23 body joints (SMPL order)
    transl: (T, 3)
    smpl_2_mj: gather indices (24 entries including root)
    """
    T = root_aa.shape[0]
    qpos = np.zeros((T, 76), dtype=np.float64)

    # Translation + body_pos offset
    qpos[:, :3] = transl.astype(np.float64) + body_pos_1

    # Root quaternion (wxyz)
    root_quat_xyzw = sRot.from_rotvec(root_aa).as_quat()
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]

    # ALL 24 joints as euler (including root, which we'll discard)
    all_aa = np.zeros((T, 24, 3), dtype=np.float64)
    all_aa[:, 0] = root_aa
    all_aa[:, 1:] = body_aa_24

    all_euler = sRot.from_rotvec(all_aa.reshape(-1, 3)).as_euler("ZYX").reshape(T, 24, 3)

    # Reorder: SMPL order → MuJoCo order (gather)
    all_euler_mj = all_euler[:, smpl_2_mj]

    # Body joints only (skip root at position 0)
    # Reference: curr_qpos = concat(trans, root_quat, curr_spose[:, 3:])
    # curr_spose[:, 3:] = all_euler_mj flattened, skip first 3 (root euler)
    qpos[:, 7:] = all_euler_mj[:, 1:].reshape(T, 69)

    return qpos

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
    above = all_z.min() > -0.05
    feet_near = all((z < 0.15 for z in foot_info.values())) if foot_info else False
    print(f"\n  [{label}]")
    print(f"    Pelvis z={pelvis_z:.4f}")
    for k, v in sorted(foot_info.items()):
        print(f"    {k}: z={v:.4f}")
    print(f"    Min/Max body z: [{all_z.min():.4f}, {all_z.max():.4f}]")
    status = ""
    if above and feet_near:
        status = "PASS ✓ (above ground + feet on ground)"
    elif above and not feet_near:
        status = "FAIL (feet floating)"
    elif not above:
        status = "FAIL (underground)"
    print(f"    Status: {status}")
    return feet_near

# =====================================================================
# TEST 0: T-pose sanity check
# =====================================================================
print("\n" + "=" * 70)
print("TEST 0: T-pose (zero joints)")
print("=" * 70)
zero_qpos = np.zeros(76)
zero_qpos[2] = 0.94
zero_qpos[3] = 1.0  # w=1 identity quaternion
check_fk("T-pose, h=0.94", zero_qpos)

# =====================================================================
# TEST 1: Reference-style conversion (NO body transform, correct gather reorder)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 1: NO body transform + correct gather reorder + ZYX euler")
print("  (Body joints stay in Y-up = original SMPL local rotations)")
print("=" * 70)
qpos1 = smpl_to_qpos_ref(root_aa_zup, body_aa_yup, transl_zup, smpl_2_mujoco)
check_fk("No body transform + correct reorder", qpos1[0])

# =====================================================================
# TEST 2: With body transform (aa @ M.T) + correct gather reorder
# =====================================================================
print("\n" + "=" * 70)
print("TEST 2: Body transform (aa @ M.T) + correct gather reorder + ZYX euler")
print("  (Body joints transformed Y-up → Z-up via axis-angle rotation)")
print("=" * 70)
body_aa_zup = (body_aa_yup.reshape(-1, 3).astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32).reshape(T, 23, 3)
qpos2 = smpl_to_qpos_ref(root_aa_zup, body_aa_zup, transl_zup, smpl_2_mujoco)
check_fk("Body transform + correct reorder", qpos2[0])

# =====================================================================
# TEST 3: Identity mapping (for comparison — the OLD wrong behavior)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 3: NO body transform + IDENTITY mapping (OLD wrong behavior)")
print("=" * 70)
identity_mapping = list(range(24))
qpos3 = smpl_to_qpos_ref(root_aa_zup, body_aa_yup, transl_zup, identity_mapping)
check_fk("No body transform + identity mapping", qpos3[0])

# =====================================================================
# TEST 4: With body transform + identity mapping (current run_smpl_rl_tracker.py)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 4: Body transform + IDENTITY mapping (current run_smpl_rl_tracker.py)")
print("=" * 70)
qpos4 = smpl_to_qpos_ref(root_aa_zup, body_aa_zup, transl_zup, identity_mapping)
check_fk("Body transform + identity mapping", qpos4[0])

# =====================================================================
# TEST 5: Previous test's WRONG scatter mapping (for comparison)
# =====================================================================
print("\n" + "=" * 70)
print("TEST 5: Previous test's scatter mapping used as gather (WRONG)")
print("=" * 70)
# This is what test_joint_reorder.py computed:
# smpl_2_mujoco_scatter[smpl_body_idx] = mj_slot
# But used it as gather → scrambled results
# Let me compute the inverse of smpl_2_mujoco to get the scatter version
scatter_mapping = [0] * 24
for mj_slot, smpl_idx in enumerate(smpl_2_mujoco):
    scatter_mapping[smpl_idx] = mj_slot
print(f"  scatter_mapping = {scatter_mapping}")
qpos5 = smpl_to_qpos_ref(root_aa_zup, body_aa_yup, transl_zup, scatter_mapping)
check_fk("Scatter mapping used as gather (WRONG)", qpos5[0])

# =====================================================================
# TEST 6: XYZ euler (correct MuJoCo convention) + correct reorder
# =====================================================================
print("\n" + "=" * 70)
print("TEST 6: NO body transform + correct reorder + XYZ euler (true MuJoCo)")
print("=" * 70)
def smpl_to_qpos_xyz(root_aa, body_aa_24, transl, smpl_2_mj):
    """Same as ref but using intrinsic XYZ euler (true MuJoCo convention)."""
    T = root_aa.shape[0]
    qpos = np.zeros((T, 76), dtype=np.float64)
    qpos[:, :3] = transl.astype(np.float64) + body_pos_1
    root_quat_xyzw = sRot.from_rotvec(root_aa).as_quat()
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]

    all_aa = np.zeros((T, 24, 3), dtype=np.float64)
    all_aa[:, 0] = root_aa
    all_aa[:, 1:] = body_aa_24

    all_euler = sRot.from_rotvec(all_aa.reshape(-1, 3)).as_euler("XYZ").reshape(T, 24, 3)
    all_euler_mj = all_euler[:, smpl_2_mj]
    qpos[:, 7:] = all_euler_mj[:, 1:].reshape(T, 69)
    return qpos

qpos6 = smpl_to_qpos_xyz(root_aa_zup, body_aa_yup, transl_zup, smpl_2_mujoco)
check_fk("No body transform + correct reorder + XYZ euler", qpos6[0])

# =====================================================================
# TEST 7: XYZ euler + body transform + correct reorder
# =====================================================================
print("\n" + "=" * 70)
print("TEST 7: Body transform + correct reorder + XYZ euler")
print("=" * 70)
qpos7 = smpl_to_qpos_xyz(root_aa_zup, body_aa_zup, transl_zup, smpl_2_mujoco)
check_fk("Body transform + correct reorder + XYZ euler", qpos7[0])

# =====================================================================
# Also check multiple frames for the best combination
# =====================================================================
print("\n" + "=" * 70)
print("MULTI-FRAME CHECK: Best combination(s) across first 10 frames")
print("=" * 70)

tests = [
    ("T1: no_xform + correct + ZYX", qpos1),
    ("T2: xform + correct + ZYX", qpos2),
    ("T6: no_xform + correct + XYZ", qpos6),
    ("T7: xform + correct + XYZ", qpos7),
]

for label, qpos_arr in tests:
    min_zs = []
    pelvis_zs = []
    for t in range(min(10, T)):
        data.qpos[:] = qpos_arr[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        min_zs.append(data.xpos[1:, 2].min())
        pelvis_zs.append(data.xpos[1, 2])
    print(f"  {label}")
    print(f"    Pelvis z: mean={np.mean(pelvis_zs):.4f}, range=[{np.min(pelvis_zs):.4f}, {np.max(pelvis_zs):.4f}]")
    print(f"    Min body z: mean={np.mean(min_zs):.4f}, range=[{np.min(min_zs):.4f}, {np.max(min_zs):.4f}]")
    above = all(z > -0.05 for z in min_zs)
    near_ground = np.mean(min_zs) < 0.15
    print(f"    All above ground: {'YES' if above else 'NO'}, Avg feet near ground: {'YES' if near_ground else 'NO'}")

print("\nDone!")
