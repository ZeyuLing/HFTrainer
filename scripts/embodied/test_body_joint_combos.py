#!/usr/bin/env python3
"""Diagnose body joint pose issues: test different coord transform + Euler combos.

FK check: does each combination produce a valid standing/crouching pose with
feet near ground level (Z≈0)?
"""
import numpy as np
import os, sys, tempfile

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco not installed"); sys.exit(1)

from scipy.spatial.transform import Rotation as sRot

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
MJCF = f"{CEPH}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
NPZ = f"{CEPH}/output/embodied_t2m_v4/data/npz/v4_crouch_002.npz"

_YUP_TO_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

# Load motion
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

print(f"Motion: T={T}")
print(f"Raw Y-up: x={transl_yup[0,0]:.4f}, y(h)={transl_yup[0,1]:.4f}, z={transl_yup[0,2]:.4f}")

# Load model
model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)
body_pos_1 = model.body_pos[1].copy()

# Z-up translation
transl_zup = (transl_yup.astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32)

# Z-up root orientation
R_root_yup = sRot.from_rotvec(smpl_pose_yup[:, :3]).as_matrix()
R_root_zup = _YUP_TO_ZUP[None] @ R_root_yup
root_aa_zup = sRot.from_matrix(R_root_zup).as_rotvec().astype(np.float32)

# Body joint axis-angle (Y-up, untransformed)
body_aa_yup = smpl_pose_yup[:, 3:66].reshape(T, 21, 3)

# Body joint axis-angle (transformed via M @ aa)
body_aa_zup_v1 = (body_aa_yup.reshape(-1, 3).astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32).reshape(T, 21, 3)

# Body joint rotation matrices (proper change of basis: M @ R @ M^T)
body_R_yup = sRot.from_rotvec(body_aa_yup.reshape(-1, 3)).as_matrix().reshape(T, 21, 3, 3)
body_R_zup = _YUP_TO_ZUP[None, None] @ body_R_yup @ _YUP_TO_ZUP.T[None, None]
body_aa_zup_v2 = sRot.from_matrix(body_R_zup.reshape(-1, 3, 3)).as_rotvec().reshape(T, 21, 3).astype(np.float32)

# Verify: v1 (aa transform) vs v2 (proper change of basis)
diff_v1_v2 = np.abs(body_aa_zup_v1[0] - body_aa_zup_v2[0]).max()
print(f"\nBody joint transform v1 vs v2 max diff: {diff_v1_v2:.6f}")
print(f"  (v1: aa @ M.T,  v2: M @ R @ M^T → aa)")

# For an orthogonal M: R' = M @ R @ M^T  has axis' = M @ axis, same angle
# So aa' = M @ aa = same as v1. They should be identical.

def build_qpos(transl_z, root_aa_z, body_aa, euler_conv):
    """Build qpos from Z-up root + body joint axis-angles."""
    T = root_aa_z.shape[0]
    qpos = np.zeros((T, 76), dtype=np.float64)
    qpos[:, :3] = transl_z.astype(np.float64) + body_pos_1
    root_quat_xyzw = sRot.from_rotvec(root_aa_z).as_quat()
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]

    ba = body_aa.reshape(-1, 3)
    # Pad: smpl has 21 joints (1:22), mujoco needs 23 (pad 2 zeros)
    body_euler = sRot.from_rotvec(ba).as_euler(euler_conv)
    body_euler = body_euler.reshape(T, 21, 3)
    # Pad joints 22, 23 with zeros
    body_euler_full = np.zeros((T, 23, 3), dtype=np.float64)
    body_euler_full[:, :21] = body_euler
    qpos[:, 7:] = body_euler_full.reshape(T, 69)
    return qpos

def check_fk(label, qpos_frame):
    data.qpos[:] = qpos_frame
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    pelvis_z = data.xpos[1, 2]
    all_z = data.xpos[1:, 2]
    # Find feet
    foot_info = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name in ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]:
            foot_info[name] = data.xpos[bid, 2]
    above = all_z.min() > -0.05
    feet_near_ground = all((z < 0.15 for z in foot_info.values())) if foot_info else False
    print(f"  [{label}]")
    print(f"    Pelvis z={pelvis_z:.4f}")
    for k, v in foot_info.items():
        print(f"    {k}: z={v:.4f}")
    print(f"    Min/Max body z: [{all_z.min():.4f}, {all_z.max():.4f}]")
    print(f"    Above ground: {'YES' if above else 'NO'}  |  Feet on ground: {'YES' if feet_near_ground else 'NO'}")
    return feet_near_ground

# =====================================================================
# Test 0: Zero pose (T-pose) — sanity check
# =====================================================================
print("\n" + "=" * 70)
print("TEST 0: Zero pose (T-pose)")
print("=" * 70)
zero_qpos = np.zeros(76)
zero_qpos[2] = 0.94  # approximate standing height
zero_qpos[3] = 1.0   # w=1 quaternion (identity)
check_fk("T-pose, h=0.94", zero_qpos)

# =====================================================================
# Test combinations
# =====================================================================
combos = [
    ("Body: YUP (no transform) + XYZ euler", body_aa_yup, "XYZ"),
    ("Body: YUP (no transform) + ZYX euler", body_aa_yup, "ZYX"),
    ("Body: YUP (no transform) + xyz euler", body_aa_yup, "xyz"),
    ("Body: ZUP (aa@M.T)      + XYZ euler", body_aa_zup_v1, "XYZ"),
    ("Body: ZUP (aa@M.T)      + ZYX euler", body_aa_zup_v1, "ZYX"),
    ("Body: ZUP (aa@M.T)      + xyz euler", body_aa_zup_v1, "xyz"),
]

print("\n" + "=" * 70)
print("TESTING ALL COMBINATIONS (frame 0)")
print("=" * 70)

for label, body_aa, euler_conv in combos:
    qpos = build_qpos(transl_zup, root_aa_zup, body_aa, euler_conv)
    check_fk(label, qpos[0])

# =====================================================================
# Additional: Check what reference code does (no coord transform)
# =====================================================================
print("\n" + "=" * 70)
print("REFERENCE: No coord transform at all (raw Y-up → qpos)")
print("=" * 70)

def build_qpos_raw(transl, smpl_pose, euler_conv):
    T = smpl_pose.shape[0]
    qpos = np.zeros((T, 76), dtype=np.float64)
    joint_aa = smpl_pose.reshape(T, 24, 3)
    qpos[:, :3] = transl.astype(np.float64) + body_pos_1
    root_quat_xyzw = sRot.from_rotvec(joint_aa[:, 0]).as_quat()
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]
    body_aa = joint_aa[:, 1:].reshape(-1, 3)
    body_euler = sRot.from_rotvec(body_aa).as_euler(euler_conv)
    body_euler = body_euler.reshape(T, 23, 3)
    qpos[:, 7:] = body_euler.reshape(T, 69)
    return qpos

qpos_raw = build_qpos_raw(transl_yup, smpl_pose_yup, "ZYX")
check_fk("Raw Y-up + ZYX (reference style)", qpos_raw[0])

print("\nDone!")
