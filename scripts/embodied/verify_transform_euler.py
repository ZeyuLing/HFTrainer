#!/usr/bin/env python3
"""Verify Y-up → Z-up transform and Euler convention for SMPL→MuJoCo.

Key questions:
1. Does _YUP_TO_ZUP produce positive Z for standing?
2. Does "xyz" vs "ZYX" Euler convention produce different qpos?
3. Does the resulting qpos place the humanoid ABOVE the ground?
"""
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import sys

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
NPZ = f"{CEPH}/output/embodied_t2m_v4/data/npz/v4_crouch_002.npz"
MJCF = f"{CEPH}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"

# ---- Load raw motion ----
data_npz = np.load(NPZ, allow_pickle=True)
motion = data_npz["motion_135"]
T = motion.shape[0]
transl_yup = motion[:, :3]
rot6d = motion[:, 3:].reshape(T, 22, 6)

print(f"Raw motion: T={T}")
print(f"  Frame 0 transl Y-up: x={transl_yup[0,0]:.4f}, y={transl_yup[0,1]:.4f}, z={transl_yup[0,2]:.4f}")
print(f"  Y (height) should be ~1.15: {transl_yup[0,1]:.4f}")

# ---- rot6d → axis-angle ----
def rot6d_to_rotmat(rot6d):
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)

rotmat = rot6d_to_rotmat(rot6d)
aa = sRot.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 22, 3)

smpl_pose = np.zeros((T, 72), dtype=np.float32)
smpl_pose[:, :3] = aa[:, 0, :]
smpl_pose[:, 3:66] = aa[:, 1:22, :].reshape(T, -1)

print(f"\n  Root orient Y-up (aa): {smpl_pose[0, :3]}")

# ===========================================================================
# TEST 1: Main script's transform: _YUP_TO_ZUP = [[0,0,1],[1,0,0],[0,1,0]]
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 1: Main script transform: [x,y,z] -> [z,x,y]")
print("=" * 70)

_YUP_TO_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

transl_zup_1 = (transl_yup.astype(np.float64) @ _YUP_TO_ZUP.T).astype(np.float32)
pose_72_zup_1 = (smpl_pose[:, :72].astype(np.float64).reshape(T * 24, 3) @ _YUP_TO_ZUP.T).reshape(T, 72).astype(np.float32)

print(f"  Transl Z-up: x={transl_zup_1[0,0]:.4f}, y={transl_zup_1[0,1]:.4f}, z={transl_zup_1[0,2]:.4f}")
print(f"  Z (height) = {transl_zup_1[0,2]:.4f} (should be ~1.15)")
print(f"  Root orient Z-up (aa): {pose_72_zup_1[0, :3]}")

# ===========================================================================
# TEST 2: Rx(-90) transform: [[1,0,0],[0,0,1],[0,-1,0]] (test_pd_standing.py)
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 2: Rx(-90°) transform: [x,y,z] -> [x,z,-y]")
print("=" * 70)

Rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

transl_zup_2 = (transl_yup.astype(np.float64) @ Rx.T).astype(np.float32)
# test_pd_standing.py transforms root orient separately via matrix multiply:
R_root = sRot.from_rotvec(smpl_pose[:, :3]).as_matrix()
R_root_zup_2 = Rx[None] @ R_root
root_orient_zup_2 = sRot.from_matrix(R_root_zup_2).as_rotvec().astype(np.float32)

print(f"  Transl Z-up: x={transl_zup_2[0,0]:.4f}, y={transl_zup_2[0,1]:.4f}, z={transl_zup_2[0,2]:.4f}")
print(f"  Z (height) = {transl_zup_2[0,2]:.4f} (should be ~1.15)")
print(f"  Root orient Z-up (aa): {root_orient_zup_2[0, :3]}")

# ===========================================================================
# TEST 3: Proper Rx(-90°) with consistent body joint transform
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 3: Rx(-90°) applied to ALL joints as rotation matrices")
print("=" * 70)

all_aa_yup = smpl_pose[:, :72].reshape(T, 24, 3)
all_R_yup = sRot.from_rotvec(all_aa_yup.reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3)
# For local body joints (not root), the transform is: R_zup = Rx @ R_yup @ Rx^T
# (change of basis for local frames)
# For root (global rotation): R_zup = Rx @ R_yup
all_R_zup_3 = np.zeros_like(all_R_yup)
all_R_zup_3[:, 0] = Rx[None] @ all_R_yup[:, 0]  # Root: global
for j in range(1, 24):
    # Local joints: change of basis
    all_R_zup_3[:, j] = Rx[None] @ all_R_yup[:, j] @ Rx.T[None]
all_aa_zup_3 = sRot.from_matrix(all_R_zup_3.reshape(-1, 3, 3)).as_rotvec().reshape(T, 24, 3).astype(np.float32)
pose_72_zup_3 = all_aa_zup_3.reshape(T, 72)

print(f"  Transl Z-up (same as TEST 2): x={transl_zup_2[0,0]:.4f}, y={transl_zup_2[0,1]:.4f}, z={transl_zup_2[0,2]:.4f}")
print(f"  Root orient Z-up (aa): {all_aa_zup_3[0, 0, :]}")
print(f"  L_Hip orient Z-up (aa): {all_aa_zup_3[0, 1, :]}")

# ===========================================================================
# Now test smpl_to_qpos with each transform
# ===========================================================================
try:
    import mujoco
except ImportError:
    print("\nERROR: mujoco not installed")
    sys.exit(1)

model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)
body_pos_1 = model.body_pos[1].copy()
print(f"\nModel body_pos[1] (Pelvis offset): {body_pos_1}")

SMPL_2_MUJOCO = list(range(23))  # identity for SMPL humanoid

# ===========================================================================
# TEST 4: smpl_to_qpos with "xyz" Euler (our script) vs "ZYX" (reference)
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 4: Compare Euler conventions on transform 1 (main script)")
print("=" * 70)

# Method A: our script's "xyz" Euler
def smpl_to_qpos_xyz(smpl_pose, transl, body_pos_1):
    T = smpl_pose.shape[0]
    qpos = np.zeros((T, 76), dtype=np.float64)
    joint_aa = smpl_pose.reshape(T, 24, 3)
    qpos[:, :3] = transl.astype(np.float64) + body_pos_1
    root_quat_xyzw = sRot.from_rotvec(joint_aa[:, 0]).as_quat()
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]  # -> wxyz
    body_aa = joint_aa[:, 1:].reshape(-1, 3)
    body_euler = sRot.from_rotvec(body_aa).as_euler("xyz")
    body_euler = body_euler.reshape(T, 23, 3)
    qpos[:, 7:] = body_euler.reshape(T, 69)
    return qpos

# Method B: reference code's "ZYX" Euler
def smpl_to_qpos_ZYX(smpl_pose, transl, body_pos_1):
    T = smpl_pose.shape[0]
    qpos = np.zeros((T, 76), dtype=np.float64)
    joint_aa = smpl_pose.reshape(T, 24, 3)
    qpos[:, :3] = transl.astype(np.float64) + body_pos_1
    root_quat_xyzw = sRot.from_rotvec(joint_aa[:, 0]).as_quat()
    qpos[:, 3:7] = root_quat_xyzw[:, [3, 0, 1, 2]]  # -> wxyz
    body_aa = joint_aa[:, 1:].reshape(-1, 3)
    body_euler = sRot.from_rotvec(body_aa).as_euler("ZYX")
    body_euler = body_euler.reshape(T, 23, 3)
    qpos[:, 7:] = body_euler.reshape(T, 69)
    return qpos

# Using transform 1 (main script's _YUP_TO_ZUP)
smpl_pose_zup1 = smpl_pose.copy()
smpl_pose_zup1[:, :72] = pose_72_zup_1

qpos_xyz = smpl_to_qpos_xyz(smpl_pose_zup1, transl_zup_1, body_pos_1)
qpos_ZYX = smpl_to_qpos_ZYX(smpl_pose_zup1, transl_zup_1, body_pos_1)

print(f"  Using 'xyz': root_pos={qpos_xyz[0,:3]}, root_quat={qpos_xyz[0,3:7]}")
print(f"  Using 'ZYX': root_pos={qpos_ZYX[0,:3]}, root_quat={qpos_ZYX[0,3:7]}")
print(f"  xyz body joints[:6] = {qpos_xyz[0, 7:13]}")
print(f"  ZYX body joints[:6] = {qpos_ZYX[0, 7:13]}")
diff = np.abs(qpos_xyz[0, 7:] - qpos_ZYX[0, 7:]).max()
print(f"  Max joint difference (xyz vs ZYX): {diff:.6f}")

# ===========================================================================
# TEST 5: MuJoCo FK — which produces correct body positions?
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 5: MuJoCo FK body positions for each method")
print("=" * 70)

def check_fk(label, qpos_frame):
    data.qpos[:] = qpos_frame
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    root_z = data.xpos[1, 2]  # Pelvis Z
    print(f"\n  [{label}]")
    print(f"    Pelvis: {data.xpos[1]} (z={root_z:.4f})")

    # Find feet
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name in ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]:
            print(f"    {name}: {data.xpos[bid]} (z={data.xpos[bid,2]:.4f})")

    # Summary
    all_z = data.xpos[1:, 2]
    print(f"    Min body Z: {all_z.min():.4f}, Max: {all_z.max():.4f}")
    is_above = all_z.min() > -0.05
    print(f"    All bodies above ground: {'YES ✓' if is_above else 'NO ✗ — UNDERGROUND!'}")
    return root_z

# Test all combinations
print("\n  --- Transform 1 (main script: [x,y,z]->[z,x,y]) ---")
check_fk("T1 + xyz Euler", qpos_xyz[0])
check_fk("T1 + ZYX Euler", qpos_ZYX[0])

# Using transform 2 (Rx(-90°), same transl, body joints via Rx@R@Rx^T)
print("\n  --- Transform 3 (Rx(-90°) proper: root=Rx@R, body=Rx@R@Rx^T) ---")
smpl_pose_zup3 = smpl_pose.copy()
smpl_pose_zup3[:, :72] = pose_72_zup_3
qpos_t3_xyz = smpl_to_qpos_xyz(smpl_pose_zup3, transl_zup_2, body_pos_1)
qpos_t3_ZYX = smpl_to_qpos_ZYX(smpl_pose_zup3, transl_zup_2, body_pos_1)
check_fk("T3 + xyz Euler", qpos_t3_xyz[0])
check_fk("T3 + ZYX Euler", qpos_t3_ZYX[0])

# ===========================================================================
# TEST 6: Reference conversion (from smpl_mujoco.py) approach
# Direct: no coord transform, just convert SMPL pose to qpos using ZYX
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 6: No coordinate transform (raw Y-up SMPL, see what happens)")
print("=" * 70)

qpos_raw_ZYX = smpl_to_qpos_ZYX(smpl_pose, transl_yup, body_pos_1)
check_fk("Raw Y-up + ZYX Euler", qpos_raw_ZYX[0])

# ===========================================================================
# TEST 7: Check the MuJoCo joint axis convention vs SciPy Euler
# ===========================================================================
print("\n" + "=" * 70)
print("TEST 7: Joint axis mapping — what does MuJoCo expect?")
print("=" * 70)

# MuJoCo hinge joints are named L_Hip_x, L_Hip_y, L_Hip_z
# They decompose into 3 hinge axes: [1,0,0], [0,1,0], [0,0,1]
# The qpos order is: L_Hip_x, L_Hip_y, L_Hip_z (x → y → z)
#
# SciPy "xyz" = R = Rz @ Ry @ Rx (extrinsic, equivalent to intrinsic X→Y→Z)
#   returns [rx, ry, rz] where rx is rotation about X first
# SciPy "ZYX" = R = Rx @ Ry @ Rz (extrinsic, equivalent to intrinsic Z→Y→X)
#   returns [rz, ry, rx] where rz is rotation about Z first
#
# MuJoCo 3-hinge: each hinge acts independently. The resulting rotation is:
#   R = Rz(q_z) @ Ry(q_y) @ Rx(q_x) where q_x, q_y, q_z are the 3 hinge angles
#   This means: applied in order x → y → z (last-applied axis = z)
#
# For MuJoCo's convention: qpos = [q_x, q_y, q_z]
# Rotation = Rz(q_z) @ Ry(q_y) @ Rx(q_x)
# This is identical to extrinsic "xyz" or intrinsic "ZYX"
# SciPy "XYZ" (intrinsic) = "xyz" (extrinsic): R = Rz @ Ry @ Rx, returns [rx, ry, rz]
# SciPy "ZYX" (intrinsic) = "zyx" (extrinsic): R = Rx @ Ry @ Rz, returns [rz, ry, rx]

# Let me test empirically:
# Set a known rotation and see which convention reproduces it

test_aa = np.array([0.3, 0.5, 0.1])  # arbitrary axis-angle
R = sRot.from_rotvec(test_aa).as_matrix()

euler_xyz = sRot.from_rotvec(test_aa).as_euler("xyz")  # extrinsic xyz
euler_XYZ = sRot.from_rotvec(test_aa).as_euler("XYZ")  # intrinsic XYZ
euler_ZYX = sRot.from_rotvec(test_aa).as_euler("ZYX")  # intrinsic ZYX
euler_zyx = sRot.from_rotvec(test_aa).as_euler("zyx")  # extrinsic zyx

print(f"  Test axis-angle: {test_aa}")
print(f"  SciPy 'xyz' (extrinsic): {euler_xyz}")
print(f"  SciPy 'XYZ' (intrinsic): {euler_XYZ}")
print(f"  SciPy 'ZYX' (intrinsic): {euler_ZYX}")
print(f"  SciPy 'zyx' (extrinsic): {euler_zyx}")

# MuJoCo hinge decomposition: R = Rz(qz) @ Ry(qy) @ Rx(qx)
# Setting qpos = euler_xyz → R_result = Rz(euler_xyz[2]) @ Ry(euler_xyz[1]) @ Rx(euler_xyz[0])
# This should match R if xyz is the right convention
# Let's verify:
from scipy.spatial.transform import Rotation

R_from_xyz = (Rotation.from_rotvec([0, 0, euler_xyz[2]]) *
              Rotation.from_rotvec([0, euler_xyz[1], 0]) *
              Rotation.from_rotvec([euler_xyz[0], 0, 0])).as_matrix()

R_from_ZYX = (Rotation.from_rotvec([0, 0, euler_ZYX[0]]) *
              Rotation.from_rotvec([0, euler_ZYX[1], 0]) *
              Rotation.from_rotvec([euler_ZYX[2], 0, 0])).as_matrix()

print(f"\n  MuJoCo hinge chain: R = Rz @ Ry @ Rx")
print(f"  Using 'xyz' angles [rx,ry,rz]→[Rx,Ry,Rz]: match = {np.allclose(R, R_from_xyz, atol=1e-6)}")
print(f"  Using 'ZYX' angles [rz,ry,rx]→[Rz,Ry,Rx]: match = {np.allclose(R, R_from_ZYX, atol=1e-6)}")

# Key insight: SciPy 'xyz' extrinsic returns [rx,ry,rz] for R = Rz @ Ry @ Rx
# MuJoCo qpos order is [q_x, q_y, q_z]
# So: qpos = [euler_xyz[0], euler_xyz[1], euler_xyz[2]] = euler_xyz
#
# SciPy 'ZYX' intrinsic returns [rZ,rY,rX] for R = RZ @ RY @ RX (same composition)
# But the order is REVERSED: [rZ,rY,rX]
# So: qpos = [euler_ZYX[2], euler_ZYX[1], euler_ZYX[0]] (need to REVERSE!)

# Verify: ZYX reversed should match xyz
print(f"\n  'ZYX' reversed == 'xyz': {np.allclose(euler_ZYX[::-1], euler_xyz, atol=1e-10)}")

# CRITICAL: If reference code uses 'ZYX' WITHOUT reversing, it would swap axes!
# Let's check: does the reference smpl_mujoco.py reverse the ZYX output?
# From the code: body_euler = sRot.from_rotvec(body_aa).as_euler("ZYX")
# Then: qpos[7:] = body_euler.flatten()
# This means qpos = [rZ, rY, rX] which is WRONG for MuJoCo [q_x, q_y, q_z]!
# UNLESS MuJoCo's hinge order is actually z,y,x (last defined first applied)?

# Let's check MuJoCo's actual joint order
print("\n  MuJoCo joint order (first body, L_Hip):")
for jid in range(min(6, model.njnt)):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"j{jid}"
    axis = model.jnt_axis[jid]
    qposadr = model.jnt_qposadr[jid]
    print(f"    jnt {jid} '{name}': axis={axis}, qposadr={qposadr}")

print("\nDone!")
