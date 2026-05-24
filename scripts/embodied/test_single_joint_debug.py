#!/usr/bin/env python3
"""Diagnostic: test single-joint rotations and check rot6d decoding.

Tests:
1. Apply known rotation to single joint, verify FK result direction
2. Check actual rot6d values from motion data
3. Verify axis-angle values are reasonable for the crouching pose
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as sRot

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
MJCF = f"{CEPH}/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
NPZ = f"{CEPH}/output/embodied_t2m_v4/data/npz/v4_crouch_002.npz"

model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)

# =====================================================================
# Part 1: Understand MuJoCo joint axes for L_Hip
# =====================================================================
print("=" * 70)
print("PART 1: MuJoCo joint axes and body positions")
print("=" * 70)

# Body tree info
for bid in range(min(10, model.nbody)):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
    parent = model.body_parentid[bid]
    pname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) or "world"
    pos = model.body_pos[bid]
    print(f"  body {bid}: {name:15s} parent={pname:15s} pos={pos}")

# Joint axes
print(f"\nJoint axes for lower body:")
for jid in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"j{jid}"
    bid = model.jnt_bodyid[jid]
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
    if bname in ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip"]:
        axis = model.jnt_axis[jid]
        qpos_adr = model.jnt_qposadr[jid]
        print(f"  {name:18s} body={bname:10s} axis={axis} qpos[{qpos_adr}]")

# =====================================================================
# Part 2: Single-joint rotation tests
# =====================================================================
print("\n" + "=" * 70)
print("PART 2: Single-joint rotation tests (L_Hip)")
print("=" * 70)

# T-pose reference
zero_qpos = np.zeros(76)
zero_qpos[2] = 0.94
zero_qpos[3] = 1.0  # identity quat
data.qpos[:] = zero_qpos
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# Get body positions for T-pose
tpose_positions = {}
for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
    if name in ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe"]:
        tpose_positions[name] = data.xpos[bid].copy()
        print(f"  T-pose {name:10s}: {data.xpos[bid]}")

# L_Hip joints are at qpos[7:10] (L_Hip_x, L_Hip_y, L_Hip_z)
# In MuJoCo Z-up model, what does each axis do?

print(f"\n--- Single hinge rotations on L_Hip ---")
for angle_deg in [30, 60, 90]:
    for axis_idx, axis_name in [(0, "L_Hip_x"), (1, "L_Hip_y"), (2, "L_Hip_z")]:
        qpos = zero_qpos.copy()
        angle_rad = np.radians(angle_deg)
        qpos[7 + axis_idx] = angle_rad
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        knee_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_Knee")]
        ankle_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_Ankle")]
        delta = knee_pos - tpose_positions["L_Knee"]
        print(f"  {axis_name}={angle_deg:3d}°: L_Knee={knee_pos}, delta={delta}")

# =====================================================================
# Part 3: Test known SMPL rotation → Euler conversion
# =====================================================================
print("\n" + "=" * 70)
print("PART 3: SMPL axis-angle → Euler → MuJoCo FK")
print("  Test: hip flexion (-90° around X in SMPL Y-up → what in MuJoCo Z-up?)")
print("=" * 70)

_YUP_TO_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

# In SMPL Y-up, hip flexion = rotation around X axis (sagittal plane)
# Positive X rotation in Y-up: thigh rotates from -Y toward +Z (forward)
smpl_hip_flexion_90 = np.array([np.pi/2, 0, 0])  # 90° around X
smpl_hip_flexion_45 = np.array([np.pi/4, 0, 0])  # 45° around X

for label, aa_yup in [("45° flexion", smpl_hip_flexion_45), ("90° flexion", smpl_hip_flexion_90)]:
    # Method A: No body transform, ZYX euler
    euler_a = sRot.from_rotvec(aa_yup).as_euler("ZYX")
    # Method B: Transform body aa to Z-up, then ZYX euler
    aa_zup = aa_yup @ _YUP_TO_ZUP.T
    euler_b = sRot.from_rotvec(aa_zup).as_euler("ZYX")
    # Method C: No body transform, XYZ euler
    euler_c = sRot.from_rotvec(aa_yup).as_euler("XYZ")

    print(f"\n  {label}:")
    print(f"    aa Y-up = {aa_yup}")
    print(f"    aa Z-up = {aa_zup}")
    print(f"    Euler ZYX (no xform) = {euler_a}")
    print(f"    Euler ZYX (xformed)  = {euler_b}")
    print(f"    Euler XYZ (no xform) = {euler_c}")

    # Apply each to L_Hip and check FK
    for method, euler_val in [("ZYX no_xform", euler_a), ("ZYX xformed", euler_b), ("XYZ no_xform", euler_c)]:
        qpos = zero_qpos.copy()
        qpos[7:10] = euler_val
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        knee_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_Knee")]
        ankle_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_Ankle")]
        toe_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_Toe")]
        print(f"    {method:20s}: L_Knee z={knee_pos[2]:.4f}, L_Ankle z={ankle_pos[2]:.4f}, L_Toe z={toe_pos[2]:.4f}")

# =====================================================================
# Part 4: Check actual motion data values
# =====================================================================
print("\n" + "=" * 70)
print("PART 4: Actual motion_135 data analysis (crouching pose)")
print("=" * 70)

data_npz = np.load(NPZ, allow_pickle=True)
motion = data_npz['motion_135']
T = motion.shape[0]
transl_yup = motion[:, :3].copy()
rot6d = motion[:, 3:].reshape(T, 22, 6)

def rot6d_to_rotmat(r6d):
    shape = r6d.shape[:-1]
    r6d = r6d.reshape(-1, 6)
    r6d = r6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = r6d[..., :3]; a2 = r6d[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).reshape(*shape, 3, 3)

rotmat = rot6d_to_rotmat(rot6d)
aa_all = sRot.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 22, 3)

# Check rotation matrices are valid
det = np.linalg.det(rotmat[0].reshape(-1, 3, 3))
print(f"Rotation matrix determinants (frame 0): min={det.min():.6f}, max={det.max():.6f} (should be ~1.0)")

# Check angle magnitudes
angles = np.linalg.norm(aa_all[0], axis=-1)
SMPL_NAMES = [
    "Root", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
    "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
    "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]

print(f"\nFrame 0 axis-angle values (Y-up):")
for j in range(22):
    aa = aa_all[0, j]
    angle = angles[j]
    print(f"  {j:2d} {SMPL_NAMES[j]:15s}: aa={aa}, angle={np.degrees(angle):.1f}°")

# Check: what does SMPL FK give for joint positions?
# We have 'positions' in the NPZ?
print(f"\nNPZ keys: {list(data_npz.keys())}")
if 'positions' in data_npz:
    positions = data_npz['positions']  # (T, 22, 3)
    print(f"Positions shape: {positions.shape}")
    print(f"Frame 0 positions (Y-up):")
    for j in range(22):
        pos = positions[0, j]
        print(f"  {j:2d} {SMPL_NAMES[j]:15s}: pos={pos}")

# =====================================================================
# Part 5: Check rot6d raw values for identity check
# =====================================================================
print("\n" + "=" * 70)
print("PART 5: Raw rot6d values - are they reasonable?")
print("=" * 70)

# Identity rot6d (row-major) should be: [1, 0, 0, 1, 0, 0]
for j in range(min(6, 22)):
    r6d = rot6d[0, j]
    # Check how close to identity
    identity_r6d = np.array([1, 0, 0, 1, 0, 0], dtype=np.float64)
    diff_to_identity = np.linalg.norm(r6d - identity_r6d)
    print(f"  Joint {j:2d} ({SMPL_NAMES[j]:15s}): rot6d={r6d}, dist_to_identity={diff_to_identity:.4f}")

# =====================================================================
# Part 6: Direct test - apply ONLY root rotation + translation, zero body
# =====================================================================
print("\n" + "=" * 70)
print("PART 6: Only root + translation, all body joints zero")
print("=" * 70)

transl_zup = (transl_yup[0:1].astype(np.float64) @ np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=np.float64).T).astype(np.float32)
R_root_yup = sRot.from_rotvec(aa_all[0:1, 0]).as_matrix()
R_root_zup = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=np.float64)[None] @ R_root_yup
root_aa_zup = sRot.from_matrix(R_root_zup).as_rotvec().astype(np.float32)
root_quat_xyzw = sRot.from_rotvec(root_aa_zup).as_quat()

qpos = np.zeros(76)
qpos[:3] = transl_zup[0] + model.body_pos[1]
qpos[3:7] = root_quat_xyzw[0, [3, 0, 1, 2]]
# body joints = all zeros (T-pose)
data.qpos[:] = qpos
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

print(f"Root-only (body=zeros):")
print(f"  Pelvis: {data.xpos[1]} (z={data.xpos[1,2]:.4f})")
for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
    if name in ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]:
        print(f"  {name}: z={data.xpos[bid, 2]:.4f}")
all_z = data.xpos[1:, 2]
print(f"  Min body z: {all_z.min():.4f}")

# =====================================================================
# Part 7: Add joints ONE AT A TIME to find which joint causes the problem
# =====================================================================
print("\n" + "=" * 70)
print("PART 7: Add body joints one at a time (using correct reorder + ZYX)")
print("=" * 70)

SMPL_BONE_ORDER_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
    "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
    "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

# Build body qpos addr
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
smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in mujoco_body_names if q in SMPL_BONE_ORDER_NAMES]

# Build smpl_pose frame 0 (Y-up, no body transform)
smpl_pose_yup = np.zeros(72, dtype=np.float32)
smpl_pose_yup[:3] = aa_all[0, 0]
smpl_pose_yup[3:66] = aa_all[0, 1:].flatten()

# Convert ALL joints to euler
joint_aa = smpl_pose_yup.reshape(24, 3)
all_euler = sRot.from_rotvec(joint_aa).as_euler("ZYX")
all_euler_mj = all_euler[smpl_2_mujoco]  # reorder

# Base qpos with correct root
base_qpos = np.zeros(76, dtype=np.float64)
base_qpos[:3] = transl_zup[0] + model.body_pos[1]
base_qpos[3:7] = root_quat_xyzw[0, [3, 0, 1, 2]]

# Add one body joint at a time
for smpl_joint_idx in range(1, 24):
    smpl_name = SMPL_BONE_ORDER_NAMES[smpl_joint_idx]
    aa_val = joint_aa[smpl_joint_idx]
    angle = np.linalg.norm(aa_val)

    if angle < 0.01:  # skip near-zero joints
        continue

    # Find where this joint goes in MuJoCo
    if smpl_name in body_qposaddr:
        qs, qe = body_qposaddr[smpl_name]
        euler_val = sRot.from_rotvec(aa_val).as_euler("ZYX")

        qpos = base_qpos.copy()
        qpos[qs:qe] = euler_val

        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        min_z = data.xpos[1:, 2].min()
        pelvis_z = data.xpos[1, 2]
        # Find foot z
        l_toe_z = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_Toe"), 2]
        r_toe_z = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_Toe"), 2]

        status = "OK" if min_z > -0.05 and min(l_toe_z, r_toe_z) < 0.15 else "BAD" if min_z < -0.5 else "FLOAT"
        print(f"  +{smpl_name:15s} (angle={np.degrees(angle):5.1f}°, euler={euler_val}): "
              f"min_z={min_z:.4f} L_Toe={l_toe_z:.4f} R_Toe={r_toe_z:.4f} [{status}]")

print("\nDone!")
