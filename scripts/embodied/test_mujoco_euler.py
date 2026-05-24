#!/usr/bin/env python3
"""Determine MuJoCo's Euler convention for 3-hinge ball joints.

MuJoCo has L_Hip_x, L_Hip_y, L_Hip_z hinges with axes [1,0,0], [0,1,0], [0,0,1].
Question: if qpos = [a, b, c], does MuJoCo compute:
  A) R = Rx(a) @ Ry(b) @ Rz(c)   (intrinsic XYZ) → use as_euler("XYZ")
  B) R = Rz(c) @ Ry(b) @ Rx(a)   (extrinsic xyz) → use as_euler("xyz")
"""
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import mujoco, tempfile, os

# Minimal model with 3 hinges in XYZ order
xml = """
<mujoco>
  <worldbody>
    <body name="test" pos="0 0 1">
      <joint name="hx" type="hinge" axis="1 0 0"/>
      <joint name="hy" type="hinge" axis="0 1 0"/>
      <joint name="hz" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
    f.write(xml)
    tmp = f.name

model = mujoco.MjModel.from_xml_path(tmp)
data = mujoco.MjData(model)
os.unlink(tmp)

print(f"Model: nq={model.nq}, njnt={model.njnt}")
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    print(f"  Joint {j}: {name}, axis={model.jnt_axis[j]}, qposadr={model.jnt_qposadr[j]}")

# Test with known angles
test_angles = [0.5, 0.3, 0.7]  # a, b, c
data.qpos[:] = test_angles
mujoco.mj_forward(model, data)

# Read body rotation matrix from MuJoCo
xmat = data.xmat[1].reshape(3, 3)
print(f"\nSet qpos = {test_angles}")
print(f"MuJoCo xmat (body 1):\n{xmat}")

# Now test each convention
R_intrinsic_XYZ = sRot.from_euler("XYZ", test_angles).as_matrix()
R_extrinsic_xyz = sRot.from_euler("xyz", test_angles).as_matrix()
R_intrinsic_ZYX = sRot.from_euler("ZYX", test_angles).as_matrix()
R_extrinsic_zyx = sRot.from_euler("zyx", test_angles).as_matrix()

print(f"\nComparison with MuJoCo xmat:")
print(f"  Intrinsic 'XYZ' match: {np.allclose(xmat, R_intrinsic_XYZ, atol=1e-6)}  (max diff: {np.abs(xmat - R_intrinsic_XYZ).max():.8f})")
print(f"  Extrinsic 'xyz' match: {np.allclose(xmat, R_extrinsic_xyz, atol=1e-6)}  (max diff: {np.abs(xmat - R_extrinsic_xyz).max():.8f})")
print(f"  Intrinsic 'ZYX' match: {np.allclose(xmat, R_intrinsic_ZYX, atol=1e-6)}  (max diff: {np.abs(xmat - R_intrinsic_ZYX).max():.8f})")
print(f"  Extrinsic 'zyx' match: {np.allclose(xmat, R_extrinsic_zyx, atol=1e-6)}  (max diff: {np.abs(xmat - R_extrinsic_zyx).max():.8f})")

# Also test: which convention, when used for decomposition, gives back the original qpos?
euler_XYZ = sRot.from_matrix(xmat).as_euler("XYZ")
euler_xyz = sRot.from_matrix(xmat).as_euler("xyz")
euler_ZYX = sRot.from_matrix(xmat).as_euler("ZYX")

print(f"\nDecomposing MuJoCo xmat:")
print(f"  as_euler('XYZ') = {euler_XYZ}  match qpos: {np.allclose(euler_XYZ, test_angles, atol=1e-6)}")
print(f"  as_euler('xyz') = {euler_xyz}  match qpos: {np.allclose(euler_xyz, test_angles, atol=1e-6)}")
print(f"  as_euler('ZYX') = {euler_ZYX}  match qpos: {np.allclose(euler_ZYX, test_angles, atol=1e-6)}")

# ==========================================================
# Now test with the SMPL humanoid model
# ==========================================================
print("\n" + "=" * 70)
print("SMPL Humanoid Model Test")
print("=" * 70)

MJCF = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"
m2 = mujoco.MjModel.from_xml_path(MJCF)
d2 = mujoco.MjData(m2)

# Set a known angle on L_Hip joints (qposadr 7, 8, 9)
mujoco.mj_resetData(m2, d2)
d2.qpos[2] = 1.0  # height
d2.qpos[3] = 1.0  # quat w
d2.qpos[7] = 0.5  # L_Hip_x
d2.qpos[8] = 0.3  # L_Hip_y
d2.qpos[9] = 0.7  # L_Hip_z
mujoco.mj_forward(m2, d2)

# Get L_Hip body's rotation relative to parent (Pelvis)
# L_Hip is body 2 (body 0=world, body 1=Pelvis, body 2=L_Hip)
pelvis_rot = d2.xmat[1].reshape(3, 3)
l_hip_rot = d2.xmat[2].reshape(3, 3)
# Relative rotation: R_lhip_in_pelvis = pelvis_rot.T @ l_hip_rot
R_rel = pelvis_rot.T @ l_hip_rot

print(f"\nL_Hip relative rotation (in Pelvis frame):")
print(f"  qpos[7:10] = [{d2.qpos[7]:.1f}, {d2.qpos[8]:.1f}, {d2.qpos[9]:.1f}]")

euler_XYZ = sRot.from_matrix(R_rel).as_euler("XYZ")
euler_xyz = sRot.from_matrix(R_rel).as_euler("xyz")
euler_ZYX = sRot.from_matrix(R_rel).as_euler("ZYX")

print(f"  as_euler('XYZ') = {euler_XYZ}")
print(f"  as_euler('xyz') = {euler_xyz}")
print(f"  as_euler('ZYX') = {euler_ZYX}")
print(f"  Original qpos: [0.5, 0.3, 0.7]")
print(f"  'XYZ' match: {np.allclose(euler_XYZ, [0.5, 0.3, 0.7], atol=1e-4)}")
print(f"  'xyz' match: {np.allclose(euler_xyz, [0.5, 0.3, 0.7], atol=1e-4)}")
print(f"  'ZYX' match: {np.allclose(euler_ZYX, [0.5, 0.3, 0.7], atol=1e-4)}")

# ==========================================================
# Conclusion for smpl_mujoco.py "ZYX" usage
# ==========================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
# If "XYZ" matches, then:
#   - smpl_mujoco.py's "ZYX" returns [rZ, rY, rX] and stores as-is
#   - This means qpos = [rZ, rY, rX] which maps to [q_x, q_y, q_z] = [rZ, rY, rX]
#   - MuJoCo computes R' = Rx(rZ) @ Ry(rY) @ Rz(rX) ≠ R
#   - This would be WRONG, but the policy was trained with it, so we need to match
#
# If "xyz" matches, then:
#   - Both "xyz" and "ZYX" give the same rotation, just different angle ordering
#   - Our "xyz" usage is correct
#   - smpl_mujoco.py's "ZYX" is also correct but stores [rZ, rY, rX] → [q_x, q_y, q_z] = [rZ, rY, rX]
#   - This would give a DIFFERENT rotation from what's intended

print("Done!")
