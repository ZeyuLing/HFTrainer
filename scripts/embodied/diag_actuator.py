#!/usr/bin/env python3
"""Diagnose zero actuator forces in SMPL MuJoCo model."""
import sys
import numpy as np

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco not installed"); sys.exit(1)

MJCF = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml"

print("=" * 70)
print("TEST 1: Original XML (no modifications)")
print("=" * 70)
m = mujoco.MjModel.from_xml_path(MJCF)
d = mujoco.MjData(m)

# Print actuator properties for first 3 actuators
for i in range(min(3, m.nu)):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
    print(f"\nActuator {i} '{name}':")
    print(f"  gaintype={m.actuator_gaintype[i]}, dyntype={m.actuator_dyntype[i]}")
    print(f"  trntype={m.actuator_trntype[i]}, trnid={m.actuator_trnid[i]}")
    print(f"  gainprm[:3]={m.actuator_gainprm[i, :3]}")
    print(f"  biastype={m.actuator_biastype[i]}, biasprm[:3]={m.actuator_biasprm[i, :3]}")
    print(f"  gear={m.actuator_gear[i]}")
    print(f"  ctrllimited={m.actuator_ctrllimited[i]}, ctrlrange={m.actuator_ctrlrange[i]}")
    print(f"  forcelimited={m.actuator_forcelimited[i]}, forcerange={m.actuator_forcerange[i]}")

# Set initial pose (standing) and a ctrl value
mujoco.mj_resetData(m, d)
d.qpos[2] = 1.2  # stand up

# Set ctrl to something non-zero
d.ctrl[:] = 0.5  # with ctrlrange [-1,1] this should be clipped to 0.5

mujoco.mj_forward(m, d)
print(f"\nAfter mj_forward:")
print(f"  actuator_force[:10] = {d.actuator_force[:10]}")
print(f"  actuator_force range: [{d.actuator_force.min():.4f}, {d.actuator_force.max():.4f}]")
print(f"  qfrc_actuator range: [{d.qfrc_actuator.min():.4f}, {d.qfrc_actuator.max():.4f}]")
print(f"  ctrl[:5] = {d.ctrl[:5]}")
print(f"  qpos[7:12] = {d.qpos[7:12]}")

mujoco.mj_step(m, d)
print(f"\nAfter mj_step:")
print(f"  actuator_force[:10] = {d.actuator_force[:10]}")
print(f"  actuator_force range: [{d.actuator_force.min():.4f}, {d.actuator_force.max():.4f}]")
print(f"  qfrc_actuator range: [{d.qfrc_actuator.min():.4f}, {d.qfrc_actuator.max():.4f}]")

print("\n" + "=" * 70)
print("TEST 2: Modified for PD (our script's configuration)")
print("=" * 70)
m2 = mujoco.MjModel.from_xml_path(MJCF)
d2 = mujoco.MjData(m2)

# Stiffness/damping from unified_pipeline.yaml
stiffness = [800.0]*9 + [500.0]*3 + [800.0]*9 + [500.0]*3 + \
            [1000.0]*9 + [500.0]*6 + [500.0]*9 + [300.0]*6 + \
            [500.0]*9 + [300.0]*6
damping = [s/10 for s in stiffness]

# Apply our modifications
m2.opt.timestep = 0.001
m2.jnt_stiffness[:] = 0.0
m2.dof_damping[:] = 0.0
m2.dof_frictionloss[:] = 0.0

for i in range(m2.nu):
    kp = stiffness[i]
    kd = damping[i]
    m2.actuator_gear[i, 0] = 1.0
    m2.actuator_gainprm[i, 0] = kp
    m2.actuator_biastype[i] = 1  # mjBIAS_AFFINE
    m2.actuator_biasprm[i, 0] = 0.0
    m2.actuator_biasprm[i, 1] = -kp
    m2.actuator_biasprm[i, 2] = -kd
    m2.actuator_ctrllimited[i] = 0
    m2.actuator_forcerange[i, 0] = -500.0
    m2.actuator_forcerange[i, 1] = 500.0
    m2.actuator_forcelimited[i] = 1

# Print modified properties
for i in range(min(3, m2.nu)):
    name = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
    print(f"\nActuator {i} '{name}':")
    print(f"  gaintype={m2.actuator_gaintype[i]}, dyntype={m2.actuator_dyntype[i]}")
    print(f"  trntype={m2.actuator_trntype[i]}, trnid={m2.actuator_trnid[i]}")
    print(f"  gainprm[:3]={m2.actuator_gainprm[i, :3]}")
    print(f"  biastype={m2.actuator_biastype[i]}, biasprm[:3]={m2.actuator_biasprm[i, :3]}")
    print(f"  gear={m2.actuator_gear[i]}")
    print(f"  ctrllimited={m2.actuator_ctrllimited[i]}, ctrlrange={m2.actuator_ctrlrange[i]}")
    print(f"  forcelimited={m2.actuator_forcelimited[i]}, forcerange={m2.actuator_forcerange[i]}")

# Set initial pose and ctrl target
mujoco.mj_resetData(m2, d2)
d2.qpos[2] = 1.2
d2.ctrl[:] = 0.5  # target joint angle = 0.5 rad

mujoco.mj_forward(m2, d2)
print(f"\nAfter mj_forward:")
print(f"  actuator_force[:10] = {d2.actuator_force[:10]}")
print(f"  actuator_force range: [{d2.actuator_force.min():.4f}, {d2.actuator_force.max():.4f}]")
print(f"  qfrc_actuator[:12] = {d2.qfrc_actuator[:12]}")
print(f"  qfrc_actuator range: [{d2.qfrc_actuator.min():.4f}, {d2.qfrc_actuator.max():.4f}]")
print(f"  ctrl[:5] = {d2.ctrl[:5]}")
print(f"  qpos[7:12] = {d2.qpos[7:12]}")

# Expected force for actuator 0: kp*(ctrl - qpos[7]) - kd*qvel[6]
print(f"\n  Manual PD check for actuator 0:")
print(f"    kp={stiffness[0]}, ctrl={d2.ctrl[0]:.6f}, q=qpos[7]={d2.qpos[7]:.6f}, qd=qvel[6]={d2.qvel[6]:.6f}")
expected = stiffness[0] * (d2.ctrl[0] - d2.qpos[7]) - damping[0] * d2.qvel[6]
print(f"    expected force = {expected:.4f}, actual = {d2.actuator_force[0]:.4f}")

mujoco.mj_step(m2, d2)
print(f"\nAfter mj_step:")
print(f"  actuator_force[:10] = {d2.actuator_force[:10]}")
print(f"  actuator_force range: [{d2.actuator_force.min():.4f}, {d2.actuator_force.max():.4f}]")
print(f"  qfrc_actuator[:12] = {d2.qfrc_actuator[:12]}")
print(f"  qfrc_actuator range: [{d2.qfrc_actuator.min():.4f}, {d2.qfrc_actuator.max():.4f}]")

# Check between step1 and step2
mujoco.mj_resetData(m2, d2)
d2.qpos[2] = 1.2
d2.ctrl[:] = 0.5
mujoco.mj_step1(m2, d2)
print(f"\nAfter mj_step1 only (before step2):")
print(f"  actuator_force[:10] = {d2.actuator_force[:10]}")
print(f"  actuator_force range: [{d2.actuator_force.min():.4f}, {d2.actuator_force.max():.4f}]")
print(f"  qfrc_actuator[:12] = {d2.qfrc_actuator[:12]}")

mujoco.mj_step2(m2, d2)
print(f"\nAfter mj_step2:")
print(f"  actuator_force[:10] = {d2.actuator_force[:10]}")
print(f"  actuator_force range: [{d2.actuator_force.min():.4f}, {d2.actuator_force.max():.4f}]")
print(f"  qfrc_actuator[:12] = {d2.qfrc_actuator[:12]}")

print("\n" + "=" * 70)
print("TEST 3: Explicit gaintype=0 setting")
print("=" * 70)
m3 = mujoco.MjModel.from_xml_path(MJCF)
d3 = mujoco.MjData(m3)

m3.opt.timestep = 0.001
m3.jnt_stiffness[:] = 0.0
m3.dof_damping[:] = 0.0
m3.dof_frictionloss[:] = 0.0

for i in range(m3.nu):
    kp = stiffness[i]
    kd = damping[i]
    m3.actuator_gaintype[i] = 0  # EXPLICITLY set FIXED gain
    m3.actuator_gear[i, 0] = 1.0
    m3.actuator_gainprm[i, 0] = kp
    m3.actuator_biastype[i] = 1
    m3.actuator_biasprm[i, 0] = 0.0
    m3.actuator_biasprm[i, 1] = -kp
    m3.actuator_biasprm[i, 2] = -kd
    m3.actuator_ctrllimited[i] = 0
    m3.actuator_forcerange[i, 0] = -500.0
    m3.actuator_forcerange[i, 1] = 500.0
    m3.actuator_forcelimited[i] = 1

mujoco.mj_resetData(m3, d3)
d3.qpos[2] = 1.2
d3.ctrl[:] = 0.5

mujoco.mj_forward(m3, d3)
print(f"\nWith explicit gaintype=0:")
print(f"  actuator_force[:10] = {d3.actuator_force[:10]}")
print(f"  actuator_force range: [{d3.actuator_force.min():.4f}, {d3.actuator_force.max():.4f}]")
print(f"  qfrc_actuator range: [{d3.qfrc_actuator.min():.4f}, {d3.qfrc_actuator.max():.4f}]")

print("\n" + "=" * 70)
print("TEST 4: MuJoCo version info")
print("=" * 70)
print(f"  mujoco version: {mujoco.__version__}")
print(f"  mj_version: {mujoco.mj_version()}")

# Check if model has actuator_actlimited or actuator_actrange
print(f"\n  model.nu = {m2.nu}")
print(f"  model.nq = {m2.nq}")
print(f"  model.nv = {m2.nv}")
print(f"  model.na = {m2.na}")  # number of actuator activations (0 if all dyntype=0)

print("\n" + "=" * 70)
print("TEST 5: Simple position actuator via XML (gold standard)")
print("=" * 70)
# Create a minimal XML with explicit position actuator to verify MuJoCo works
import tempfile, os
simple_xml = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0.3 0 0">
        <joint name="hinge1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0.3 0 0" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="pos1" joint="hinge1" kp="100" kv="10"/>
  </actuator>
</mujoco>
"""
with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
    f.write(simple_xml)
    tmp = f.name

ms = mujoco.MjModel.from_xml_path(tmp)
ds = mujoco.MjData(ms)
os.unlink(tmp)

print(f"  Simple model: nu={ms.nu}, nq={ms.nq}, nv={ms.nv}")
print(f"  Actuator gaintype={ms.actuator_gaintype[0]}, dyntype={ms.actuator_dyntype[0]}")
print(f"  gainprm[:3]={ms.actuator_gainprm[0, :3]}")
print(f"  biastype={ms.actuator_biastype[0]}, biasprm[:3]={ms.actuator_biasprm[0, :3]}")
print(f"  gear={ms.actuator_gear[0]}")

ds.ctrl[0] = 1.0  # target = 1 rad
mujoco.mj_forward(ms, ds)
print(f"\n  After forward with ctrl=1.0:")
print(f"    actuator_force = {ds.actuator_force[0]:.4f}")
print(f"    qfrc_actuator = {ds.qfrc_actuator}")
print(f"    qpos = {ds.qpos}")

print("\n" + "=" * 70)
print("TEST 6: Motor actuator converted to PD manually (minimal)")
print("=" * 70)
motor_xml = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0.3 0 0">
        <joint name="hinge1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0.3 0 0" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="hinge1" gear="500" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
    f.write(motor_xml)
    tmp = f.name

mm = mujoco.MjModel.from_xml_path(tmp)
dm = mujoco.MjData(mm)
os.unlink(tmp)

print(f"  Before modification:")
print(f"    gaintype={mm.actuator_gaintype[0]}, gainprm[:3]={mm.actuator_gainprm[0,:3]}")
print(f"    biastype={mm.actuator_biastype[0]}, biasprm[:3]={mm.actuator_biasprm[0,:3]}")
print(f"    gear={mm.actuator_gear[0]}")
print(f"    ctrllimited={mm.actuator_ctrllimited[0]}, ctrlrange={mm.actuator_ctrlrange[0]}")

# Convert motor to PD (same as our script)
kp = 800.0
kd = 80.0
mm.actuator_gear[0, 0] = 1.0
mm.actuator_gainprm[0, 0] = kp
mm.actuator_biastype[0] = 1
mm.actuator_biasprm[0, 0] = 0.0
mm.actuator_biasprm[0, 1] = -kp
mm.actuator_biasprm[0, 2] = -kd
mm.actuator_ctrllimited[0] = 0
mm.actuator_forcerange[0, 0] = -500.0
mm.actuator_forcerange[0, 1] = 500.0
mm.actuator_forcelimited[0] = 1

print(f"\n  After modification:")
print(f"    gaintype={mm.actuator_gaintype[0]}, gainprm[:3]={mm.actuator_gainprm[0,:3]}")
print(f"    biastype={mm.actuator_biastype[0]}, biasprm[:3]={mm.actuator_biasprm[0,:3]}")
print(f"    gear={mm.actuator_gear[0]}")
print(f"    ctrllimited={mm.actuator_ctrllimited[0]}, ctrlrange={mm.actuator_ctrlrange[0]}")
print(f"    forcelimited={mm.actuator_forcelimited[0]}, forcerange={mm.actuator_forcerange[0]}")

dm.ctrl[0] = 0.5  # target = 0.5 rad, current = 0 rad
mujoco.mj_forward(mm, dm)
expected = kp * (0.5 - 0.0) - kd * 0.0  # = 400.0, clipped to 500
print(f"\n  After forward with ctrl=0.5:")
print(f"    actuator_force = {dm.actuator_force[0]:.4f} (expected ~400.0, clipped to 500 if forcelimited)")
print(f"    qfrc_actuator = {dm.qfrc_actuator}")

mujoco.mj_step(mm, dm)
print(f"\n  After step:")
print(f"    actuator_force = {dm.actuator_force[0]:.4f}")
print(f"    qfrc_actuator = {dm.qfrc_actuator}")
print(f"    qpos = {dm.qpos}")

print("\nDone!")
