#!/bin/bash
# Run debug: per-frame root/joint tracking to understand fall cause
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

python3 -c "
import numpy as np
import mujoco
import sys
sys.path.insert(0, '.')
from scripts.embodied.run_smpl_physics_sim import (
    decode_motion_135, yup_to_zup, smpl_to_qpos, load_mujoco_model,
)

smpl_pose, transl, fps = decode_motion_135('output/embodied_t2m_v4/data/npz/v4_walk_001.npz')
smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)
model, data = load_mujoco_model('ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml')
body_pos_1 = model.body_pos[1].copy()
ref_qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)

T = ref_qpos.shape[0]
sim_dt = model.opt.timestep
ctrl_dt = 1.0 / fps
decimation = max(1, int(round(ctrl_dt / sim_dt)))

# Initialize
data.qpos[:] = ref_qpos[0]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

print('Frame | root_h_ref | root_h_sim | root_drift | joint_err | max_qvel | ncon')
print('-' * 95)

for t in range(min(T, 40)):
    # Set PD targets
    data.ctrl[:] = ref_qpos[t, 7:]

    # Step physics
    for _ in range(decimation):
        mujoco.mj_step(model, data)

    # Compute metrics
    root_h_ref = ref_qpos[t, 2]
    root_h_sim = data.qpos[2]
    root_drift = np.linalg.norm(data.qpos[:3] - ref_qpos[t, :3])
    joint_err = np.mean(np.abs(data.qpos[7:] - ref_qpos[t, 7:]))
    max_qvel = np.max(np.abs(data.qvel))
    ncon = data.ncon

    # Root orientation drift
    from scipy.spatial.transform import Rotation as sRot
    ref_quat = ref_qpos[t, 3:7][[1,2,3,0]]  # wxyz -> xyzw
    sim_quat = data.qpos[3:7][[1,2,3,0]]
    R_diff = sRot.from_quat(ref_quat).inv() * sRot.from_quat(sim_quat)
    rot_err_deg = np.degrees(R_diff.magnitude())

    flag = '***FALL***' if root_h_sim < 0.3 else ''
    print(f'{t:5d} | {root_h_ref:10.4f} | {root_h_sim:10.4f} | {root_drift:10.4f} | '
          f'{joint_err:9.4f} | {max_qvel:8.2f} | {ncon:4d}  rot_err={rot_err_deg:5.1f}deg {flag}')

    if root_h_sim < 0.3:
        # What's happening at fall
        print(f'  Root pos ref: {ref_qpos[t, :3]}')
        print(f'  Root pos sim: {data.qpos[:3]}')
        print(f'  Root vel: {data.qvel[:6]}')
        break
" 2>&1
