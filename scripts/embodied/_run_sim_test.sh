#!/bin/bash
# Run SMPL physics sim on a single NPZ file
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

python3 scripts/embodied/run_smpl_physics_sim.py \
    --npz-file output/embodied_t2m_v4/data/npz/v4_walk_001.npz \
    --output-dir output/embodied_t2m_v4/data/smpl_mesh_physics \
    --xml-path ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --stats-dir output/embodied_t2m_v4/data/sim_stats \
    2>&1
