# PRISM 1B text+pose-to-motion, multi-frame conditioning + KT-RoPE DFS
#
# DFS reindexing maps each joint to its DFS traversal position in the SMPL-22
# kinematic tree. This ensures parent-child joints get adjacent indices.
# Simpler ablation baseline for spectral KT-RoPE.
# Correlation with tree distance: 0.628 vs 0.397 (sequential).
#
# Resume training from sequential checkpoint:
#   bash tools/taiji_dist_train.sh configs/prism/prism_1b_tp2m_multiframe_kt_dfs.py --auto-resume

_base_ = './prism_1b_tp2m_multiframe.py'

model = dict(
    transformer=dict(
        joint_pos_mode="dfs",  # KT-RoPE DFS mode
    ),
)
