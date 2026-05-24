# PRISM 1B text-to-motion with KT-RoPE DFS mode
# Uses DFS traversal order to encode skeletal structure
#
# KT-RoPE DFS Advantages:
#   - Simpler than spectral mode: reindexes joints by DFS traversal
#   - Parent-child joints get adjacent indices (locality in joint space)
#   - Moderate correlation with kinematic tree distance (0.628 vs 0.397 for sequential)
#   - Good balance between structural awareness and computational simplicity

_base_ = './prism_1b_tp2m_1frame.py'

model = dict(
    transformer=dict(
        joint_pos_mode="dfs",  # KT-RoPE DFS mode
    ),
)
