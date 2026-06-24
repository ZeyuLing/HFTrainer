# TABLE VIII (KT-RoPE ablation) — "DFS Reindexing" row.
# Same epoch_15 weights, joint_pos_mode toggled to "dfs" at inference
# (KT-RoPE is parameter-free; see prism_1b_tp2m_multiframe_kt_seqInfer.py).
_base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified.py'

model = dict(
    transformer=dict(
        joint_pos_mode="dfs",
    ),
)
