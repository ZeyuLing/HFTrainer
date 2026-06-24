# TABLE VIII (KT-RoPE ablation) — "Sequential RoPE (no-KT)" row.
#
# KT-RoPE is parameter-free: joint_pos_mode only changes the position values fed
# to Q/K (joint_freqs are persistent=False buffers, recomputed at init, NOT in
# the checkpoint). So the LATEST checkpoint (kt_spectral_unified epoch_15) can be
# run in "sequential" (no-KT) inference mode by toggling joint_pos_mode here and
# loading the epoch_15 weights via --checkpoint. This yields the "no-KT version
# of the latest checkpoint" for the ablation + the t2m_compare viewer column.
_base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified.py'

model = dict(
    transformer=dict(
        joint_pos_mode="sequential",
    ),
)
