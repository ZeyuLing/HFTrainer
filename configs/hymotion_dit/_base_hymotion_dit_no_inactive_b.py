# HyMotion DiT Base — No-Inactive VACE ablation.
#
# input_dim = 135 + 2*135 = 405 (no inactive channel)
# Size: feat_dim=1024, num_layers=18, num_heads=16 → ~288M params

_base_ = './_base_hymotion_dit_no_inactive_s.py'

work_dir = 'work_dirs/hymotion_dit_no_inactive_b'

model = dict(
    motion_transformer=dict(
        feat_dim=1024,
        num_layers=18,
        num_heads=16,
    ),
)
