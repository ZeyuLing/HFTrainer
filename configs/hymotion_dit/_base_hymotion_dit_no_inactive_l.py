# HyMotion DiT Large — No-Inactive VACE ablation.
#
# input_dim = 135 + 2*135 = 405 (no inactive channel)
# Size: feat_dim=1024, num_layers=24, num_heads=16 → ~383M params

_base_ = './_base_hymotion_dit_no_inactive_s.py'

work_dir = 'work_dirs/hymotion_dit_no_inactive_fm_man_l'

model = dict(
    motion_transformer=dict(
        feat_dim=1024,
        num_layers=24,
        num_heads=16,
    ),
)

train_dataloader = dict(batch_size=24)
