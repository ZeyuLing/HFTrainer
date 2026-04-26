# HyMotion DiT Small (~49M) — FM + Mask-Aware Noise.
#
# Text-free DiT with flow matching (pred_type='velocity') and mask-aware noise.
# Train from scratch, no pretrained weights.
#
# Launch:
#   python tools/train.py configs/hymotion_dit/hymotion_dit_fm_man_s.py
#   bash tools/dist_train.sh configs/hymotion_dit/hymotion_dit_fm_man_s.py 8

_base_ = './_base_hymotion_dit_s.py'

work_dir = 'work_dirs/hymotion_dit_fm_man_s'

model = dict(
    pred_type='velocity',
    losses_cfg=dict(
        velocity_weight=1.0,
        x1_weight=0.0,
    ),
)

trainer = dict(
    mask_aware_noise=True,
)
