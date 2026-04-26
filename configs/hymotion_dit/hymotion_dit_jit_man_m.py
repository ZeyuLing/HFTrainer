# HyMotion DiT Medium (~162M) — JiT + Mask-Aware Noise.
#
# Text-free DiT with JiT loss (pred_type='x1') and mask-aware noise.
# Train from scratch, no pretrained weights.
#
# Launch:
#   python tools/train.py configs/hymotion_dit/hymotion_dit_jit_man_m.py
#   bash tools/dist_train.sh configs/hymotion_dit/hymotion_dit_jit_man_m.py 8

_base_ = './_base_hymotion_dit_m.py'

work_dir = 'work_dirs/hymotion_dit_jit_man_m'

model = dict(
    pred_type='x1',
    losses_cfg=dict(
        velocity_weight=1.0,
        x1_weight=1.0,
    ),
)

trainer = dict(
    mask_aware_noise=True,
)
