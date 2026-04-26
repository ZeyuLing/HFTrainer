# HyMotion DiT Base (~288M) — No-Inactive VACE + FM + MAN.
#
# Ablation: VACE = cat([reactive, mask]) only, no inactive channel.
#
# Taiji (2 nodes, 16 GPU):
#   python3 tools/taiji_submit.py dit_noinact_fm_man_b configs/hymotion_dit/hymotion_dit_no_inactive_fm_man_b.py --host_num 2

_base_ = './_base_hymotion_dit_no_inactive_b.py'

work_dir = 'work_dirs/hymotion_dit_no_inactive_fm_man_b'

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
