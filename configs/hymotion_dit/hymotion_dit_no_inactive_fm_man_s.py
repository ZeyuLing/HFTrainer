# HyMotion DiT Small (~49M) — No-Inactive VACE + FM + MAN.
#
# Ablation: VACE = cat([reactive, mask]) only, no inactive channel.
# With MAN, x_t[known] = clean, so inactive is redundant.
#
# Launch (8 GPU):
#   bash tools/dist_train.sh configs/hymotion_dit/hymotion_dit_no_inactive_fm_man_s.py 8
# Taiji (2 nodes, 16 GPU):
#   python3 tools/taiji_submit.py dit_noinact_fm_man_s configs/hymotion_dit/hymotion_dit_no_inactive_fm_man_s.py --host_num 2

_base_ = './_base_hymotion_dit_no_inactive_s.py'

work_dir = 'work_dirs/hymotion_dit_no_inactive_fm_man_s'

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
