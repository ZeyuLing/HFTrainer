# HyMotion M2M v2 — padding fix verification run.
#
# Purpose: confirm that after fixing `tgt_length/src_length` in
# `PrepareM2Mv2Condition` to use the real pre-pad `num_frames`, training
# still runs end-to-end and loss behaves normally (no NaN, velocity loss
# trending down within the first few hundred iters).
#
# Inherits `hymotion_m2m_v2_uncond_local_046b.py` (smallest fast config).
# Overrides:
#   - work_dir → dedicated verify dir (does not collide with production runs)
#   - by_epoch=False, max_iters=500 (short run, ~1 hour on 1x8 V100)
#   - LoggerHook.iter_interval=1 (log every step for quick diagnosis)
#   - CheckpointHook disabled by_epoch, interval=200 iters (just a few ckpts)
#
# Launch (Taiji, 1 host × 8 V100):
#   python3 tools/taiji_submit.py m2m_v2_padfix_verify \
#       configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_padfix_verify.py \
#       --host_num 1

_base_ = './hymotion_m2m_v2_uncond_local_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_padfix_verify'

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=500,
    val_interval=10_000,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=200,
        max_keep_ckpts=2,
        save_last=True,
    ),
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)
