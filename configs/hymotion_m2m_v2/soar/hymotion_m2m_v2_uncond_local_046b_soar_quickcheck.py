# Quick SOAR verification on uncond_local — 150 iter, small bs, to verify:
#   1. load_from succeeds on real SFT checkpoint
#   2. loss_velocity and loss_soar_corr are finite
#   3. loss trends downward (research success criterion per user spec)
#
# Derived from hymotion_m2m_v2_uncond_local_046b_soar.py but shorter.

_base_ = './hymotion_m2m_v2_uncond_local_046b_soar.py'

work_dir = 'work_dirs/hymotion_m2m_v2_uncond_local_046b_soar_quickcheck'

train_dataloader = dict(
    batch_size=8,           # smaller for quick verification
    num_workers=2,
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=400,
    val_interval=100000,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=5),
    checkpoint=dict(type='CheckpointHook', interval=200, by_epoch=False,
                    max_keep_ckpts=1, save_last=False),
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)
