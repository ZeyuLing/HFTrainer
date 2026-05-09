_base_ = './hymotion_m2m_v2_uncond_local_046b_component_mean.py'

work_dir = 'work_dirs/hymotion_m2m_v2_component_mean_debug'

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=80,
    val_interval=10_000,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=None,
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)
