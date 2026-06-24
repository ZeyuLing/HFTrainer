# Smoke variant of the G1-native T2M fine-tune: tiny subset + few iters to
# validate the full code path (dataset -> 38-d target -> warm-start load ->
# flow-matching step) before launching the real run on Taiji.

_base_ = 'hymotion_g1_t2m_38dim.py'

work_dir = 'work_dirs/hymotion_g1_t2m_38dim_smoke'

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(max_items=64),
)

train_cfg = dict(
    by_epoch=False,
    max_iters=3,
    val_interval=100000,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(type='CheckpointHook', interval=100000, save_last=False),
)
