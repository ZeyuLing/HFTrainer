"""Continue HYMotion-G1 + Any2Track reward training to a comparable stage."""

_base_ = "verify_hymotion_g1_any2track.py"

auto_resume = True

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=2000,
    val_interval=999999,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=10),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=500,
        max_keep_ckpts=8,
        save_last=True,
    ),
)
