"""Formal 2k continuation for HYMotion-G1 + HumanoidGPT reward from 130k base."""

_base_ = "verify_hymotion_g1_humanoidgpt_130k_safe.py"

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
        interval=250,
        max_keep_ckpts=12,
        save_last=True,
    ),
)
