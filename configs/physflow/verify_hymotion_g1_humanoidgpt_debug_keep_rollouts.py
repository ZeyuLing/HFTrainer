"""Short HumanoidGPT debug continuation that preserves per-step rollout dirs."""

_base_ = "verify_hymotion_g1_humanoidgpt_continue2k.py"

trainer = dict(
    keep_rollouts=True,
    rollout_dir="work_dirs/physflow_verify_hymotion_g1_humanoidgpt_headless/hgpt_debug_rollouts",
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=420,
    val_interval=999999,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=1),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=500,
        max_keep_ckpts=8,
        save_last=False,
    ),
)
