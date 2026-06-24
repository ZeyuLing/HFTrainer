"""HYMotion-G1 generator + ProtoMotions reward from the frozen 130k base.

This is the controlled rerun after the ckpt99000 pilot: all tracker-reward arms
start from the same hard-linked base snapshot and save every 100 iters so the
best frozen-eval checkpoint can be selected instead of trusting the last step.
"""

_base_ = "verify_hymotion_g1_protomotions.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_protomotions_130k_safe"

trainer = dict(
    judge_backend="protomotions",
    num_samples=4,
    diffusion_steps=30,
    anchor_weight=1.5,
    gt_weight=0.75,
    tracker_pool_dir=None,
    export_gt_to_pool=False,
)

load_from = dict(
    _delete_=True,
    path="work_dirs/physflow_verify_hymotion_g1_base/checkpoint-iter_130000",
    load_scope="model",
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=600,
    val_interval=999999,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=10),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=100,
        max_keep_ckpts=8,
        save_last=True,
    ),
)
