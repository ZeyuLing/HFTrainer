"""Strict trackability rerun for HYMotion-G1 + ProtoMotions reward.

The proto2k run lowered the scalar reward a little but did not clearly improve
heldout trackability. This config makes the online SFT target pass the same
direct gates used by the final evaluation: no fall, high completion, bounded
joint tracking error, and bounded root tracking error. If this run cannot raise
trackable_basic_rate / lower root error, it should be reported as ineffective.
"""

_base_ = "verify_hymotion_g1_protomotions_130k_safe.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_protomotions_130k_trackstrict_0618"

trainer = dict(
    judge_backend="protomotions",
    num_samples=8,
    diffusion_steps=30,
    anchor_weight=2.0,
    gt_weight=1.0,
    tracker_pool_dir=None,
    export_gt_to_pool=False,
    accept_min_completion=0.95,
    accept_max_score=1.7,
    accept_max_joint_error_rad=0.7,
    accept_max_root_trajectory_error_mean_m=0.25,
    accept_max_root_displacement_error_m=0.35,
    accept_require_root_metrics=True,
    accept_min_joint_std=0.05,
    accept_max_root_disp_if_frozen=1.0,
    accept_frozen_joint_std=0.03,
)

load_from = dict(
    _delete_=True,
    path="work_dirs/physflow_verify_hymotion_g1_base/checkpoint-iter_130000",
    load_scope="model",
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=3000,
    val_interval=999999,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=10),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=250,
        max_keep_ckpts=8,
        save_last=True,
    ),
)
