"""Formal relative-trackability rerun for HYMotion-G1 + ProtoMotions reward.

The 0618 relgate run proved that all-metric margins are too sparse: many
same-noise candidates were hard-trackable, but none survived the simultaneous
score/joint/root margin gates, so reward SFT often had zero signal. This run
keeps the same-noise base comparison, but accepts candidates by weighted net
trackability advantage and weights SFT by that advantage.
"""

_base_ = "verify_hymotion_g1_protomotions_130k_safe.py"

auto_resume = True
work_dir = "work_dirs/physflow_verify_hymotion_g1_protomotions_130k_reltrack_0619"

trainer = dict(
    judge_backend="protomotions",
    num_samples=8,
    diffusion_steps=30,
    anchor_weight=1.5,
    gt_weight=0.75,
    tracker_pool_dir=None,
    export_gt_to_pool=False,
    # Absolute gates remain safety filters, but they are deliberately looser than
    # the final held-out pass/fail definition. The relative advantage below is
    # the actual training signal.
    accept_min_completion=0.90,
    accept_max_score=2.60,
    accept_max_joint_error_rad=0.95,
    accept_max_root_trajectory_error_mean_m=0.60,
    accept_max_root_displacement_error_m=1.25,
    accept_require_root_metrics=True,
    accept_soft_fallback=False,
    accept_min_joint_std=0.055,
    accept_max_root_disp_if_frozen=1.0,
    accept_frozen_joint_std=0.03,
    relative_to_base=True,
    relative_mode="advantage",
    relative_min_advantage=0.02,
    relative_score_weight=1.0,
    relative_joint_weight=0.75,
    relative_root_trajectory_weight=2.0,
    relative_root_displacement_weight=0.35,
    relative_completion_weight=1.0,
    relative_fall_weight=2.0,
    relative_max_completion_drop=0.02,
    relative_require_no_fall_regression=True,
    relative_select_by_advantage=True,
    relative_weight_by_advantage=True,
    relative_advantage_weight_scale=0.75,
    relative_advantage_weight_max=2.5,
)

load_from = dict(
    _delete_=True,
    path="work_dirs/physflow_verify_hymotion_g1_base/checkpoint-iter_130000",
    load_scope="model",
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=6000,
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
