"""Relative-to-base Any2Track rerun for HYMotion-G1.

The earlier Any2Track reward run learned from absolute trackability only; heldout
metrics improved only weakly and did not produce a clear generator improvement.
This rerun only accepts a same-noise candidate when it beats the frozen 130k
generator under the Any2Track judge, preventing reward-SFT from reinforcing
"trackable but not better than base" samples.
"""

_base_ = "verify_hymotion_g1_any2track_130k_safe.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_any2track_130k_relgate_0618"

trainer = dict(
    judge_backend="any2track",
    any2track_input_fps=30,
    any2track_max_steps=300,
    num_samples=8,
    diffusion_steps=30,
    anchor_weight=1.5,
    gt_weight=0.75,
    tracker_pool_dir=None,
    export_gt_to_pool=False,
    accept_min_completion=0.95,
    accept_max_score=1.8,
    accept_max_joint_error_rad=0.75,
    accept_max_root_trajectory_error_mean_m=0.30,
    accept_max_root_displacement_error_m=0.40,
    accept_require_root_metrics=True,
    accept_soft_fallback=False,
    accept_min_joint_std=0.05,
    accept_max_root_disp_if_frozen=1.0,
    accept_frozen_joint_std=0.03,
    relative_to_base=True,
    relative_min_score_improvement=0.15,
    relative_min_joint_error_improvement=0.02,
    relative_min_root_trajectory_improvement=0.02,
    relative_min_root_displacement_improvement=0.0,
    relative_max_completion_drop=0.02,
    relative_require_no_fall_regression=True,
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
