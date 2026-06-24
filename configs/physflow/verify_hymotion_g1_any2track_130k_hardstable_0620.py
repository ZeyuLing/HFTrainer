"""Any2Track hard-valid stability bootstrap for PhysFlow.

The soft-fallback Any2Track rerun produced gradients, but most late-step
updates still came from candidates that failed the hard tracker gates. This
branch removes soft fallback from the objective and uses no-fall/full-completion
as the early hard validity test, while same-noise relative advantage still
chooses and weights the SFT target.
"""

_base_ = "verify_hymotion_g1_any2track_130k_relboot_0619.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_any2track_130k_hardstable_0620"
auto_resume = True

trainer = dict(
    # Do not train on fallen or non-accepted samples. The stopped fallback branch
    # proved that relative-improvement-only fallback can lower loss without
    # producing a clearly more trackable generator.
    accept_soft_fallback=False,
    accept_soft_fallback_require_relative=False,
    # Avoid starving the run on early Any2Track scalar/root gates. A candidate
    # must still complete and not fall, and the same-noise base comparison below
    # still decides whether it is an improvement.
    accept_min_completion=0.95,
    accept_max_score=4.0,
    accept_max_joint_error_rad=1.25,
    accept_require_root_metrics=False,
    accept_max_root_trajectory_error_mean_m=None,
    accept_max_root_displacement_error_m=None,
    accept_min_joint_std=0.055,
    # Make the accepted target visibly prefer tracker-success: score/joint and
    # completion/no-fall dominate, root displacement is only a weak diagnostic
    # because Any2Track-root gates were starving otherwise valid motions.
    relative_min_advantage=0.02,
    relative_score_weight=1.5,
    relative_joint_weight=1.25,
    relative_root_trajectory_weight=0.25,
    relative_root_displacement_weight=0.0,
    relative_completion_weight=2.0,
    relative_fall_weight=4.0,
    relative_select_by_advantage=True,
    relative_weight_by_advantage=True,
    relative_advantage_weight_scale=2.0,
    relative_advantage_weight_max=4.0,
    # Let successful tracker-reward samples move the model more than the old
    # conservative branch, while keeping real-data SFT as a stabilizer.
    anchor_weight=0.75,
    gt_weight=1.0,
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
