_base_ = "verify_hymotion_g1_humanoidgpt_130k_relboot_rootwide_fallback_0619.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_humanoidgpt_130k_relboot_posefallback_0619"
auto_resume = True

trainer = dict(
    # HumanoidGPT motions are often local or nearly in-place. The prior HGPT
    # relboot branches were starved by root-displacement hard gates, so this
    # branch tests whether the generator can improve when acceptance is based on
    # pose/joint trackability plus same-noise relative advantage.
    accept_require_root_metrics=False,
    accept_max_root_trajectory_error_mean_m=None,
    accept_max_root_displacement_error_m=None,
    accept_soft_fallback=True,
    accept_soft_fallback_require_relative=True,
    relative_mode="advantage",
    relative_min_advantage=-0.01,
    relative_score_weight=1.0,
    relative_joint_weight=1.25,
    relative_root_trajectory_weight=0.25,
    relative_root_displacement_weight=0.0,
    relative_select_by_advantage=True,
    relative_weight_by_advantage=True,
    relative_advantage_weight_scale=1.5,
    relative_advantage_weight_max=3.0,
)
