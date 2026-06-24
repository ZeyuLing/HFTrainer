_base_ = "verify_hymotion_g1_humanoidgpt_130k_relboot_0619.py"

work_dir = "work_dirs/physflow_verify_hymotion_g1_humanoidgpt_130k_relboot_rootwide_0619"
auto_resume = True

trainer = dict(
    # HumanoidGPT judge reports useful low score candidates early, but the strict
    # root displacement hard gate rejects all of them before the relative
    # advantage gate can choose among candidates. Keep fall/completion/score/joint
    # gates, and let same-noise advantage penalize root regressions during
    # bootstrap.
    accept_max_root_trajectory_error_mean_m=1.0,
    accept_max_root_displacement_error_m=1.5,
    relative_min_advantage=-0.01,
)
