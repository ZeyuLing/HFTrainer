# PhysFlow LEARNABILITY-FRONTIER co-evolution OVERFIT config (direction B).
#
# Purpose: validate the redesigned closed loop end-to-end on the tiny fixed
# prompt set before the formal run -- specifically that the per-judge Q/T split,
# the regret-max SFT target, the frontier pool export, and the kinematic quality
# gate all flow through the orchestrated GENERATOR -> JUDGE(Q+T) -> SFT -> pool
# -> TRAINEE -> JUDGE-SYNC loop without crashing and with sane telemetry
# (n_frontier_mean, sel_trainee_compl). Run with --judge-mode anchor so the
# trainee judge T enters the ensemble from round 1.

_base_ = 'physflow_coevo_overfit_g1.py'

work_dir = 'work_dirs/physflow_coevo_overfit_frontier_g1'

trainer = dict(
    frontier_mode=True,
    quality_judge='frozen',
    trainee_judge='trainee',
    sft_target='regret',
    frontier_t_low=0.2,
    frontier_t_high=0.9,
    accept_min_completion=0.9,
    accept_require_no_fall=True,
    accept_max_score=None,
    accept_min_joint_std=0.05,
    accept_max_root_disp_if_frozen=1.0,
    accept_frozen_joint_std=0.03,
    quality_max_joint_vel=30.0,
    quality_max_root_vel=8.0,
    gt_pool_freq=1,
)
