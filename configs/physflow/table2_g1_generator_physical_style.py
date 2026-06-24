# Table 2 generator row: tracker physical feedback + G1 robot-style reward.
#
# This row isolates whether the qpos-level style bank improves robot-style
# matching beyond physical executability.  Anti-degeneration replay stays off;
# the full row adds it separately.

_base_ = 'physflow_online_adv_g1_38dim.py'

work_dir = 'work_dirs/table2_g1_generator_physical_style'

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_g1_t2m_38dim/checkpoint-iter_339000',
    load_scope='model',
)

trainer = dict(
    type='PhysFlowG1Trainer',
    _delete_=True,
    num_samples=4,
    diffusion_steps=50,
    reward_weighted=False,
    enable_reward=True,
    style_reward_bank='data/g1_style_bank/train_minus_heldout_20k.npz',
    style_reward_weight=0.5,
    keep_rollouts=False,
    judge_backend='protomotions',
    accept_min_completion=0.9,
    accept_require_no_fall=True,
    accept_max_score=2.5,
    anchor_weight=0.0,
    gt_weight=0.0,
    export_gt_to_pool=False,
    gt_pool_freq=0,
    accept_min_joint_std=0.05,
    accept_max_root_disp_if_frozen=1.0,
    accept_frozen_joint_std=0.03,
    tracker_pool_dir='work_dirs/table2_g1_generator_physical_style/tracker_motion_pool',
    pool_max_motions=4000,
)
