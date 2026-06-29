# Table 2 generator row: tracker physical feedback only.
#
# This is the clean causal row for "SFT + tracker physical feedback": it starts
# from the same G1-native base generator and uses only frozen-tracker physical
# reward to select online SFT targets.  Robot-style reward and anti-degeneration
# replay are disabled here; those belong to later Table 2 rows.

_base_ = 'physflow_online_adv_g1_38dim.py'

work_dir = 'work_dirs/table2_g1_generator_physical'

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_g1_t2m_38dim_scene_clean_minus_heldout/checkpoint-g1base',
    load_scope='model',
)

trainer = dict(
    type='PhysFlowG1Trainer',
    _delete_=True,
    num_samples=4,
    diffusion_steps=50,
    reward_weighted=False,
    enable_reward=True,
    style_reward_bank=None,
    style_reward_weight=0.0,
    keep_rollouts=False,
    judge_backend='protomotions',
    accept_min_completion=0.9,
    accept_require_no_fall=True,
    accept_max_score=2.5,
    # Disable anti-degeneration replay/anchors for the physical-only row.
    anchor_weight=0.0,
    gt_weight=0.0,
    export_gt_to_pool=False,
    gt_pool_freq=0,
    accept_min_joint_std=0.05,
    accept_max_root_disp_if_frozen=1.0,
    accept_frozen_joint_std=0.03,
    tracker_pool_dir='work_dirs/table2_g1_generator_physical/tracker_motion_pool',
    pool_max_motions=4000,
)
