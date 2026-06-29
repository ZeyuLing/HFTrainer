# Table 2 generator row: full GenTrack generator update.
#
# Full = tracker physical feedback + G1 robot-style reward +
# anti-degeneration replay/anchor.  It keeps the production safeguards from the
# G1 online PhysFlow config while writing to a Table 2-specific work directory.

_base_ = 'physflow_online_adv_g1_38dim.py'

work_dir = 'work_dirs/table2_g1_generator_full'

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_g1_t2m_38dim_scene_clean_minus_heldout/checkpoint-g1base',
    load_scope='model',
)

trainer = dict(
    style_reward_bank='data/g1_style_bank/train_minus_heldout_scene_clean_20k.npz',
    style_reward_weight=0.5,
    tracker_pool_dir='work_dirs/table2_g1_generator_full/tracker_motion_pool',
)
