# Smoke for the G1-native PhysFlow online loop WITHOUT the heavy judge env.
# enable_reward=False -> _score_samples returns score=0 (no CSV->.motion->MuJoCo
# convert), so this validates the new sample/decode/CSV/sft code paths and the
# PhysFlowG1Trainer wiring end-to-end on a single GPU in the generator's env.

_base_ = 'physflow_online_adv_g1_38dim.py'

work_dir = 'work_dirs/physflow_online_adv_g1_38dim_smoke'

trainer = dict(
    type='PhysFlowG1Trainer',
    _delete_=True,
    num_samples=2,
    diffusion_steps=10,
    enable_reward=False,          # <-- skip judge (no py38/MuJoCo needed)
    keep_rollouts=False,
    anchor_weight=1.0,
    accept_min_completion=0.0,
    accept_require_no_fall=False,
    accept_min_joint_std=0.0,
    tracker_pool_dir=None,        # no pool export in smoke
)

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    dataset=dict(
        anno_file='data/annotation/train_g1_t2m_overfit100.json',
        random_caption=False,
    ),
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=3,
    val_interval=999999,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=999999, save_last=False),
)
