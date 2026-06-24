# FULL closed-loop smoke for the G1-native PhysFlow online loop, WITH the frozen
# MuJoCo+ONNX judge.  Proves the whole chain executes end-to-end:
#   sample (flow-matching) -> decode_g1_to_qpos -> qpos CSV
#   -> convert_g1_csv_to_proto (py38) -> .motion
#   -> MuJoCo + frozen g1-bones-deploy ONNX rollout -> adversarial score
#   -> accept filter -> reward-filtered FM SFT + anchor
#   -> accepted motions exported to tracker_motion_pool (gen->trainee).
# Accept thresholds are permissive so the early generator's motions still flow
# through the pool-export path (we are validating plumbing, not motion quality).

_base_ = 'physflow_online_adv_g1_38dim.py'

work_dir = 'work_dirs/physflow_online_adv_g1_38dim_judge_smoke'

trainer = dict(
    type='PhysFlowG1Trainer',
    _delete_=True,
    num_samples=2,
    diffusion_steps=10,
    enable_reward=True,
    judge_backend='protomotions',
    keep_rollouts=True,           # keep rollouts for inspection
    anchor_weight=1.0,
    # permissive accept so the pool-export path is exercised
    accept_min_completion=0.0,
    accept_require_no_fall=False,
    accept_max_score=None,
    accept_min_joint_std=0.0,
    accept_max_root_disp_if_frozen=None,
    tracker_pool_dir='work_dirs/physflow_online_adv_g1_38dim_judge_smoke/tracker_motion_pool',
    pool_max_motions=100,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        anno_file='data/annotation/train_g1_t2m_overfit100.json',
        random_caption=False,
    ),
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=2,
    val_interval=999999,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=999999, save_last=False),
)
