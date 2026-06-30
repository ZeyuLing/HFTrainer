"""Table 2 generator row: online update with a Humanoid-GPT judge.

This is the runnable generator + Humanoid-GPT arm.  It keeps the G1-native
PhysFlow generator/trainer stack, replaces the frozen ProtoMotions judge with
Humanoid-GPT, and writes accepted qpos motions for later replay diagnostics.
"""

_base_ = "physflow_online_adv_g1_38dim.py"

work_dir = "work_dirs/table2_g1_generator_humanoidgpt"

# The scene-clean base checkpoint has not been materialized in this checkout yet.
# Use the latest available G1-native supervised base so this arm is executable now;
# switch to work_dirs/hymotion_g1_t2m_38dim_scene_clean_minus_heldout/checkpoint-g1base
# once that clean base exists.
load_from = dict(
    _delete_=True,
    path="work_dirs/hymotion_g1_t2m_38dim/checkpoint-iter_339000",
    load_scope="model",
)

trainer = dict(
    judge_backend="hgpt",
    hgpt_freq=50,
    hgpt_input_fps=30,
    num_samples=4,
    diffusion_steps=30,
    anchor_weight=1.5,
    gt_weight=0.75,
    style_reward_bank=None,
    style_reward_weight=0.0,
    tracker_pool_dir=None,
    tracker_qpos_pool_dir="work_dirs/table2_g1_generator_humanoidgpt/qpos_pool",
    tracker_qpos_pool_fps=30.0,
    export_gt_to_pool=True,
    gt_pool_freq=2,
    gt_pool_accept_mode="kinematic",
    pool_max_motions=8000,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        anno_file="data/annotation/train_g1_t2m_emb_minus_heldout_scene_clean.json",
        max_items=None,
    ),
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
        interval=150,
        max_keep_ckpts=8,
        save_last=True,
    ),
)
