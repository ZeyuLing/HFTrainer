"""HYMotion-G1 generator + Any2Track closed-loop co-evolution config.

The outer loop updates the Any2Track ONNX between rounds; this config is the
generator half. Accepted generated/GT qpos trajectories are exported as NPZ so
the Any2Track DAgger stack can train on the same replay pool.
"""

_base_ = "verify_hymotion_g1_any2track_130k_safe.py"

work_dir = "work_dirs/physflow_coevo_any2track_g1"

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(max_items=None),
)

trainer = dict(
    judge_backend="any2track",
    any2track_input_fps=30,
    any2track_max_steps=300,
    num_samples=4,
    diffusion_steps=30,
    anchor_weight=1.5,
    gt_weight=0.75,
    frontier_mode=True,
    sft_target="easiest",
    gt_pool_accept_mode="kinematic",
    tracker_pool_dir=None,
    tracker_qpos_pool_dir="work_dirs/physflow_coevo_any2track_g1/qpos_pool",
    tracker_qpos_pool_fps=30.0,
    export_gt_to_pool=True,
    gt_pool_freq=2,
    pool_max_motions=8000,
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=120,
    val_interval=999999,
    max_grad_norm=1.0,
)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=10),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=120,
        max_keep_ckpts=4,
        save_last=True,
    ),
)
