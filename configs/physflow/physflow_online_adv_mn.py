# PhysFlow online-adversarial FORMAL run -- MULTI-NODE (data-parallel DDP).
#
# Same RAFT / best-of-N reward-SFT loop as physflow_online_adv_v3.py, scaled
# out across multiple 8x V100 nodes via accelerate multi-node DDP. Verified on
# a single 8-GPU node (work_dirs/physflow_8gpu_verify): all ranks light up,
# cross-rank gradient all-reduce does not hang, n_good aggregates across ranks,
# loss / sel_joint_std stay healthy, ~3.8x net throughput per node.
#
# Effective batch = num_nodes * 8 (one prompt per rank, accumulation=1). The
# per-node MuJoCo/convert scoring is CPU-bound and does NOT contend across
# nodes (each node uses its own 96 cores), so throughput scales ~linearly per
# node: 2 nodes ~7.6x, 4 nodes ~15x vs a single GPU.
#
# Launch is handled by tools/taiji_dist_train.sh (reads NODE_LIST/NODE_NUM/
# CHIEF_IP/INDEX from the Taiji platform). Submit via tools/taiji_submit.py
# with a custom start_cmd that bootstraps the MuJoCo/convert deps onto each
# FRESH vermo container (mujoco onnxruntime dm_control typer) and points the
# csv->.motion converter at the container python (PHYSFLOW_CONVERT_PYTHON).

_base_ = './physflow_online_adv_v3.py'

work_dir = 'work_dirs/physflow_online_adv_mn'

# Continue optimizing from the finished v3 generator (checkpoint-iter_3000):
# load weights only (load_scope='model' -> fresh optimizer + step counter), so
# this is a NEW run that picks up where v3 left off at larger scale. On the
# first launch work_dir is empty so --auto-resume finds nothing and falls back
# to this load_from; on any re-schedule/restart it full-resumes the mn run's
# own latest checkpoint instead.
load_from = dict(
    _delete_=True,
    path='work_dirs/physflow_online_adv_v3/checkpoint-iter_3000',
    load_scope='model',
)

# Keep the multi-node accepted-motion pool separate from the v3 pool.
trainer = dict(
    tracker_pool_dir='work_dirs/physflow_online_adv_mn/tracker_motion_pool',
)

# Step budget. With H nodes (8H ranks, accumulation=1) each step covers 8H
# prompts; 1500 steps @ 2 nodes = 24000 prompt-draws (~8x the v3 single-GPU
# 3000-step run). Adjust to taste; auto-resume makes it safe to extend later.
train_cfg = dict(
    by_epoch=False,
    max_iters=1500,
    val_interval=999999,
    max_grad_norm=1.0,
)

# Per-step optimizer update (accumulation=1); effective batch = 8 * num_nodes.
# lr kept at the v3 value (5e-6, set in the base config); the larger batch is
# still modest (16-32) and RAFT is lr-sensitive, so we do not scale it up
# aggressively. Bump later if convergence looks too slow.
accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', interval=200, max_keep_ckpts=8, save_last=True),
)
