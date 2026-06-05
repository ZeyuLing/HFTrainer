# PhysFlow 8-GPU DDP *verification* run (NOT a formal training).
#
# Purpose: before committing to a long multi-GPU run, confirm on a single
# 8x V100 node that data-parallel DDP actually works for the PhysFlow online
# best-of-N reward-SFT loop, i.e.:
#   1. all 8 ranks light up (one prompt per rank per step -> 8x prompt
#      throughput at roughly the same ~12s/step wall time as 1 GPU);
#   2. cross-rank gradient all-reduce does not hang (every step must produce
#      gradients for ALL trainable params -- anchor_weight=1.0 guarantees a
#      real backward even when no candidate is accepted);
#   3. the loss / sel_joint_std stay sane (no collapse, no NaN), confirming
#      the larger effective batch (8 prompts/step vs 1) trains normally;
#   4. multi-rank checkpoint save/load round-trips.
#
# This inherits the v3 formal config and only overrides what the verify needs.
# It writes to its OWN work_dir + tracker pool so it never touches the finished
# v3 run (work_dirs/physflow_online_adv_v3/checkpoint-iter_3000).
#
# Launch (inside the 8x V100 container), all 8 GPUs visible (do NOT set
# CUDA_VISIBLE_DEVICES), with CPU threads capped so 8 ranks don't oversubscribe
# the 96 cores when each runs MuJoCo/ONNX scoring in parallel:
#   OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
#   accelerate launch --num_processes 8 --num_machines 1 \
#       --mixed_precision no --main_process_port 29555 \
#       tools/train.py configs/physflow/physflow_8gpu_verify.py

_base_ = './physflow_online_adv_v3.py'

work_dir = 'work_dirs/physflow_8gpu_verify'

# Start fresh from the base KIMODO-G1 generator (no resume). The bundle loads
# the pretrained KIMODO weights in from_config; auto-resume finds nothing in
# this fresh work_dir.
load_from = None

# Keep the verify's accepted-motion pool separate from the v3 pool.
trainer = dict(
    tracker_pool_dir='work_dirs/physflow_8gpu_verify/tracker_motion_pool',
)

# Short run: enough steps to measure steady-state per-step time after warmup.
train_cfg = dict(
    by_epoch=False,
    max_iters=20,
    val_interval=999999,
    max_grad_norm=1.0,
)

# Per-step optimizer update (accumulation=1) so DDP gradient sync is exercised
# every step. Effective batch becomes num_processes (8 prompts/step).
accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)

# Log every step; save one checkpoint at the end to verify DDP save/load.
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=1),
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=1, save_last=True),
)
