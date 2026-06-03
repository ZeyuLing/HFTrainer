#!/bin/bash
# Continue the v2 GPU trainee run with a CORRECT step budget.
# Root-cause of the previous "stop at epoch 90": max_epochs = training_max_steps
# // num_envs // num_steps (num_steps=32). So 3,000,000 // 1024 // 32 = 91 epochs.
# This script warm-starts from the latest checkpoint and gives a budget large
# enough for ~3000 epochs (100M env-steps), saving a checkpoint every 100 epochs.
set -u

PROTO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions
HFT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
# NEW experiment name => warm_start mode (resolved_configs.pt of the old run baked
# training_max_steps=3M, which RESUME would reload and stop at epoch 91). A fresh
# name rebuilds the config from CLI (100M budget) while load_parameters() still
# restores epoch=91 + optimizer/disc/EMA state from the warm-start checkpoint.
EXP=physflow_online_g1_trainee_gpu2
RESULTS="$PROTO/results/$EXP"
SNAP="$HFT/work_dirs/physflow_online_adv_v2/trainee_gpu/pool_snapshot"
WARM="$PROTO/results/physflow_online_g1_trainee_gpu/last.ckpt"
PY=/root/physflow_isaacgym_py38_cu118/bin/python
LOG=$HFT/work_dirs/physflow_online_adv_v2/trainee_gpu/gpu_train_v3.log

for v in 14 13 12 11 10 9; do r=/opt/rh/gcc-toolset-$v/root/usr; if [ -d "$r/bin" ]; then export PATH="$r/bin:$PATH" CC="$r/bin/gcc" CXX="$r/bin/g++" LD_LIBRARY_PATH="$r/lib64:${LD_LIBRARY_PATH:-}"; break; fi; done
export PYTHONPATH="$PROTO:${PYTHONPATH:-}" ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=0

cd "$PROTO" || exit 1
if [ -f "$RESULTS/last.ckpt" ]; then CKPT="$RESULTS/last.ckpt"; else CKPT="$WARM"; fi

echo "=== $(date -Is) continue from $CKPT (budget 100M steps ~3000 epochs) ===" >> "$LOG"
nohup $PY protomotions/train_agent.py \
    --robot-name g1 --simulator isaacgym \
    --experiment-path examples/experiments/mimic/physflow_g1_xy_offset.py \
    --experiment-name "$EXP" \
    --motion-file "$SNAP" \
    --checkpoint "$CKPT" \
    --num-envs 1024 --batch-size 8192 \
    --training-max-steps 100000000 \
    --headless True --skip-initial-eval \
    --overrides agent.save_last_checkpoint_every=100 >> "$LOG" 2>&1 &
echo "LAUNCHED_PID=$!" | tee -a "$LOG"
