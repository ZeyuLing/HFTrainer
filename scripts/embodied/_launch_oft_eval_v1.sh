#!/usr/bin/env bash
# PhysFlow periodic-eval watcher launcher (run inside tmux on node).
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export HF_HOME=checkpoints/kimodo
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${OFT_EVAL_GPU:-1}
echo "[eval-launch] $(date) GPU=$CUDA_VISIBLE_DEVICES host=$(hostname)"
python3 scripts/embodied/physflow_periodic_eval.py \
  --config configs/physflow/physflow_online_adv_v1.py \
  --num-prompts ${OFT_EVAL_N:-48} \
  --gen-batch 8 \
  --watch --poll-sec 120
echo "[eval-launch] watcher exited code=$? $(date)"
