#!/bin/bash
# Periodic eval watcher for PhysFlow online-adversarial v2 (GPU1).
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export HF_HOME=checkpoints/kimodo
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
exec python3 scripts/embodied/physflow_periodic_eval.py \
    --config configs/physflow/physflow_online_adv_v2.py \
    --num-prompts 48 --split test --watch --poll-sec 120
