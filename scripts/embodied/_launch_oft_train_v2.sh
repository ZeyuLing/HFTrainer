#!/bin/bash
# Launch PhysFlow online-adversarial v2 (collapse-fixed) training in tmux on GPU2.
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
mkdir -p work_dirs/physflow_online_adv_v2
export HF_HOME=checkpoints/kimodo
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2
exec python3 tools/train.py configs/physflow/physflow_online_adv_v2.py
