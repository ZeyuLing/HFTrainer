#!/usr/bin/env bash
# PhysFlow online-adversarial FORMAL training launcher (run inside tmux on node).
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export HF_HOME=checkpoints/kimodo
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${OFT_GPU:-0}
mkdir -p work_dirs/physflow_online_adv_v1
echo "[launch] $(date) GPU=$CUDA_VISIBLE_DEVICES host=$(hostname)"
python3 tools/train.py configs/physflow/physflow_online_adv_v1.py
echo "[launch] training exited code=$? $(date)"
