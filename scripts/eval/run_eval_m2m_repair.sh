#!/bin/bash
# Run M2M repair evaluation on a Taiji GPU node.
# Usage: This is the start_cmd for a Taiji task.

set -e

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/

# Install seaborn (needed by MoGenDiT import chain)
pip install -q seaborn 2>/dev/null || true

# Run the evaluation
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_m2m_repair.py \
    --max-samples 200 \
    --num-steps 50 \
    --mogendit-steps 10 \
    --device cuda:0 \
    --mogendit-device cuda:0 \
    --output-dir output/m2m_repair_eval_v2 \
    --configs uncond_fm uncond_fm_man \
    2>&1 | tee output/m2m_repair_eval_v2_log.txt

echo "===== EVAL COMPLETE ====="
# Keep alive for a bit so logs can be inspected
sleep 300
