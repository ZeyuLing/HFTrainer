#!/bin/bash
# Run keypose guidance evaluation on Taiji debug machine.
# Usage: ssh into debug machine, then run this script.

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

echo "=== Keypose Guidance Evaluation ==="
echo "Starting at $(date)"

# Install deps if needed
pip install flask scipy 2>/dev/null || true

# Run evaluation with all 5 models
# Split across GPUs if available
export CUDA_VISIBLE_DEVICES=0

echo "=== Phase 1: M2M Models (GPU 0) ==="
python3 scripts/eval_keypose_guidance.py \
    --models uncond_fm_man uncond_jit_man uncond_fm_man_globalrot uncond_jit_man_globalrot \
    --max-samples 50 \
    --num-steps 50 \
    --device cuda:0 \
    2>&1 | tee output/keypose_eval/m2m_eval.log

echo "=== Phase 2: MoGenDIT (GPU 0) ==="
python3 scripts/eval_keypose_guidance.py \
    --models mogendit \
    --max-samples 50 \
    --num-steps 50 \
    --device cuda:0 \
    --mogendit-device cuda:0 \
    2>&1 | tee output/keypose_eval/mogendit_eval.log

echo "=== Done at $(date) ==="
