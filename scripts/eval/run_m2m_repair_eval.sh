#!/bin/bash
# Run M2M repair evaluation on Taiji GPU node.
# Usage: bash scripts/run_m2m_repair_eval.sh

set -e

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

echo "=== M2M Repair Evaluation ==="
echo "PWD: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi || echo "nvidia-smi not found"

# Run evaluation: 200 samples, both configs, both inpaint/edit modes
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_m2m_repair.py \
    --max-samples 200 \
    --num-steps 50 \
    --mogendit-steps 10 \
    --device cuda:0 \
    --mogendit-device cuda:0 \
    --configs uncond_fm uncond_fm_man \
    --output-dir output/m2m_repair_eval \
    2>&1 | tee output/m2m_repair_eval_log.txt

echo "=== Done ==="
echo "Results in: output/m2m_repair_eval/"
