#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 tools/eval_vermo_overfit_alltasks.py \
  --config configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr2e5_1gpu_skipguard.py \
  --checkpoint work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr2e5_from1250_1gpu_skipguard/checkpoint-iter_500 \
  --samples-per-task 1 \
  --mode teacher_forced \
  --output-json work_dirs/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage2_outputonly_lr2e5_from1250_1gpu_skipguard/eval_overfit_ckpt500_spt1_teacher_manual1gpu.json
