#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

python3 -m accelerate.commands.launch \
  --num_machines=1 \
  --num_processes=8 \
  --machine_rank=0 \
  --main_process_ip=127.0.0.1 \
  --main_process_port="${PORT:-29500}" \
  tools/train.py \
  configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks_stage3_all190_lr5e6_from_focusckpt150.py
