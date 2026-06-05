#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,1,7}"
export PORT="${PORT:-29637}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export VERMO_DEBUG_TIMING="${VERMO_DEBUG_TIMING:-1}"
export VERMO_DEBUG_ALL_RANKS="${VERMO_DEBUG_ALL_RANKS:-1}"
export VERMO_DEBUG_TIMING_STEPS="${VERMO_DEBUG_TIMING_STEPS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python3 -c 'from mmengine.config import Config; Config.fromfile("configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_18tasks.py")'
bash tools/taiji_dist_train.sh configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_mainpipeline_18tasks.py 4
