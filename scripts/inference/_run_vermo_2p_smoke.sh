#!/usr/bin/env bash
set -euo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:${PYTHONPATH:-}
python3 scripts/inference/batch_infer_vermo.py --models llama1b --tasks t2m_2p --num-samples 3 --output-dir work_dirs/vermo_eval_2p_smoke
