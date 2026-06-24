#!/usr/bin/env bash
# Convenience wrapper for TABLE VII BABEL KT-RoPE ablation generation.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PHASE=${PHASE:-gen} \
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/babel_seq/ktmodes_20260616_e29} \
METHODS=${METHODS:-"kt_seq kt_dfs"} \
NUM_GPUS=${NUM_GPUS:-8} \
KT_CKPT=${KT_CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_29} \
KAFS_MODE=${KAFS_MODE:-none} \
bash scripts/eval/run_prism_babel_checkpoint_compare.sh
