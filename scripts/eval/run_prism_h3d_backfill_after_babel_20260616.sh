#!/usr/bin/env bash
# Backfill missing TABLE VI/VII H3D generations on an idle debug node.
# Intended to be launched while BABEL kt-mode generation is running; it waits
# for that job to finish, then fills missing H3D DFS and depth-driven samples.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

while pgrep -f "gen_prism_babel_seq.py|run_prism_babel_ktmodes_20260616|run_prism_babel_checkpoint_compare.sh" >/dev/null 2>&1; do
  echo "[backfill] $(date) waiting for BABEL generation to finish..."
  sleep 60
done

COMMON_CKPT=work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_29

echo "[backfill] $(date) start H3D depth-driven backfill"
NGPU=8 TOTAL_SHARDS=8 SHARD_BASE=0 \
CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py \
CKPT="$COMMON_CKPT" \
KAFS_MODE=depth_driven \
OUT=outputs/evaluation/prism_ablation_tables_20260616_e29/h3d_kafs \
OUT_SUBDIR=depth_driven \
bash scripts/eval/run_prism_genonly_param.sh

echo "[backfill] $(date) start H3D DFS backfill"
NGPU=8 TOTAL_SHARDS=8 SHARD_BASE=0 \
CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_dfsInfer.py \
CKPT="$COMMON_CKPT" \
KAFS_MODE=depth_driven \
OUT=outputs/evaluation/prism_ablation_tables_20260616_e29/h3d_kt \
OUT_SUBDIR=dfs_depth \
bash scripts/eval/run_prism_genonly_param.sh

echo "[backfill] $(date) done"
