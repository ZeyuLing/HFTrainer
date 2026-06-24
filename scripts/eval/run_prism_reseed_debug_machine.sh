#!/usr/bin/env bash
# Run one PRISM reseed job on an already-running debug machine.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

SEED=${SEED:?set SEED}
BASE=${BASE:-outputs/evaluation/t2m/humanml3d_official_test/ms272/prism_epoch31_smooth_reseed_badcases_20260618}
OUT=${OUT:-$BASE/seed_${SEED}/h3d}
ID_FILE=${ID_FILE:-$BASE/best_of_current/remaining_bad_ids.txt}

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
export CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_31}
export KAFS_MODE=${KAFS_MODE:-depth_driven}
export OUT OUT_SUBDIR=${OUT_SUBDIR:-depth_driven}
export ANNO=${ANNO:-data/annotation/test_hml3d_official272_gtlen.json}
export REWRITTEN=${REWRITTEN:-}
export DATA_DIR=${DATA_DIR:-.}
export ID_FILE SEED
export TOTAL_SHARDS=${TOTAL_SHARDS:-8}
export SHARD_BASE=${SHARD_BASE:-0}
export NGPU=${NGPU:-8}
export SMOOTH_OUTPUT=${SMOOTH_OUTPUT:-1}
export SKIP_MOTION_EXISTENCE_CHECK=${SKIP_MOTION_EXISTENCE_CHECK:-1}
export MIN_FRAMES=${MIN_FRAMES:-24}
export MAX_FRAMES=${MAX_FRAMES:-360}

mkdir -p "$OUT"
echo "[debug-reseed] $(date) seed=$SEED id_file=$ID_FILE out=$OUT"
bash scripts/eval/run_prism_genonly_param.sh
