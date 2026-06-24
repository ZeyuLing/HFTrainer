#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "$ROOT" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$ROOT"

RUN_ROOT=${RUN_ROOT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_translation_ablation_epoch39_20260621}

export CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
export CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_39}
export KAFS_MODE=${KAFS_MODE:-depth_driven}
export OUT_SUBDIR=${OUT_SUBDIR:-depth_driven}
export ANNO=${ANNO:-data/annotation/test_hml3d_official272_gtlen.json}
export DATA_DIR=${DATA_DIR:-.}
export REWRITTEN=${REWRITTEN:-}
export STEPS=${STEPS:-50}
export GUIDANCE=${GUIDANCE:-5.0}
export SEED=${SEED:-42}
export SMOOTH_OUTPUT=${SMOOTH_OUTPUT:-1}
export SKIP_MOTION_EXISTENCE_CHECK=${SKIP_MOTION_EXISTENCE_CHECK:-1}
export MIN_FRAMES=${MIN_FRAMES:-1}
export MAX_FRAMES=${MAX_FRAMES:-360}
export NGPU=${NGPU:-8}
export TOTAL_SHARDS=${TOTAL_SHARDS:-16}
export SHARD_BASE=${SHARD_BASE:-0}
export TRANSLATION_DECODE_MODE=${TRANSLATION_DECODE_MODE:-absolute}
export OUT=${OUT:-$RUN_ROOT/absolute/h3d}

bash scripts/eval/run_prism_genonly_param.sh
