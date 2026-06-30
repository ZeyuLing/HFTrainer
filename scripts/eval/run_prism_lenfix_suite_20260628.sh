#!/usr/bin/env bash
# Run the PRISM epoch-43 official-selected HumanML3D length-fix suite on one
# already-running GPU node. This is intentionally generation-only; repack and
# MotionStreamer-272 metrics are launched after coverage/length validation.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

SUITE="${SUITE:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_epoch43_official_selected_lenfix_20260628}"
ANNO="${ANNO:-outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json}"
CONFIG="${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}"
CKPT="${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_43}"
NGPU="${NGPU:-4}"
TOTAL_SHARDS="${TOTAL_SHARDS:-$NGPU}"
SHARD_BASE="${SHARD_BASE:-0}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-5.0}"
PAD_TO_FRAMES="${PAD_TO_FRAMES:-360}"
TRANSLATION_DECODE_MODE="${TRANSLATION_DECODE_MODE:-rollout}"

mkdir -p "$SUITE/_remote_logs"

if [[ "${WAIT_FOR_IDLE:-0}" == "1" ]]; then
  echo "[suite] waiting for active PRISM workers to finish..."
  while pgrep -f "eval_prism_kafs_ablation.py.*${SUITE}/raw" >/dev/null; do
    date -Is
    pgrep -af "eval_prism_kafs_ablation.py.*${SUITE}/raw" || true
    sleep "${WAIT_SLEEP:-60}"
  done
fi

run_one() {
  local policy="$1"
  local mode="$2"
  local log="$SUITE/_remote_logs/${policy}_${mode}_suite.log"
  echo "[suite] start policy=$policy mode=$mode $(date -Is)" | tee -a "$SUITE/_remote_logs/suite_runner.log"
  CONFIG="$CONFIG" \
  CKPT="$CKPT" \
  KAFS_MODE="$mode" \
  OUT_SUBDIR="$mode" \
  OUT="$SUITE/raw/$policy" \
  ANNO="$ANNO" \
  DATA_DIR=. \
  LENGTH_POLICY="$policy" \
  PAD_TO_FRAMES="$PAD_TO_FRAMES" \
  MIN_FRAMES=1 \
  MAX_FRAMES=360 \
  TOTAL_SHARDS="$TOTAL_SHARDS" \
  SHARD_BASE="$SHARD_BASE" \
  NGPU="$NGPU" \
  STEPS="$STEPS" \
  GUIDANCE="$GUIDANCE" \
  TRANSLATION_DECODE_MODE="$TRANSLATION_DECODE_MODE" \
  bash scripts/eval/run_prism_genonly_param.sh > "$log" 2>&1
  echo "[suite] done policy=$policy mode=$mode $(date -Is)" | tee -a "$SUITE/_remote_logs/suite_runner.log"
}

for policy in direct_len pad360_crop; do
  for mode in depth_driven none uniform random; do
    run_one "$policy" "$mode"
  done
done

echo "[suite] all generation jobs completed $(date -Is)" | tee -a "$SUITE/_remote_logs/suite_runner.log"
