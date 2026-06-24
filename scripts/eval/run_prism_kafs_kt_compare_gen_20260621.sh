#!/usr/bin/env bash
# Generate strict official-272 PRISM ablation outputs for diagnosing the
# MotionStreamer-evaluator FID mismatch.
#
# Variants:
#   epoch39_no_kafs         epoch39 KT-spectral checkpoint, KAFS disabled
#   iter15k_no_kt_no_kafs   original iter15000 sequential checkpoint, KAFS disabled
#
# Both use the same official HumanML3D-272 annotation and absolute translation
# decoding so the comparison isolates KT-RoPE/KAFS/checkpoint effects.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

VARIANT="${VARIANT:-epoch39_no_kafs}"
RUN_ROOT="${RUN_ROOT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_kafs_kt_compare_20260621}"

case "$VARIANT" in
  epoch39_no_kafs)
    CONFIG="configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py"
    CKPT="work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_39"
    OUT="$RUN_ROOT/epoch39_no_kafs/h3d"
    ;;
  iter15k_no_kt_no_kafs)
    CONFIG="configs/prism/prism_1b_tp2m_multiframe_iter15k.py"
    CKPT="work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"
    OUT="$RUN_ROOT/iter15k_no_kt_no_kafs/h3d"
    if [[ "${LOCALIZE_CKPT:-0}" == "1" ]]; then
      LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-/dev/shm/prism_iter15k_ckpt}"
      mkdir -p "$LOCAL_CKPT_DIR"
      if [[ ! -s "$LOCAL_CKPT_DIR/model.pt" ]] || [[ "$(stat -c%s "$LOCAL_CKPT_DIR/model.pt" 2>/dev/null || echo 0)" != "$(stat -c%s "$CKPT/model.pt")" ]]; then
        echo "[compare-gen] localizing iter15k checkpoint to $LOCAL_CKPT_DIR"
        tmp="$LOCAL_CKPT_DIR/model.pt.tmp.$$"
        cp "$CKPT/meta.pt" "$LOCAL_CKPT_DIR/meta.pt"
        cp "$CKPT/model.pt" "$tmp"
        mv "$tmp" "$LOCAL_CKPT_DIR/model.pt"
      fi
      CKPT="$LOCAL_CKPT_DIR"
    fi
    ;;
  *)
    echo "unknown VARIANT=$VARIANT" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT" "$RUN_ROOT/logs"
echo "[compare-gen] $(date -Is) variant=$VARIANT shard_base=${SHARD_BASE:-0} total=${TOTAL_SHARDS:-8} out=$OUT"

CONFIG="$CONFIG" \
CKPT="$CKPT" \
KAFS_MODE=none \
OUT_SUBDIR=none \
OUT="$OUT" \
ANNO=data/annotation/test_hml3d_official272_gtlen.json \
REWRITTEN="" \
DATA_DIR=. \
STEPS="${STEPS:-50}" \
GUIDANCE="${GUIDANCE:-5.0}" \
SEED="${SEED:-42}" \
SMOOTH_OUTPUT=1 \
SKIP_MOTION_EXISTENCE_CHECK=1 \
MIN_FRAMES=1 \
MAX_FRAMES=360 \
TOTAL_SHARDS="${TOTAL_SHARDS:-8}" \
SHARD_BASE="${SHARD_BASE:-0}" \
NGPU="${NGPU:-8}" \
TRANSLATION_DECODE_MODE=absolute \
bash scripts/eval/run_prism_genonly_param.sh
