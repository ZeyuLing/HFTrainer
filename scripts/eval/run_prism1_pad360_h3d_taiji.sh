#!/usr/bin/env bash
# Taiji worker for rerunning PRISM 1.0 on HumanML3D with pad360/crop inference.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SUITE="${SUITE:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism1_pad360crop_official_selected_20260630}"
ANNO="${ANNO:-outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/test_hml3d_official272_gtlen_motionclip_selected_caption.json}"
CONFIG="${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_iter15k.py}"
CKPT="${CKPT:-work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000}"
NGPU="${NGPU:-8}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-5.0}"
TRANSLATION_DECODE_MODE="${TRANSLATION_DECODE_MODE:-xz_rollout_y_absolute}"

HOST_RANK="${INDEX:-${SHARD_BASE:-${HOST_RANK:-0}}}"
mkdir -p "$SUITE/_remote_logs"

{
  echo "[prism1-pad360] start=$(date -Is)"
  echo "[prism1-pad360] host_rank=$HOST_RANK node_list=${NODE_LIST:-}"
  echo "[prism1-pad360] suite=$SUITE"
  echo "[prism1-pad360] config=$CONFIG"
  echo "[prism1-pad360] ckpt=$CKPT"
  echo "[prism1-pad360] anno=$ANNO"
  echo "[prism1-pad360] translation=$TRANSLATION_DECODE_MODE"
} | tee "$SUITE/_remote_logs/host_${HOST_RANK}_start.log"

CONFIG="$CONFIG" \
CKPT="$CKPT" \
KAFS_MODE="none" \
OUT_SUBDIR="none" \
OUT="$SUITE/raw/pad360_crop" \
ANNO="$ANNO" \
DATA_DIR="." \
LENGTH_POLICY="pad360_crop" \
PAD_TO_FRAMES="360" \
MIN_FRAMES="1" \
MAX_FRAMES="360" \
NGPU="$NGPU" \
STEPS="$STEPS" \
GUIDANCE="$GUIDANCE" \
TRANSLATION_DECODE_MODE="$TRANSLATION_DECODE_MODE" \
SKIP_MOTION_EXISTENCE_CHECK="1" \
bash scripts/eval/run_prism_genonly_param.sh \
  > "$SUITE/_remote_logs/host_${HOST_RANK}_gen.log" 2>&1

count=$(find "$SUITE/raw/pad360_crop/none" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
{
  echo "[prism1-pad360] done=$(date -Is)"
  echo "[prism1-pad360] host_rank=$HOST_RANK"
  echo "[prism1-pad360] raw_count=$count"
} | tee "$SUITE/_remote_logs/host_${HOST_RANK}_done.log"
