#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo "Usage: $0 CONFIG CHECKPOINT SAMPLES_PER_TASK MODE OUT_PREFIX CHECKPOINT_TAG" >&2
  exit 2
fi

CONFIG="$1"
CHECKPOINT="$2"
SAMPLES_PER_TASK="$3"
MODE="$4"
OUT_PREFIX="$5"
CHECKPOINT_TAG="$6"

cd "$(dirname "$0")/.."

declare -a GROUP_TASKS=(
  "t2m,m2t"
  "n2tm,pred,inbetween"
  "m2d,d2m"
  "t2md,g2md"
  "n2md,m2d_ar"
  "d2m_ar,s2g"
  "g2s,t2sg,n2sg"
  "ss2sg,s2g_ar"
)

declare -a PIDS=()
for GPU in "${!GROUP_TASKS[@]}"; do
  GROUP_PREFIX="${OUT_PREFIX}_${CHECKPOINT_TAG}_${MODE}_group${GPU}"
  bash tools/run_vermo_eval_group.sh \
    "$GPU" \
    "$CONFIG" \
    "$CHECKPOINT" \
    "$SAMPLES_PER_TASK" \
    "$MODE" \
    "${GROUP_TASKS[$GPU]}" \
    "$GROUP_PREFIX" &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    STATUS=1
  fi
done

exit "$STATUS"
