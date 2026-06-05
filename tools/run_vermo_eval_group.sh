#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "Usage: $0 GPU CONFIG CHECKPOINT SAMPLES_PER_TASK MODE TASKS OUT_PREFIX" >&2
  exit 2
fi

GPU="$1"
CONFIG="$2"
CHECKPOINT="$3"
SAMPLES_PER_TASK="$4"
MODE="$5"
TASKS="$6"
OUT_PREFIX="$7"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="$GPU" python3 tools/eval_vermo_overfit_alltasks.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --samples-per-task "$SAMPLES_PER_TASK" \
  --mode "$MODE" \
  --tasks "$TASKS" \
  --output-json "${OUT_PREFIX}.json" \
  > "${OUT_PREFIX}.log" 2>&1
