#!/usr/bin/env bash
set -o pipefail

if [ "$#" -ne 6 ]; then
  echo "Usage: $0 <gpu> <shard_index> <num_shards> <out_root> <log_file> <exit_file>" >&2
  exit 2
fi

GPU_ID="$1"
SHARD_INDEX="$2"
NUM_SHARDS="$3"
OUT_ROOT="$4"
LOG_FILE="$5"
EXIT_FILE="$6"

REPO_ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
if [ ! -d "$REPO_ROOT" ]; then
  REPO_ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi

PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$EXIT_FILE")" "$OUT_ROOT"
cd "$REPO_ROOT" || exit 1

CUDA_VISIBLE_DEVICES="$GPU_ID" \
HFTRAINER_SKIP_AUTOREGISTER=1 \
"$PYTHON_BIN" scripts/eval/select_hml3d_gt_caption_by_motionclip.py \
  --device cuda \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --out-root "$OUT_ROOT" \
  --forward-batch-size 64 \
  > "$LOG_FILE" 2>&1
STATUS=$?
echo "$STATUS" > "$EXIT_FILE"
exit "$STATUS"
