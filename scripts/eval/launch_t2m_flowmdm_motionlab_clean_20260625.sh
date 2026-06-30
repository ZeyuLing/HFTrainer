#!/usr/bin/env bash
# Launch FlowMDM and MotionLab clean HumanML3D T2M reruns on one 8-GPU host.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HFTRAINER_SKIP_AUTOREGISTER=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

BASE="outputs/evaluation/t2m/humanml3d_official_test"
RUN_ROOT="$BASE/_runs/flowmdm_motionlab_clean_20260625_launcher"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

LOCK_DIR="$RUN_ROOT/.lock"
DONE_FILE="$RUN_ROOT/_DONE"
DISABLE_FILE="$RUN_ROOT/_DISABLE_FULL_LAUNCH"
if [[ -f "$DISABLE_FILE" ]]; then
  echo "[launcher-skip] disabled by marker: $DISABLE_FILE"
  exit 0
fi
if [[ -f "$DONE_FILE" ]]; then
  echo "[launcher-skip] done marker exists: $DONE_FILE"
  exit 0
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[launcher-skip] another FlowMDM/MotionLab clean rerun is active: $LOCK_DIR"
  exit 0
fi
cleanup_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

FLOW_GPU_LIST="${FLOW_GPU_LIST:-0,1,2,3}"
MOTIONLAB_GPU_LIST="${MOTIONLAB_GPU_LIST:-4,5,6,7}"
FLOW_SHARDS="${FLOW_SHARDS:-4}"
MOTIONLAB_SHARDS="${MOTIONLAB_SHARDS:-4}"
FLOW_LOCAL_SHARDS="${FLOW_LOCAL_SHARDS:-$FLOW_SHARDS}"
MOTIONLAB_LOCAL_SHARDS="${MOTIONLAB_LOCAL_SHARDS:-$MOTIONLAB_SHARDS}"

echo "[launcher-start] $(date -Is)" | tee "$LOG_DIR/launcher.log"
echo "[launcher-config] flow_gpus=$FLOW_GPU_LIST motionlab_gpus=$MOTIONLAB_GPU_LIST flow_shards=$FLOW_SHARDS motionlab_shards=$MOTIONLAB_SHARDS" | tee -a "$LOG_DIR/launcher.log"

launch_one() {
  local method="$1"
  local gpu_list="$2"
  local shards="$3"
  local local_shards="$4"
  local log="$LOG_DIR/${method}.log"
  (
    set -euo pipefail
    METHOD="$method" \
    RUN_TAG="${method}_clean_20260625" \
    GPU_LIST="$gpu_list" \
    TOTAL_SHARDS="$shards" \
    LOCAL_SHARDS="$local_shards" \
    CLEAN="${CLEAN:-1}" \
    bash scripts/eval/run_t2m_flowmdm_motionlab_clean_20260625.sh
  ) >"$log" 2>&1 &
  echo "$!" > "$LOG_DIR/${method}.pid"
  echo "[launch] method=$method pid=$(cat "$LOG_DIR/${method}.pid") log=$log" | tee -a "$LOG_DIR/launcher.log"
}

launch_one flowmdm "$FLOW_GPU_LIST" "$FLOW_SHARDS" "$FLOW_LOCAL_SHARDS"
launch_one motionlab "$MOTIONLAB_GPU_LIST" "$MOTIONLAB_SHARDS" "$MOTIONLAB_LOCAL_SHARDS"

rc=0
for method in flowmdm motionlab; do
  pid="$(cat "$LOG_DIR/${method}.pid")"
  if ! wait "$pid"; then
    echo "[fail] method=$method pid=$pid $(date -Is)" | tee -a "$LOG_DIR/launcher.log"
    rc=1
  else
    echo "[done] method=$method pid=$pid $(date -Is)" | tee -a "$LOG_DIR/launcher.log"
  fi
done

if [[ "$rc" != "0" ]]; then
  echo "[launcher-fail] $(date -Is)" | tee -a "$LOG_DIR/launcher.log"
  exit "$rc"
fi

echo "[launcher-done] $(date -Is)" | tee -a "$LOG_DIR/launcher.log"
touch "$DONE_FILE"
