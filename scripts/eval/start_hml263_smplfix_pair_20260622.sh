#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

ROLE="${ROLE:-}"
RUN_ID="${RUN_ID:-table1_hml263_smplfix_20260622}"
REFINE_ITERS="${REFINE_ITERS:-80}"
SUITE="outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/$RUN_ID"
LAUNCH="$SUITE/launch"
mkdir -p "$LAUNCH"

start_method() {
  local method="$1"
  local gpu_list="$2"
  local shards="$3"
  local out="$LAUNCH/${method}.${ROLE}.out"
  local pidfile="$LAUNCH/${method}.${ROLE}.pid"
  if [[ -f "$pidfile" ]]; then
    local old_pid
    old_pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[skip-running] $method pid=$old_pid"
      return
    fi
  fi
  setsid bash -c '
    echo "$$" > "$1"
    exec env METHOD="$2" NUM_SHARDS="$3" GPU_LIST="$4" REFINE_ITERS="$5" RUN_ID="$6" \
      bash scripts/eval/run_hml263_exact_smplfix_20260622.sh > "$7" 2>&1
  ' _ "$pidfile" "$method" "$shards" "$gpu_list" "$REFINE_ITERS" "$RUN_ID" "$out" &
  sleep 0.2
  local pid
  pid="$(cat "$pidfile")"
  echo "[launched] $method role=$ROLE pid=$pid gpus=$gpu_list shards=$shards out=$out"
}

case "$ROLE" in
  machine1)
    start_method mdm 0,1,2,3 4
    start_method motiongpt3 4,5,6,7 4
    ;;
  machine2)
    start_method flowmdm 0,1,2,3 4
    start_method motionlab 4,5,6,7 4
    ;;
  *)
    echo "[error] set ROLE=machine1 or ROLE=machine2" >&2
    exit 2
    ;;
esac
