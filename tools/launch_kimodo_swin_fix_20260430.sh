#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:?group required}"
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
LOG_DIR="$ROOT/work_dirs/kimodo_swin_fix_20260430/logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

LOG="$LOG_DIR/scheduler_${GROUP}.log"
PID_FILE="$LOG_DIR/scheduler_${GROUP}.pid"
nohup python3 tools/run_kimodo_swin_fix_20260430.py --group "$GROUP" > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$PID_FILE"
echo "kimodo_swin_${GROUP}_pid=$(cat "$PID_FILE") log=$LOG"
