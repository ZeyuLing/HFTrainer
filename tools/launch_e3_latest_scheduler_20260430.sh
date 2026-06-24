#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model required}"
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
LOG_DIR="$ROOT/work_dirs/e3_latest_20260430_1747/logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

LOG="$LOG_DIR/scheduler_${MODEL}.log"
PID_FILE="$LOG_DIR/scheduler_${MODEL}.pid"
nohup python3 tools/run_e3_latest_sharded_20260430.py --model "$MODEL" > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$PID_FILE"
echo "e3_${MODEL}_scheduler_pid=$(cat "$PID_FILE") log=$LOG"
