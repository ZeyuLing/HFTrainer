#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
cd "$ROOT"

SUITE=${SUITE:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_epoch43_translation_decode_t2m_20260629}
mkdir -p "$SUITE/logs"
SHARD_BASE_TAG=${SHARD_BASE:-0}
HOST_TAG=$(hostname | tr -c 'A-Za-z0-9_.-' '_')
DRIVER_LOG="$SUITE/logs/driver_base${SHARD_BASE_TAG}_${HOST_TAG}.log"
DRIVER_PID="$SUITE/driver_base${SHARD_BASE_TAG}_${HOST_TAG}.pid"

if pgrep -af "run_prism_t2m_translation_decode_ablation_20260629.sh" >/dev/null; then
    echo "[start] driver already running"
    pgrep -af "run_prism_t2m_translation_decode_ablation_20260629.sh"
    exit 0
fi

nohup bash scripts/eval/run_prism_t2m_translation_decode_ablation_20260629.sh \
  > "$DRIVER_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$DRIVER_PID"
echo "[start] launched driver pid=$pid"
echo "[start] driver log=$DRIVER_LOG"
sleep 2
if kill -0 "$pid" 2>/dev/null; then
  echo "[start] driver alive"
else
  echo "[start][error] driver exited quickly"
  tail -80 "$DRIVER_LOG" || true
  exit 1
fi
