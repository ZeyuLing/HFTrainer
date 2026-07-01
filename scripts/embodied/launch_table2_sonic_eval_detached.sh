#!/usr/bin/env bash
# Detach one SONIC Table-2 shard from an interactive Taiji exec shell.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy7/share_305994131/home/zeyuling/hf_trainer}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:?set PROTOCOL_ROOT}"
INPUTS_ROOT="${INPUTS_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1/inputs}"
SPLITS="${SPLITS:-lafan1_fixed600,amass_test_fixed600,wild_clean_fixed600}"
TOTAL_SHARDS="${TOTAL_SHARDS:?set TOTAL_SHARDS}"
SHARD_ID="${SHARD_ID:?set SHARD_ID}"
GPU_ID="${GPU_ID:-0}"
INTERFACE="${INTERFACE:-bond1}"
SETUP_OVERHEAD_SECONDS="${SETUP_OVERHEAD_SECONDS:-35}"
FORCE_EVAL="${FORCE_EVAL:-1}"
KILL_LARGE_OCCUPANTS="${KILL_LARGE_OCCUPANTS:-1}"
SONIC_TARGET_FPS="${SONIC_TARGET_FPS:-50}"
HOST_LABEL="${HOST_LABEL:-$(hostname)}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROTOCOL_ROOT}/runs/sonic/logs"
ln -sfn "${INPUTS_ROOT}" "${PROTOCOL_ROOT}/inputs"

launch_log="${PROTOCOL_ROOT}/runs/sonic/logs/launcher_${HOST_LABEL}_shard${SHARD_ID}_gpu${GPU_ID}.out"
pid_file="${PROTOCOL_ROOT}/runs/sonic/logs/launcher_${HOST_LABEL}_shard${SHARD_ID}_gpu${GPU_ID}.pid"

setsid nohup env \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  PROTOCOL_ROOT="${PROTOCOL_ROOT}" \
  SPLITS="${SPLITS}" \
  TOTAL_SHARDS="${TOTAL_SHARDS}" \
  SHARD_ID="${SHARD_ID}" \
  GPU_ID="${GPU_ID}" \
  INTERFACE="${INTERFACE}" \
  SETUP_OVERHEAD_SECONDS="${SETUP_OVERHEAD_SECONDS}" \
  FORCE_EVAL="${FORCE_EVAL}" \
  KILL_LARGE_OCCUPANTS="${KILL_LARGE_OCCUPANTS}" \
  SONIC_TARGET_FPS="${SONIC_TARGET_FPS}" \
  bash scripts/embodied/run_table2_sonic_eval_shards.sh \
  > "${launch_log}" 2>&1 < /dev/null &

pid=$!
echo "${pid}" > "${pid_file}"
disown "${pid}" 2>/dev/null || true
echo "[sonic-launcher] host=${HOST_LABEL} shard=${SHARD_ID}/${TOTAL_SHARDS} gpu=${GPU_ID} pid=${pid} log=${launch_log}"
