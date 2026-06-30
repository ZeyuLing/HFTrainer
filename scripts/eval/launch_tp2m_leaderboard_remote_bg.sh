#!/usr/bin/env bash
# Start one TP2M leaderboard completion runner as a detached process.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [ ! -d "${ROOT}" ]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi

METHOD="${1:?method required}"
LABEL="${2:?label required}"
SHARD_OFFSET="${3:-0}"
SHARD_COUNT="${4:-8}"
TOTAL_SHARDS="${5:-16}"
CONDS="${6:-1 5 9}"
GPU_LIST="${7:-0,1,2,3,4,5,6,7}"
RUN_ROOT="${RUN_ROOT:-outputs/evaluation/tp2m/_runs/leaderboard_missing_20260629}"
LOG_DIR="${ROOT}/outputs/evaluation/tp2m/_runs/a100pro_20260629"
LOG="${LOG_DIR}/${LABEL}.log"
PID_FILE="${LOG_DIR}/${LABEL}.pid"
TMUX_SESSION="tp2m_${LABEL}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "${TMUX_SESSION}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT}' && env RUN_ROOT='${RUN_ROOT}' METHOD='${METHOD}' TOTAL_SHARDS='${TOTAL_SHARDS}' SHARD_OFFSET='${SHARD_OFFSET}' SHARD_COUNT='${SHARD_COUNT}' GPU_LIST='${GPU_LIST}' CONDS='${CONDS}' bash scripts/eval/run_tp2m_leaderboard_missing_remote.sh > '${LOG}' 2>&1"
  pid="$(tmux display-message -p -t "${TMUX_SESSION}" "#{pane_pid}")"
else
  nohup env \
    RUN_ROOT="${RUN_ROOT}" \
    METHOD="${METHOD}" \
    TOTAL_SHARDS="${TOTAL_SHARDS}" \
    SHARD_OFFSET="${SHARD_OFFSET}" \
    SHARD_COUNT="${SHARD_COUNT}" \
    GPU_LIST="${GPU_LIST}" \
    CONDS="${CONDS}" \
    bash scripts/eval/run_tp2m_leaderboard_missing_remote.sh \
    > "${LOG}" 2>&1 < /dev/null &
  pid="$!"
fi
echo "${pid}" > "${PID_FILE}"
echo "PID=${pid}"
echo "LOG=${LOG}"
echo "PID_FILE=${PID_FILE}"
echo "TMUX_SESSION=${TMUX_SESSION}"
