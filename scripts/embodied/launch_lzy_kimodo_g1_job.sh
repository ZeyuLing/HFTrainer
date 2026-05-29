#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUT_DIR="${1:?output directory required}"
GPU_ID="${2:-1}"
SEED="${3:-42}"

mkdir -p "${OUT_DIR}"

nohup bash scripts/embodied/run_kimodo_g1_smoke3_lzy.sh "${OUT_DIR}" "${GPU_ID}" "${SEED}" \
  > "${OUT_DIR}/launcher.log" 2>&1 < /dev/null &

PID="$!"
echo "${PID}" > "${OUT_DIR}/pid.txt"
echo "PID:${PID}"
sleep 1
ps -p "${PID}" -o pid,etime,cmd --no-headers || true
ls -la "${OUT_DIR}"
