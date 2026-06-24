#!/usr/bin/env bash
# Start the HML263 bridge hybrid diagnostic in the background.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/hml263_bridge_hybrid_diag4_0606}
LOGDIR="${OUT_ROOT}/logs"
mkdir -p "${LOGDIR}"

nohup env OUT_ROOT="${OUT_ROOT}" GPUS="${GPUS:-0,1,2,3}" \
  bash scripts/eval/run_hml263_bridge_hybrid_diag_0606.sh \
  > "${LOGDIR}/nohup.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${OUT_ROOT}/_PID"
echo "${pid}"
