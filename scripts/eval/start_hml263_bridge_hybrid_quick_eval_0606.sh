#!/usr/bin/env bash
# Start fast HML263 bridge hybrid eval in the background.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/hml263_bridge_hybrid_diag5_0606}
MAX_PAIRS=${MAX_PAIRS:-512}
LOGDIR="${OUT_ROOT}/logs_quick"
mkdir -p "${LOGDIR}"

nohup env OUT_ROOT="${OUT_ROOT}" MAX_PAIRS="${MAX_PAIRS}" GPUS="${GPUS:-0,1,2,3}" \
  bash scripts/eval/run_hml263_bridge_hybrid_quick_eval_0606.sh \
  > "${LOGDIR}/nohup.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${LOGDIR}/_PID"
echo "${pid}"
