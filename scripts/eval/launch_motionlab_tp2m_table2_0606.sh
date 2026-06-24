#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"

OUT=${OUT:-outputs/evaluation/motionlab_tp2m_table2_0606}
mkdir -p "${OUT}/logs"

setsid bash scripts/eval/run_motionlab_tp2m_table2_0606.sh \
  > "${OUT}/logs/run_all.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${OUT}/logs/run_all.pid"
echo "started:${pid}"
