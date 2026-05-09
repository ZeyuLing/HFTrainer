#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_ROOT}"

OUT_ROOT="${1:-work_dirs/m2m_v2_latest_selective_rerun_20260429}"
mkdir -p "${OUT_ROOT}/logs"

nohup bash tools/run_m2m_v2_latest_selective_rerun_20260429.sh "${OUT_ROOT}" \
  > "${OUT_ROOT}/driver.log" 2>&1 &
pid=$!
echo "${pid}" > "${OUT_ROOT}/driver.pid"
echo "started ${OUT_ROOT} pid=${pid}"
