#!/bin/bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
OUT=${OUT:-outputs/evaluation/prism_tp2m_table2_0606}
mkdir -p "$OUT/logs"

setsid bash scripts/eval/run_prism_tp2m_table2_0606.sh \
  > "$OUT/logs/run_all.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$OUT/logs/run_all.pid"
echo "started:$pid"
