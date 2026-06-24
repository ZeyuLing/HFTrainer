#!/bin/bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
OUT=${OUT:-outputs/evaluation/motionstreamer_align0606}
mkdir -p "$OUT/logs"

nohup bash scripts/eval/run_motionstreamer_align0606.sh \
  > "$OUT/logs/run_all.log" 2>&1 &
pid=$!
echo "$pid" > "$OUT/logs/run_all.pid"
echo "started:$pid"
