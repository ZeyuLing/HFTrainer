#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
mkdir -p outputs/evaluation/motionhub_hml3d263_rewrite_0605
nohup bash scripts/eval/run_motionhub_flowmdm_hml263_full_0605.sh \
  > outputs/evaluation/motionhub_hml3d263_rewrite_0605/flowmdm_nohup.log \
  2>&1 < /dev/null &
echo "$!" > outputs/evaluation/motionhub_hml3d263_rewrite_0605/flowmdm_nohup.pid
echo "started $(cat outputs/evaluation/motionhub_hml3d263_rewrite_0605/flowmdm_nohup.pid)"
