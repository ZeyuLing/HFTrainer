#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
mkdir -p outputs/evaluation/motionhub_motionclip135_rewrite_0605
nohup bash scripts/eval/run_motionhub_motionlab_retarget_eval_0605.sh \
  > outputs/evaluation/motionhub_motionclip135_rewrite_0605/motionlab_retarget_nohup.log \
  2>&1 < /dev/null &
echo "$!" > outputs/evaluation/motionhub_motionclip135_rewrite_0605/motionlab_retarget.pid
echo "started $(cat outputs/evaluation/motionhub_motionclip135_rewrite_0605/motionlab_retarget.pid)"
