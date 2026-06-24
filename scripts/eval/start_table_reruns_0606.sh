#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"

FLOW_LOGDIR="outputs/evaluation/flow_motionlab_orig_eval0606/logs"
HY_LOGDIR="outputs/evaluation/hylite_h3d_anno0606/logs"
mkdir -p "${FLOW_LOGDIR}" "${HY_LOGDIR}"

nohup bash scripts/eval/run_flow_motionlab_orig_eval_0606.sh \
  > "${FLOW_LOGDIR}/nohup.log" 2>&1 < /dev/null &
echo "$!" > "${FLOW_LOGDIR}/pid"

GPUS="${HY_GPUS:-5,6,7,0}" EVAL_GPU="${HY_EVAL_GPU:-5}" \
  nohup bash scripts/eval/run_hylite_h3d_anno_infer_eval_0606.sh \
  > "${HY_LOGDIR}/nohup.log" 2>&1 < /dev/null &
echo "$!" > "${HY_LOGDIR}/pid"

echo "flow_pid=$(cat "${FLOW_LOGDIR}/pid")"
echo "hylite_pid=$(cat "${HY_LOGDIR}/pid")"
