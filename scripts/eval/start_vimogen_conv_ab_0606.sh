#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"

LOGDIR="outputs/evaluation/vimogen_t2m_0605/conv_ab_0606/logs"
mkdir -p "${LOGDIR}"

RUN="${RUN:-h3d_rw_full0605_dn1coord_lock_s0of8}" \
DATASET="${DATASET:-h3d}" \
GPU="${GPU:-4}" \
CHUNK_SIZE="${CHUNK_SIZE:-64}" \
SUMMARY="${SUMMARY:-outputs/evaluation/vimogen_t2m_0605/conv_ab_0606/summary.txt}" \
  nohup bash scripts/eval/run_vimogen_conversion_ab_0605.sh \
  > "${LOGDIR}/nohup_${DATASET}.log" 2>&1 < /dev/null &

echo "$!" > "${LOGDIR}/pid_${DATASET}"
echo "pid=$(cat "${LOGDIR}/pid_${DATASET}")"
