#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model name required}"
SETTING="${2:?setting required}"
GPU="${3:?gpu id required}"
OUT_ROOT="${4:?output root required}"

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

job_dir="${OUT_ROOT}/${MODEL}_E15_${SETTING}"
log="${OUT_ROOT}/logs/${MODEL}_E15_${SETTING}.log"
mkdir -p "${job_dir}" "${OUT_ROOT}/logs"

echo "[launch-one] host=$(hostname) model=${MODEL} setting=${SETTING} gpu=${GPU} job_dir=${job_dir}"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. nohup python3 scripts/eval/eval_m2m_v2_all_tasks.py \
    --models "${MODEL}" \
    --tasks E15 \
    --settings "${SETTING}" \
    --max-samples 250 \
    --num-steps 50 \
    --replacement-guidance skip_last \
    --text-guidance-scale 1.0 \
    --save-npz \
    --output-dir "${job_dir}" \
    > "${log}" 2>&1 &
echo "$!" > "${job_dir}/pid.txt"
echo "[launch-one] pid=$(cat "${job_dir}/pid.txt") log=${log}"
