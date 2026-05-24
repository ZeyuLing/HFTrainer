#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model name required}"
OUT_ROOT="${2:?output root required}"
GPU_LIST="${3:-0 1 2 3 4}"

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

settings=(default sweep_fast sweep_slow sweep_ncond5 sweep_ncond60)
read -r -a gpus <<< "${GPU_LIST}"

mkdir -p "${OUT_ROOT}/logs"

echo "[launch] host=$(hostname) model=${MODEL} out=${OUT_ROOT} gpus=${GPU_LIST}"
for i in "${!settings[@]}"; do
    setting="${settings[$i]}"
    gpu="${gpus[$((i % ${#gpus[@]}))]}"
    job_dir="${OUT_ROOT}/${MODEL}_E15_${setting}"
    log="${OUT_ROOT}/logs/${MODEL}_E15_${setting}.log"
    mkdir -p "${job_dir}"
    echo "[launch] setting=${setting} gpu=${gpu} job_dir=${job_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. nohup python3 scripts/eval/eval_m2m_v2_all_tasks.py \
        --models "${MODEL}" \
        --tasks E15 \
        --settings "${setting}" \
        --max-samples 250 \
        --num-steps 50 \
        --replacement-guidance skip_last \
        --text-guidance-scale 1.0 \
        --save-npz \
        --output-dir "${job_dir}" \
        > "${log}" 2>&1 &
    echo "$!" > "${job_dir}/pid.txt"
done

echo "[launch] started ${#settings[@]} jobs"
