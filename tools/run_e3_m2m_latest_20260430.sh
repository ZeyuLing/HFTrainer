#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

OUT_ROOT="${1:-work_dirs/e3_m2m_latest_20260430}"
mkdir -p "${OUT_ROOT}/logs"

settings=(every_5f every_10f every_15f every_30f every_60f adaptive)
models=(uncond_local uncond_global)
max_parallel="${MAX_PARALLEL:-8}"
gpu=0

echo "Output root: ${OUT_ROOT}"
echo "Start: $(date)"

for setting in "${settings[@]}"; do
  for model in "${models[@]}"; do
    while [ "$(jobs -pr | wc -l)" -ge "${max_parallel}" ]; do
      wait -n
    done
    run_name="${model}_${setting}"
    log_path="${OUT_ROOT}/logs/${run_name}.log"
    echo "[launch] gpu=${gpu} ${run_name}"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 tools/eval_m2m_v2_all_tasks.py \
      --models "${model}" \
      --tasks E3 \
      --settings "${setting}" \
      --max-samples 240 \
      --num-steps 50 \
      --replacement-guidance skip_last \
      --text-guidance-scale 5.0 \
      --save-npz \
      --output-dir "${OUT_ROOT}/${run_name}" \
      --device cuda \
      > "${log_path}" 2>&1 &
    gpu=$(( (gpu + 1) % 8 ))
  done
done

wait
echo "Done: $(date)"
