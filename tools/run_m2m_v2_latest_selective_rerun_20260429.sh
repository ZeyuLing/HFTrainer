#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

OUT_ROOT="${1:-work_dirs/m2m_v2_latest_selective_rerun_20260429}"
mkdir -p "${OUT_ROOT}/logs"

jobs=(
  "0 uncond_local  E14 L       100"
  "1 uncond_global E14 L       100"
  "2 uncond_local  E14 M       100"
  "3 uncond_global E14 M       100"
  "4 uncond_local  E15 default 200"
  "5 uncond_global E15 default 200"
  "6 uncond_local  E8  D       200"
  "7 uncond_global E8  D       200"
)

echo "Output root: ${OUT_ROOT}"
echo "Start time: $(date)"

for spec in "${jobs[@]}"; do
  read -r gpu model task setting max_samples <<< "${spec}"
  run_name="${model}_${task}_${setting}"
  log_path="${OUT_ROOT}/logs/${run_name}.log"
  echo "[launch] gpu=${gpu} ${run_name} max_samples=${max_samples} -> ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python3 tools/eval_m2m_v2_all_tasks.py \
    --models "${model}" \
    --tasks "${task}" \
    --settings "${setting}" \
    --max-samples "${max_samples}" \
    --num-steps 50 \
    --replacement-guidance skip_last \
    --text-guidance-scale 5.0 \
    --save-npz \
    --output-dir "${OUT_ROOT}/${run_name}" \
    --device cuda \
    > "${log_path}" 2>&1 &
done

wait
echo "Done time: $(date)"
