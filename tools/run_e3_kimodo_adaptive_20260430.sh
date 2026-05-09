#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

OUT_ROOT="${1:-work_dirs/e3_kimodo_adaptive_20260430}"
mkdir -p "${OUT_ROOT}/logs"

total=240
shards=8
chunk=$(( (total + shards - 1) / shards ))

echo "Output root: ${OUT_ROOT}"
echo "Start: $(date)"

for shard in $(seq 0 $((shards - 1))); do
  start=$(( shard * chunk ))
  end=$(( start + chunk ))
  if [ "${end}" -gt "${total}" ]; then end="${total}"; fi
  shard_dir="${OUT_ROOT}/shard_${shard}"
  log_path="${OUT_ROOT}/logs/shard_${shard}_${start}_${end}.log"
  echo "[launch] gpu=${shard} shard=${shard} range=[${start},${end})"
  CUDA_VISIBLE_DEVICES="${shard}" python3 tools/run_kimodo_all_tasks.py \
    --tasks E3 \
    --settings adaptive \
    --max-samples "${total}" \
    --start-idx "${start}" \
    --end-idx "${end}" \
    --use-caption no \
    --output-dir "${shard_dir}" \
    --device cuda \
    > "${log_path}" 2>&1 &
done

wait

python3 tools/merge_kimodo_shards_simple.py \
  --shard-root "${OUT_ROOT}" \
  --final-dir "${OUT_ROOT}/merged/E3_adaptive" \
  --task-key E3_adaptive \
  --task-id E3 \
  --setting adaptive \
  --expected "${total}"

echo "Done: $(date)"
