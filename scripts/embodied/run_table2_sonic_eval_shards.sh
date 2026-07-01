#!/usr/bin/env bash
# Run SONIC on the unified Table-2 protocol, one case at a time.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy7/share_305994131/home/zeyuling/hf_trainer}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1}"
SONIC_REPO="${SONIC_REPO:-${PROJECT_ROOT}/ref_repo/GR00T-WholeBodyControl}"
XML_PATH="${XML_PATH:-${PROJECT_ROOT}/ref_repo/OpenTrack/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml}"
SPLITS="${SPLITS:-lafan1_fixed600 amass_test_fixed600 wild_clean_fixed600}"
TOTAL_SHARDS="${TOTAL_SHARDS:-16}"
SHARD_ID="${SHARD_ID:-0}"
GPU_ID="${GPU_ID:-0}"
INTERFACE="${INTERFACE:-bond1}"
SETUP_OVERHEAD_SECONDS="${SETUP_OVERHEAD_SECONDS:-35}"
FORCE_EVAL="${FORCE_EVAL:-0}"
SONIC_TARGET_FPS="${SONIC_TARGET_FPS:-50}"
SONIC_PYTHON="${SONIC_PYTHON:-python3}"
MIN_RUN_SECONDS="${MIN_RUN_SECONDS:-120}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROTOCOL_ROOT}/runs/sonic/logs"
exec > >(tee -a "${PROTOCOL_ROOT}/runs/sonic/logs/run_$(hostname)_shard${SHARD_ID}_gpu${GPU_ID}.log") 2>&1

echo "[sonic-table2] start $(date) host=$(hostname) shard=${SHARD_ID}/${TOTAL_SHARDS} gpu=${GPU_ID}"

for split in ${SPLITS//,/ }; do
  manifest="${PROTOCOL_ROOT}/inputs/${split}/manifest.json"
  mapfile -t names < <("${SONIC_PYTHON}" - "${manifest}" "${TOTAL_SHARDS}" "${SHARD_ID}" <<'PY'
import json, sys
names = json.loads(open(sys.argv[1]).read())
total = int(sys.argv[2])
shard = int(sys.argv[3])
for i, name in enumerate(names):
    if i % total == shard:
        print(name)
PY
)
  echo "[sonic-table2] split=${split} cases=${#names[@]}"
  for name in "${names[@]}"; do
    ref_npz="${PROTOCOL_ROOT}/inputs/${split}/npz/${name}.npz"
    out_dir="${PROTOCOL_ROOT}/runs/sonic/${split}/${name}"
    metric_json="${out_dir}/metrics.json"
    if [[ "${FORCE_EVAL}" != "1" && -s "${metric_json}" ]]; then
      echo "[sonic-table2] skip done ${split}/${name}"
      continue
    fi
    ref_root="${out_dir}/reference_parent"
    sonic_ref_npz="${out_dir}/sonic_reference_qpos.npz"
    rm -rf "${ref_root}"
    "${SONIC_PYTHON}" scripts/embodied/prepare_sonic_reference_from_npz.py \
      --npz "${ref_npz}" \
      --out-root "${ref_root}" \
      --name "${name}" \
      --target-fps "${SONIC_TARGET_FPS}" \
      --out-npz "${sonic_ref_npz}"
    read -r frames fps < <("${SONIC_PYTHON}" - "${sonic_ref_npz}" <<'PY'
import numpy as np, sys
data = np.load(sys.argv[1], allow_pickle=True)
q = data["qpos"]
fps = float(np.asarray(data["frequency"]).reshape(-1)[0])
print(q.shape[0], int(round(fps)))
PY
)
    run_seconds=$(( SETUP_OVERHEAD_SECONDS + (frames + fps - 1) / fps + 3 ))
    if (( run_seconds < MIN_RUN_SECONDS )); then
      run_seconds="${MIN_RUN_SECONDS}"
    fi
    echo "[sonic-table2] run ${split}/${name} frames=${frames} seconds=${run_seconds}"
    GPU_ID="${GPU_ID}" INTERFACE="${INTERFACE}" RUN_SECONDS="${run_seconds}" OUT_DIR="${out_dir}" REFERENCE_DIR="${ref_root}" SONIC_PYTHON="${SONIC_PYTHON}" \
      bash scripts/embodied/run_sonic_reference_smoke.sh
    "${SONIC_PYTHON}" scripts/embodied/eval_sonic_qpos_logs.py \
      --ref-npz "${sonic_ref_npz}" \
      --run-dir "${out_dir}" \
      --xml "${XML_PATH}" \
      --output-json "${metric_json}" \
      --require-q-log
  done
done

echo "[sonic-table2] done $(date)"
