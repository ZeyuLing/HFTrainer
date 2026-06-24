#!/usr/bin/env bash
# Run Humanoid-GPT Table-2 tracker evaluation on all requested splits.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/humanoid_gpt}"
SPLITS="${SPLITS:-lafan1 wild amass}"
NUM_SHARDS="${NUM_SHARDS:-4}"
AMASS_NUM_SHARDS="${AMASS_NUM_SHARDS:-32}"
TIMEOUT_S="${TIMEOUT_S:-7200}"
AMASS_OUT_NAME="${AMASS_OUT_NAME:-amass_v3}"

LAFAN_ROOT="${LAFAN_ROOT:-${PROJECT_ROOT}/data/LAFAN1_Retargeted_for_G1/UnitreeG1}"
LAFAN_FROZEN_MANIFEST="${LAFAN_FROZEN_MANIFEST:-${PROJECT_ROOT}/output/opentrack_lafan1_g1/local_py311_full_localmetric_20260604_233656/manifests/all.json}"
WILD_ROOT="${WILD_ROOT:-${PROJECT_ROOT}/output/heldout_frozen_score}"
AMASS_ROOT="${AMASS_ROOT:-${PROJECT_ROOT}/data/AMASS_Retarged_for_G1/g1}"

HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-/dev/shm/hgpt_venv311/bin/python}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_BASE}"

if [[ ! -x "${HGPT_PYTHON}" ]]; then
  echo "[hgpt-v2] building HGPT worker env"
  bash scripts/embodied/physflow_hgpt_node_setup.sh
fi
HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-/dev/shm/hgpt_venv311/bin/python}"

echo "[hgpt-v2] start $(date)"
echo "[hgpt-v2] host=$(hostname)"
echo "[hgpt-v2] python=${HGPT_PYTHON}"
echo "[hgpt-v2] splits=${SPLITS}"

make_shards() {
  local manifest="$1"
  local out_dir="$2"
  local num_shards="$3"
  mkdir -p "${out_dir}/manifests"
  "${HGPT_PYTHON}" - "${manifest}" "${out_dir}/manifests" "${num_shards}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
manifest_dir = Path(sys.argv[2])
num_shards = int(sys.argv[3])
names = json.loads(manifest.read_text())
shards = [[] for _ in range(num_shards)]
for i, name in enumerate(names):
    shards[i % num_shards].append(name)
for i, shard in enumerate(shards):
    (manifest_dir / f"shard_{i}.json").write_text(json.dumps(shard, indent=2) + "\n")
print(f"motions={len(names)} nonempty_shards={sum(bool(s) for s in shards)}")
PY
}

score_to_manifest() {
  local score_json="$1"
  local manifest="$2"
  "${HGPT_PYTHON}" - "${score_json}" "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

score = json.loads(Path(sys.argv[1]).read_text())
names = [f"h{int(row['idx']):03d}_gen" for row in score["rows"]]
Path(sys.argv[2]).write_text(json.dumps(names, indent=2) + "\n")
print(f"manifest={sys.argv[2]} motions={len(names)}")
PY
}

recursive_manifest() {
  local motion_dir="$1"
  local manifest="$2"
  "${HGPT_PYTHON}" scripts/embodied/run_table2_hgpt_eval.py \
    --motion-dir "${motion_dir}" \
    --manifest "${manifest}" \
    --out-dir "$(dirname "${manifest}")/.manifest_probe" \
    --write-recursive-manifest "${manifest}"
}

run_shards() {
  local motion_dir="$1"
  local out_dir="$2"
  local num_shards="$3"
  mkdir -p "${out_dir}"
  make_shards "${out_dir}/manifest.json" "${out_dir}" "${num_shards}"
  local pids=()
  for shard in $(seq 0 $((num_shards - 1))); do
    local shard_manifest="${out_dir}/manifests/shard_${shard}.json"
    local count
    count="$("${HGPT_PYTHON}" -c "import json; print(len(json.load(open('${shard_manifest}'))))")"
    if [[ "${count}" == "0" ]]; then
      echo "[hgpt-v2] ${out_dir} shard ${shard}: empty"
      continue
    fi
    (
      "${HGPT_PYTHON}" scripts/embodied/run_table2_hgpt_eval.py \
        --motion-dir "${motion_dir}" \
        --manifest "${shard_manifest}" \
        --out-dir "${out_dir}/shard_${shard}" \
        --hgpt-python "${HGPT_PYTHON}" \
        --timeout-s "${TIMEOUT_S}" \
        > "${out_dir}/shard_${shard}.log" 2>&1
    ) &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "[hgpt-v2] ERROR: one or more shards failed under ${out_dir}" >&2
    exit 2
  fi
  "${HGPT_PYTHON}" scripts/embodied/aggregate_hgpt_eval.py --eval-root "${out_dir}"
}

for split in ${SPLITS}; do
  case "${split}" in
    lafan1)
      out_dir="${OUT_BASE}/lafan1_v2"
      mkdir -p "${out_dir}"
      cp "${LAFAN_FROZEN_MANIFEST}" "${out_dir}/manifest.json"
      run_shards "${LAFAN_ROOT}" "${out_dir}" "${NUM_SHARDS}"
      ;;
    wild)
      out_dir="${OUT_BASE}/wild_v2"
      mkdir -p "${out_dir}"
      score_to_manifest "${WILD_ROOT}/heldout_score.json" "${out_dir}/manifest.json"
      run_shards "${WILD_ROOT}" "${out_dir}" "${NUM_SHARDS}"
      ;;
    amass)
      out_dir="${OUT_BASE}/${AMASS_OUT_NAME}"
      mkdir -p "${out_dir}"
      recursive_manifest "${AMASS_ROOT}" "${out_dir}/manifest.json"
      run_shards "${AMASS_ROOT}" "${out_dir}" "${AMASS_NUM_SHARDS}"
      ;;
    *)
      echo "[hgpt-v2] ERROR unknown split ${split}" >&2
      exit 3
      ;;
  esac
done

echo "[hgpt-v2] done $(date)"
