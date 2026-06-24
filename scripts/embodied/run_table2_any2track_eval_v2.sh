#!/usr/bin/env bash
# Run Any2Track/OpenTrack Table-2 tracker evaluation with the corrected metrics.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/output/venvs/opentrack_eval/bin/python}"
OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/any2track_open}"
SPLITS="${SPLITS:-lafan1 wild}"
NUM_SHARDS="${NUM_SHARDS:-8}"
AMASS_NUM_SHARDS="${AMASS_NUM_SHARDS:-16}"
MAX_STEPS="${MAX_STEPS:-}"
AMASS_MAX_STEPS="${AMASS_MAX_STEPS:-600}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
QUAT_ORDER="${QUAT_ORDER:-xyzw}"
AMASS_OUT_NAME="${AMASS_OUT_NAME:-amass_v3_xyzw}"
LAFAN_OUT_NAME="${LAFAN_OUT_NAME:-lafan1_v2}"
WILD_OUT_NAME="${WILD_OUT_NAME:-wild_v2}"

OPENTRACK_ROOT="${OPENTRACK_ROOT:-${PROJECT_ROOT}/ref_repo/OpenTrack}"
XML_PATH="${XML_PATH:-${OPENTRACK_ROOT}/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml}"
CONFIG_PATH="${CONFIG_PATH:-${OPENTRACK_ROOT}/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/config.json}"
ONNX_PATH="${ONNX_PATH:-${OPENTRACK_ROOT}/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/model.onnx}"

LAFAN_ROOT="${LAFAN_ROOT:-${PROJECT_ROOT}/data/LAFAN1_Retargeted_for_G1/UnitreeG1}"
LAFAN_FROZEN_MANIFEST="${LAFAN_FROZEN_MANIFEST:-${PROJECT_ROOT}/output/opentrack_lafan1_g1/local_py311_full_localmetric_20260604_233656/manifests/all.json}"
WILD_ROOT="${WILD_ROOT:-${PROJECT_ROOT}/output/heldout_frozen_score}"
AMASS_INPUT_ROOT="${AMASS_INPUT_ROOT:-${PROJECT_ROOT}/data/AMASS_Retarged_for_G1/g1}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_BASE}"

PY_VER_RAW="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 8) else 1)
PY
then
  :
elif [[ -x /usr/local/bin/python3.10 ]]; then
  echo "[any2track-v2] ${PYTHON_BIN} is Python ${PY_VER_RAW}; switching to /usr/local/bin/python3.10"
  PYTHON_BIN="/usr/local/bin/python3.10"
elif [[ -x /usr/bin/python3.11 ]]; then
  echo "[any2track-v2] ${PYTHON_BIN} is Python ${PY_VER_RAW}; switching to /usr/bin/python3.11"
  PYTHON_BIN="/usr/bin/python3.11"
else
  echo "[any2track-v2] ERROR: need Python >=3.8, got ${PY_VER_RAW} from ${PYTHON_BIN}" >&2
  exit 4
fi

# The venv was created through the /apdcephfs/AILab_DHA realpath, while Taiji
# containers often expose only /apdcephfs_cq11.  Make the mounted venv packages
# visible even when Python resolves sys.prefix to the unavailable realpath.
PYTHON_MINOR="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
TAIJI_SITE="${TAIJI_SITE:-${PROJECT_ROOT}/output/venvs/opentrack_eval_taiji_py${PYTHON_MINOR}/site-packages}"
export PYTHONPATH="${TAIJI_SITE}:${PROJECT_ROOT}/output/venvs/opentrack_eval/lib/python${PYTHON_MINOR}/site-packages:${PROJECT_ROOT}/output/venvs/opentrack_eval/lib64/python${PYTHON_MINOR}/site-packages:${PYTHONPATH:-}"

ensure_eval_deps() {
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import mujoco, numpy, onnxruntime, scipy, tqdm
PY
  then
    return
  fi
  echo "[any2track-v2] installing eval deps into ${TAIJI_SITE}"
  mkdir -p "${TAIJI_SITE}"
  "${PYTHON_BIN}" -m pip install \
    --target "${TAIJI_SITE}" \
    --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
    --upgrade \
    "numpy==1.26.0" "scipy==1.12.0" "mujoco==3.1.6" \
    "onnxruntime==1.19.2" "tqdm==4.68.3"
}

echo "[any2track-v2] start $(date)"
echo "[any2track-v2] host=$(hostname)"
echo "[any2track-v2] python=${PYTHON_BIN}"
echo "[any2track-v2] pythonpath=${PYTHONPATH}"
ensure_eval_deps
"${PYTHON_BIN}" - <<'PY'
import onnxruntime as ort
import sys
print("python", sys.version.split()[0])
print("onnxruntime", ort.__version__)
PY

for required in "${PYTHON_BIN}" "${XML_PATH}" "${CONFIG_PATH}" "${ONNX_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[any2track-v2] ERROR missing ${required}" >&2
    exit 2
  fi
done

make_shards() {
  local manifest="$1"
  local out_dir="$2"
  local num_shards="$3"
  mkdir -p "${out_dir}/manifests"
  "${PYTHON_BIN}" - "${manifest}" "${out_dir}/manifests" "${num_shards}" <<'PY'
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

run_eval_shards() {
  local motion_dir="$1"
  local out_dir="$2"
  local num_shards="$3"
  local max_steps="$4"
  echo "[any2track-v2] eval motion_dir=${motion_dir} out=${out_dir} shards=${num_shards} max_steps=${max_steps:-full}"
  for shard in $(seq 0 $((num_shards - 1))); do
    local manifest="${out_dir}/manifests/shard_${shard}.json"
    local count
    count="$("${PYTHON_BIN}" -c "import json; print(len(json.load(open('${manifest}'))))")"
    if [[ "${count}" == "0" ]]; then
      echo "[any2track-v2] shard ${shard}: empty"
      continue
    fi
    extra=()
    if [[ -n "${max_steps}" ]]; then
      extra+=(--max-steps "${max_steps}")
    fi
    (
      "${PYTHON_BIN}" scripts/embodied/eval_opentrack_onnx_mujoco.py \
        --motion-dir "${motion_dir}" \
        --manifest "${manifest}" \
        --xml "${XML_PATH}" \
        --config "${CONFIG_PATH}" \
        --onnx "${ONNX_PATH}" \
        --output-json "${out_dir}/eval_shard_${shard}.json" \
        --output-csv "${out_dir}/eval_shard_${shard}.csv" \
        "${extra[@]}" \
        > "${out_dir}/eval_shard_${shard}.log" 2>&1
    ) &
  done
  wait
  "${PYTHON_BIN}" scripts/embodied/aggregate_opentrack_eval.py --eval-root "${out_dir}"
}

prepare_score_manifest() {
  local score_json="$1"
  local manifest="$2"
  "${PYTHON_BIN}" - "${score_json}" "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

score = json.loads(Path(sys.argv[1]).read_text())
names = [f"h{int(row['idx']):03d}_gen" for row in score["rows"]]
Path(sys.argv[2]).write_text(json.dumps(names, indent=2) + "\n")
print(f"manifest={sys.argv[2]} motions={len(names)}")
PY
}

prepare_amass() {
  local out_dir="$1"
  local motion_dir="${out_dir}/UnitreeG1"
  if [[ -d "${motion_dir}" && -n "$(find "${motion_dir}" -maxdepth 1 -name '*.npz' -print -quit)" ]]; then
    echo "[any2track-v2] AMASS converted dir already exists: ${motion_dir}"
    return
  fi
  mkdir -p "${motion_dir}"
  echo "[any2track-v2] converting AMASS into ${motion_dir}"
  for shard in $(seq 0 $((AMASS_NUM_SHARDS - 1))); do
    (
      "${PYTHON_BIN}" scripts/embodied/convert_amass_g1_to_opentrack_npz.py \
        --input-dir "${AMASS_INPUT_ROOT}" \
        --output-dir "${motion_dir}" \
        --xml "${XML_PATH}" \
        --output-fps "${OUTPUT_FPS}" \
        --quat-order "${QUAT_ORDER}" \
        --num-rank "${AMASS_NUM_SHARDS}" \
        --slurm-rank "${shard}" \
        --manifest "${out_dir}/convert_manifest_shard_${shard}.json" \
        --force \
        > "${out_dir}/convert_shard_${shard}.log" 2>&1
    ) &
  done
  wait
}

prepare_all_manifest_from_dir() {
  local motion_dir="$1"
  local manifest="$2"
  "${PYTHON_BIN}" - "${motion_dir}" "${manifest}" <<'PY'
import json
import sys
from pathlib import Path
motion_dir = Path(sys.argv[1])
names = sorted(p.stem for p in motion_dir.glob("*.npz"))
Path(sys.argv[2]).write_text(json.dumps(names, indent=2) + "\n")
print(f"manifest={sys.argv[2]} motions={len(names)}")
PY
}

for split in ${SPLITS}; do
  case "${split}" in
    lafan1)
      out_dir="${OUT_BASE}/${LAFAN_OUT_NAME}"
      mkdir -p "${out_dir}"
      cp "${LAFAN_FROZEN_MANIFEST}" "${out_dir}/manifest.json"
      make_shards "${out_dir}/manifest.json" "${out_dir}" "${NUM_SHARDS}"
      run_eval_shards "${LAFAN_ROOT}" "${out_dir}" "${NUM_SHARDS}" "${MAX_STEPS}"
      ;;
    wild)
      out_dir="${OUT_BASE}/${WILD_OUT_NAME}"
      mkdir -p "${out_dir}"
      prepare_score_manifest "${WILD_ROOT}/heldout_score.json" "${out_dir}/manifest.json"
      make_shards "${out_dir}/manifest.json" "${out_dir}" "${NUM_SHARDS}"
      run_eval_shards "${WILD_ROOT}" "${out_dir}" "${NUM_SHARDS}" "${MAX_STEPS}"
      ;;
    amass)
      out_dir="${OUT_BASE}/${AMASS_OUT_NAME}"
      mkdir -p "${out_dir}"
      prepare_amass "${out_dir}"
      prepare_all_manifest_from_dir "${out_dir}/UnitreeG1" "${out_dir}/manifest.json"
      make_shards "${out_dir}/manifest.json" "${out_dir}" "${AMASS_NUM_SHARDS}"
      run_eval_shards "${out_dir}/UnitreeG1" "${out_dir}" "${AMASS_NUM_SHARDS}" "${AMASS_MAX_STEPS}"
      ;;
    *)
      echo "[any2track-v2] ERROR unknown split ${split}" >&2
      exit 3
      ;;
  esac
done

echo "[any2track-v2] done $(date)"
