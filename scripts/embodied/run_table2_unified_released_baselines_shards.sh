#!/usr/bin/env bash
# Run released tracker baselines on the unified Table-2 fixed-window protocol.
#
# Launch one copy per Taiji host.  For a 4-host A100 instance:
#   SHARD_START=0 LOCAL_SHARDS=8 TOTAL_SHARDS=32 bash ...
#   SHARD_START=8 LOCAL_SHARDS=8 TOTAL_SHARDS=32 bash ...
#   ...
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1}"
METHODS="${METHODS:-any2track humanoid_gpt}"
SPLITS="${SPLITS:-lafan1_fixed600 wild_clean_fixed600 amass_fixed600}"
METHODS="${METHODS//,/ }"
SPLITS="${SPLITS//,/ }"
TOTAL_SHARDS="${TOTAL_SHARDS:-32}"
LOCAL_SHARDS="${LOCAL_SHARDS:-8}"
SHARD_START="${SHARD_START:-0}"
COMPLETE_THRESH="${COMPLETE_THRESH:-0.95}"
HGPT_TIMEOUT_S="${HGPT_TIMEOUT_S:-7200}"
HGPT_DEVICE="${HGPT_DEVICE:-cpu}"

OPENTRACK_ROOT="${OPENTRACK_ROOT:-${PROJECT_ROOT}/ref_repo/OpenTrack}"
XML_PATH="${XML_PATH:-${OPENTRACK_ROOT}/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml}"
CONFIG_PATH="${CONFIG_PATH:-${OPENTRACK_ROOT}/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/config.json}"
ONNX_PATH="${ONNX_PATH:-${OPENTRACK_ROOT}/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/model.onnx}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/output/venvs/opentrack_eval/bin/python}"
HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-/dev/shm/hgpt_venv311/bin/python}"
if [[ -d "${HGPT_PYTHON}" ]]; then
  HGPT_PYTHON="${HGPT_PYTHON}/bin/python"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${PROTOCOL_ROOT}/logs"
HOST_TAG="$(hostname)_s${SHARD_START}_n${LOCAL_SHARDS}"
exec > >(tee -a "${PROTOCOL_ROOT}/logs/run_${HOST_TAG}.log") 2>&1

echo "[unified-baselines] start $(date)"
echo "[unified-baselines] host=$(hostname) shard_start=${SHARD_START} local_shards=${LOCAL_SHARDS} total_shards=${TOTAL_SHARDS}"
echo "[unified-baselines] methods=${METHODS}"
echo "[unified-baselines] splits=${SPLITS}"

if [[ "${SKIP_BUILD:-0}" != "1" && ! -f "${PROTOCOL_ROOT}/protocol_summary.json" ]]; then
  python3 scripts/embodied/build_table2_unified_protocol_inputs.py --out-root "${PROTOCOL_ROOT}"
fi

make_shards() {
  local split="$1"
  local manifest="${PROTOCOL_ROOT}/inputs/${split}/manifest.json"
  local out_dir="${PROTOCOL_ROOT}/manifests/${split}"
  mkdir -p "${out_dir}"
  python3 - "${manifest}" "${out_dir}" "${TOTAL_SHARDS}" <<'PY'
import json, sys
from pathlib import Path
manifest = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
num = int(sys.argv[3])
names = json.loads(manifest.read_text())
shards = [[] for _ in range(num)]
for i, name in enumerate(names):
    shards[i % num].append(name)
for i, shard in enumerate(shards):
    (out_dir / f"shard_{i}.json").write_text(json.dumps(shard, indent=2) + "\n")
print(f"{manifest}: motions={len(names)} shards={num}")
PY
}

ensure_opentrack_python() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
  fi
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 8) else 1)
PY
  then
    if [[ -x /usr/local/bin/python3.10 ]]; then
      PYTHON_BIN=/usr/local/bin/python3.10
    elif [[ -x /usr/bin/python3.11 ]]; then
      PYTHON_BIN=/usr/bin/python3.11
    elif [[ -x /usr/bin/python3.10 ]]; then
      PYTHON_BIN=/usr/bin/python3.10
    else
      echo "[unified-baselines] ERROR: need Python >=3.8 for OpenTrack eval, got $(${PYTHON_BIN} -V 2>&1)" >&2
      exit 4
    fi
  fi
  local py_minor
  py_minor="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  export TAIJI_SITE="${TAIJI_SITE:-${PROJECT_ROOT}/output/venvs/opentrack_eval_taiji_py${py_minor}/site-packages}"
  export PYTHONPATH="${TAIJI_SITE}:${PROJECT_ROOT}/output/venvs/opentrack_eval/lib/python${py_minor}/site-packages:${PROJECT_ROOT}/output/venvs/opentrack_eval/lib64/python${py_minor}/site-packages:${PYTHONPATH:-}"
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import mujoco, numpy, onnxruntime, scipy, tqdm
PY
  then
    return
  fi
  echo "[unified-baselines] installing OpenTrack eval deps into ${TAIJI_SITE}"
  mkdir -p "${TAIJI_SITE}"
  "${PYTHON_BIN}" -m pip install \
    --target "${TAIJI_SITE}" \
    --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
    --upgrade \
    "numpy==1.26.0" "scipy==1.12.0" "mujoco==3.1.6" \
    "onnxruntime==1.19.2" "tqdm==4.68.3"
}

run_any2track_split() {
  local split="$1"
  local motion_dir="${PROTOCOL_ROOT}/inputs/${split}/npz"
  local out_dir="${PROTOCOL_ROOT}/runs/any2track/${split}"
  mkdir -p "${out_dir}"
  echo "[unified-baselines] Any2Track split=${split}"
  local pids=()
  for shard in $(seq "${SHARD_START}" $((SHARD_START + LOCAL_SHARDS - 1))); do
    local manifest="${PROTOCOL_ROOT}/manifests/${split}/shard_${shard}.json"
    local count
    count="$("${PYTHON_BIN}" -c "import json; print(len(json.load(open('${manifest}'))))")"
    if [[ "${count}" == "0" ]]; then
      echo "[unified-baselines] Any2Track ${split} shard ${shard}: empty"
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES="$(( (shard - SHARD_START) % 8 ))"
      "${PYTHON_BIN}" scripts/embodied/eval_opentrack_onnx_mujoco.py \
        --motion-dir "${motion_dir}" \
        --manifest "${manifest}" \
        --xml "${XML_PATH}" \
        --config "${CONFIG_PATH}" \
        --onnx "${ONNX_PATH}" \
        --output-json "${out_dir}/eval_shard_${shard}.json" \
        --output-csv "${out_dir}/eval_shard_${shard}.csv" \
        > "${out_dir}/eval_shard_${shard}.log" 2>&1
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
}

ensure_hgpt_python() {
  if [[ -x "${HGPT_PYTHON}" ]] && ! env -u PYTHONPATH "${HGPT_PYTHON}" - <<'PY' >/dev/null 2>&1
import numpy, mujoco, onnxruntime, scipy, tyro, tree
PY
  then
    echo "[unified-baselines] removing broken Humanoid-GPT env ${HGPT_PYTHON%/bin/python}"
    rm -rf "${HGPT_PYTHON%/bin/python}"
  fi
  if [[ ! -x "${HGPT_PYTHON}" ]]; then
    echo "[unified-baselines] building Humanoid-GPT env"
    env -u PYTHONPATH \
      PHYSFLOW_HGPT_VENV="${PHYSFLOW_HGPT_VENV:-${HGPT_PYTHON%/bin/python}}" \
      PHYSFLOW_HGPT_ORT_PACKAGE="${PHYSFLOW_HGPT_ORT_PACKAGE:-onnxruntime<1.24}" \
      bash scripts/embodied/physflow_hgpt_node_setup.sh
  fi
  HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-/dev/shm/hgpt_venv311/bin/python}"
  if [[ -d "${HGPT_PYTHON}" ]]; then
    HGPT_PYTHON="${HGPT_PYTHON}/bin/python"
  fi
}

run_hgpt_split() {
  local split="$1"
  local motion_dir="${PROTOCOL_ROOT}/inputs/${split}/npz"
  local out_dir="${PROTOCOL_ROOT}/runs/humanoid_gpt/${split}"
  mkdir -p "${out_dir}"
  echo "[unified-baselines] Humanoid-GPT split=${split}"
  local pids=()
  for shard in $(seq "${SHARD_START}" $((SHARD_START + LOCAL_SHARDS - 1))); do
    local manifest="${PROTOCOL_ROOT}/manifests/${split}/shard_${shard}.json"
    local count
    count="$("${HGPT_PYTHON}" -c "import json; print(len(json.load(open('${manifest}'))))")"
    if [[ "${count}" == "0" ]]; then
      echo "[unified-baselines] Humanoid-GPT ${split} shard ${shard}: empty"
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES="$(( (shard - SHARD_START) % 8 ))"
      env PYTHONPATH="${PROJECT_ROOT}" "${HGPT_PYTHON}" scripts/embodied/run_table2_hgpt_eval.py \
        --motion-dir "${motion_dir}" \
        --manifest "${manifest}" \
        --out-dir "${out_dir}/shard_${shard}" \
        --hgpt-python "${HGPT_PYTHON}" \
        --device "${HGPT_DEVICE}" \
        --complete-thresh "${COMPLETE_THRESH}" \
        --timeout-s "${HGPT_TIMEOUT_S}" \
        > "${out_dir}/shard_${shard}.log" 2>&1
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
}

for split in ${SPLITS}; do
  make_shards "${split}"
done

for method in ${METHODS}; do
  case "${method}" in
    any2track)
      ensure_opentrack_python
      for split in ${SPLITS}; do run_any2track_split "${split}"; done
      ;;
    humanoid_gpt)
      ensure_hgpt_python
      for split in ${SPLITS}; do run_hgpt_split "${split}"; done
      ;;
    *)
      echo "[unified-baselines] ERROR unknown method ${method}" >&2
      exit 3
      ;;
  esac
done

echo "[unified-baselines] done $(date)"
