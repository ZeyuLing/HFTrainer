#!/usr/bin/env bash
# Run BeyondMimic per-motion specialists on the unified Table-2 protocol.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1}"
SPLITS="${SPLITS:-lafan1_fixed600 amass_test_fixed600 wild_clean_fixed600}"
SPLITS="${SPLITS//,/ }"
TOTAL_SHARDS="${TOTAL_SHARDS:-768}"
LOCAL_SHARDS="${LOCAL_SHARDS:-8}"
SHARD_START="${SHARD_START:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
BEYONDMIMIC_MAX_ITERATIONS="${BEYONDMIMIC_MAX_ITERATIONS:-3000}"
BEYONDMIMIC_NUM_ENVS="${BEYONDMIMIC_NUM_ENVS:-2048}"
BEYONDMIMIC_PLAY_VIDEO_LENGTH="${BEYONDMIMIC_PLAY_VIDEO_LENGTH:-1000}"
BEYONDMIMIC_MODE="${BEYONDMIMIC_MODE:-table2}"
BEYONDMIMIC_VENV="${BEYONDMIMIC_VENV:-/root/beyondmimic_isaacsim_py311}"
FORCE_EVAL="${FORCE_EVAL:-0}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROTOCOL_ROOT}/logs"
HOST_TAG="$(hostname)_beyondmimic_s${SHARD_START}_n${LOCAL_SHARDS}"
exec > >(tee -a "${PROTOCOL_ROOT}/logs/run_${HOST_TAG}.log") 2>&1

echo "[table2-beyondmimic] start $(date)"
echo "[table2-beyondmimic] host=$(hostname) shard_start=${SHARD_START} local_shards=${LOCAL_SHARDS} total_shards=${TOTAL_SHARDS}"
echo "[table2-beyondmimic] splits=${SPLITS}"
echo "[table2-beyondmimic] max_iterations=${BEYONDMIMIC_MAX_ITERATIONS} num_envs=${BEYONDMIMIC_NUM_ENVS}"
echo "[table2-beyondmimic] gpu_list=${GPU_LIST} max_parallel=${MAX_PARALLEL}"

if [[ ! -f "${PROTOCOL_ROOT}/protocol_summary.json" ]]; then
  python3 scripts/embodied/build_table2_unified_protocol_inputs.py --out-root "${PROTOCOL_ROOT}"
fi

make_shards() {
  local split="$1"
  local manifest="${PROTOCOL_ROOT}/inputs/${split}/manifest.json"
  local out_dir="${PROTOCOL_ROOT}/manifests/${split}"
  mkdir -p "${out_dir}"
  python3 - "${manifest}" "${out_dir}" "${TOTAL_SHARDS}" <<'PY'
import json, os, sys, uuid
from pathlib import Path
manifest = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
num = int(sys.argv[3])
names = json.loads(manifest.read_text())
shards = [[] for _ in range(num)]
for i, name in enumerate(names):
    shards[i % num].append(name)
token = f".{os.uname().nodename}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
for i, shard in enumerate(shards):
    path = out_dir / f"shard_{i}.json"
    text = json.dumps(shard, indent=2) + "\n"
    if path.is_file():
        try:
            if json.loads(path.read_text()) == shard:
                continue
        except json.JSONDecodeError:
            pass
    tmp = path.with_name(path.name + token)
    tmp.write_text(text)
    tmp.replace(path)
print(f"{manifest}: motions={len(names)} shards={num}")
PY
}

manifest_names() {
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
for name in json.loads(Path(sys.argv[1]).read_text()):
    print(name)
PY
}

wrap_summary() {
  local split="$1"
  local shard="$2"
  local name="$3"
  local log_dir="$4"
  local out_dir="$5"
  python3 - "${split}" "${shard}" "${name}" "${log_dir}" "${out_dir}" <<'PY'
import json, os, sys, uuid
from pathlib import Path
split, shard, name, log_dir, out_dir = sys.argv[1:]
log_dir = Path(log_dir)
out_dir = Path(out_dir)
src = log_dir / "summary.json"
if not src.is_file():
    raise SystemExit(f"missing summary: {src}")
data = json.loads(src.read_text())
motions = data.get("motions", [])
if isinstance(motions, list):
    motions = {row.get("motion", name): row for row in motions}
payload = {
    "summary": data.get("summary", {}),
    "motions": motions,
    "input": {
        "split": split,
        "shard": int(shard),
        "motion": name,
        "log_dir": str(log_dir),
    },
}
out_dir.mkdir(parents=True, exist_ok=True)
tmp = out_dir / f".summary.json.{os.uname().nodename}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(out_dir / "summary.json")
PY
}

gpu_array=(${GPU_LIST})
running=0
gpu_i=0
pids=()

wait_one_batch() {
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  pids=()
  running=0
  if [[ "${failed}" != "0" ]]; then
    exit 6
  fi
}

run_case_bg() {
  local split="$1"
  local shard="$2"
  local name="$3"
  local gpu="$4"
  local source_npz="${PROTOCOL_ROOT}/inputs/${split}/npz/${name}.npz"
  local out_dir="${PROTOCOL_ROOT}/runs/beyondmimic/${split}/shard_${shard}"
  mkdir -p "${out_dir}"
  if [[ "${FORCE_EVAL}" != "1" && -s "${out_dir}/summary.json" ]]; then
    echo "[table2-beyondmimic] ${split} shard ${shard} ${name}: already done"
    return
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export ROOT="${PROJECT_ROOT}"
    export BEYONDMIMIC_MODE
    export BEYONDMIMIC_MOTION_NAME="${name}"
    export BEYONDMIMIC_SOURCE_NPZ="${source_npz}"
    export BEYONDMIMIC_EVAL_REF_NPZ="${source_npz}"
    export BEYONDMIMIC_MAX_ITERATIONS
    export BEYONDMIMIC_NUM_ENVS
    export BEYONDMIMIC_PLAY_VIDEO_LENGTH
    export BEYONDMIMIC_VENV
    export BEYONDMIMIC_TAG="table2_bm_${split}_s${shard}_${name}"
    echo "[table2-beyondmimic] launch split=${split} shard=${shard} motion=${name} gpu=${gpu}"
    bash scripts/embodied/taiji_beyondmimic_official_train.sh
    log_dir="${PROJECT_ROOT}/output/beyondmimic_official/table2_bm_${split}_s${shard}_${name}"
    wrap_summary "${split}" "${shard}" "${name}" "${log_dir}" "${out_dir}"
    echo "[table2-beyondmimic] done split=${split} shard=${shard} motion=${name}"
  ) > "${out_dir}/run.log" 2>&1 &
  pids+=("$!")
  running=$((running + 1))
  gpu_i=$(((gpu_i + 1) % ${#gpu_array[@]}))
  if (( running >= MAX_PARALLEL )); then
    wait_one_batch
  fi
}

for split in ${SPLITS}; do
  make_shards "${split}"
  for shard in $(seq "${SHARD_START}" $((SHARD_START + LOCAL_SHARDS - 1))); do
    manifest="${PROTOCOL_ROOT}/manifests/${split}/shard_${shard}.json"
    if [[ ! -f "${manifest}" ]]; then
      echo "[table2-beyondmimic] missing manifest ${manifest}, skip"
      continue
    fi
    while IFS= read -r name; do
      [[ -z "${name}" ]] && continue
      run_case_bg "${split}" "${shard}" "${name}" "${gpu_array[${gpu_i}]}"
    done < <(manifest_names "${manifest}")
  done
done

if (( running > 0 )); then
  wait_one_batch
fi

python3 scripts/embodied/aggregate_table2_beyondmimic.py --protocol-root "${PROTOCOL_ROOT}" --allow-missing
echo "[table2-beyondmimic] done $(date)"
