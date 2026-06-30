#!/usr/bin/env bash
# Multi-host-safe full-AMASS stress runner for V100 elastic jobs.
#
# Each Taiji pod computes its shard range from INDEX and SHARD_BASE_OFFSET:
#   shard_start = SHARD_BASE_OFFSET + INDEX * LOCAL_SHARDS
#
# The script writes into a separate protocol root and never removes shared
# outputs, so multiple elastic jobs can run concurrently without clobbering
# each other.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
export BASE_PROTOCOL_ROOT="${BASE_PROTOCOL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1_v100elastic384}"
export SPLITS="${SPLITS:-amass_fixed600}"
export TOTAL_SHARDS="${TOTAL_SHARDS:-384}"
export LOCAL_SHARDS="${LOCAL_SHARDS:-8}"
export SHARD_BASE_OFFSET="${SHARD_BASE_OFFSET:-0}"
if [[ -z "${INDEX:-}" && -z "${JOB_RANK:-}" && "${REQUIRE_TAIJI_INDEX:-1}" == "1" ]]; then
  echo "[table2-v100elastic] ERROR: missing Taiji INDEX/JOB_RANK; refusing to duplicate shard ranges across hosts." >&2
  exit 7
fi
export NODE_RANK="${NODE_RANK:-${INDEX:-${JOB_RANK:-0}}}"
export SHARD_START="${SHARD_START:-$((SHARD_BASE_OFFSET + NODE_RANK * LOCAL_SHARDS))}"
export FORCE_CONVERT="${FORCE_CONVERT:-0}"
export FORCE_PACK="${FORCE_PACK:-0}"
export FORCE_EVAL="${FORCE_EVAL:-0}"
export RUN_NODE_SETUP="${RUN_NODE_SETUP:-1}"
export SKIP_BUILD="${SKIP_BUILD:-1}"
export NUM_ENVS="${NUM_ENVS:-128}"
export REFERENCE_FPS="${REFERENCE_FPS:-30}"
export TRACKER_CONTROL_FPS="${TRACKER_CONTROL_FPS:-50}"
export MAX_REFERENCE_FRAMES="${MAX_REFERENCE_FRAMES:-600}"
export OUTPUT_FPS="${OUTPUT_FPS:-${TRACKER_CONTROL_FPS}}"
if [[ -z "${MAX_EVAL_STEPS+x}" ]]; then
  export MAX_EVAL_STEPS="$(( (MAX_REFERENCE_FRAMES * TRACKER_CONTROL_FPS + REFERENCE_FPS - 1) / REFERENCE_FPS ))"
else
  export MAX_EVAL_STEPS
fi
export PHYSFLOW_TRACKER_PYTHON_CMD="${PHYSFLOW_TRACKER_PYTHON_CMD:-/root/physflow_isaacgym_py38_cu118/bin/python}"

cd "${PROJECT_ROOT}"
if [[ "${RUN_FULL_AMASS_STRESS:-0}" != "1" ]]; then
  echo "[table2-v100elastic] refusing to run full-AMASS stress sweep by default." >&2
  echo "[table2-v100elastic] set RUN_FULL_AMASS_STRESS=1 to run this optional stress test." >&2
  exit 9
fi
mkdir -p "${PROTOCOL_ROOT}/logs"
HOST_TAG="$(hostname)_rank${NODE_RANK}_s${SHARD_START}_n${LOCAL_SHARDS}"
LOG="${PROTOCOL_ROOT}/logs/table2_v100elastic_${HOST_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "[table2-v100elastic] start $(date)"
echo "[table2-v100elastic] host=$(hostname) index=${INDEX:-unset} node_rank=${NODE_RANK}"
echo "[table2-v100elastic] protocol_root=${PROTOCOL_ROOT}"
echo "[table2-v100elastic] base_protocol_root=${BASE_PROTOCOL_ROOT}"
echo "[table2-v100elastic] total_shards=${TOTAL_SHARDS} local_shards=${LOCAL_SHARDS} shard_start=${SHARD_START}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

python3 - <<'PY'
import json
import os
import socket
import uuid
from pathlib import Path

base = Path(os.environ["BASE_PROTOCOL_ROOT"])
root = Path(os.environ["PROTOCOL_ROOT"])
split = "amass_fixed600"
total = int(os.environ["TOTAL_SHARDS"])
token = f"{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}"

summary = json.loads((base / "protocol_summary.json").read_text())
expected = {
    "amass_body_quat_order": "wxyz",
    "amass_root_frame_correction": "none",
}
bad = {k: (summary.get(k), v) for k, v in expected.items() if summary.get(k) != v}
if bad:
    raise SystemExit(f"Refusing to run stale AMASS protocol: {bad}")

root.mkdir(parents=True, exist_ok=True)
summary_path = root / "protocol_summary.json"
tmp = summary_path.with_suffix(f".{token}.tmp")
tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
tmp.replace(summary_path)

inputs_link = root / "inputs"
if not inputs_link.exists():
    tmp_link = root / f".inputs.{token}.tmp"
    tmp_link.symlink_to(base / "inputs", target_is_directory=True)
    try:
        tmp_link.replace(inputs_link)
    except FileExistsError:
        tmp_link.unlink(missing_ok=True)

names = json.loads((base / "inputs" / split / "manifest.json").read_text())
out_dir = root / "manifests" / split
out_dir.mkdir(parents=True, exist_ok=True)
shards = [[] for _ in range(total)]
for i, name in enumerate(names):
    shards[i % total].append(name)
for i, shard in enumerate(shards):
    path = out_dir / f"shard_{i}.json"
    tmp = out_dir / f".shard_{i}.{token}.tmp"
    tmp.write_text(json.dumps(shard, indent=2) + "\n")
    tmp.replace(path)

print({"amass_motions": len(names), "num_shards": total, "min_shard": min(map(len, shards)), "max_shard": max(map(len, shards))})
PY

echo "[table2-v100elastic] running ProtoMotions"
bash scripts/embodied/run_table2_unified_proto_shards.sh

echo "[table2-v100elastic] running Any2Track + Humanoid-GPT"
export METHODS="${METHODS:-any2track humanoid_gpt}"
export PHYSFLOW_HGPT_VENV="${PHYSFLOW_HGPT_VENV:-/dev/shm/hgpt_venv311_gpu}"
export PHYSFLOW_HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-${PHYSFLOW_HGPT_VENV}/bin/python}"
export PHYSFLOW_HGPT_ORT_PACKAGE="${PHYSFLOW_HGPT_ORT_PACKAGE:-onnxruntime<1.24}"
export HGPT_DEVICE="${HGPT_DEVICE:-cuda:0}"
export HGPT_TIMEOUT_S="${HGPT_TIMEOUT_S:-14400}"
bash scripts/embodied/run_table2_unified_released_baselines_shards.sh

echo "[table2-v100elastic] aggregating with allow-missing"
python3 scripts/embodied/aggregate_table2_unified_proto.py --protocol-root "${PROTOCOL_ROOT}" --allow-missing || true
python3 scripts/embodied/aggregate_table2_unified_protocol.py --protocol-root "${PROTOCOL_ROOT}" --allow-missing || true

echo "[table2-v100elastic] done $(date)"
