#!/usr/bin/env bash
# Multi-host Table-2 released-baseline runner for V100 elastic jobs.
#
# This is a repair/resume launcher: it only runs Any2Track and Humanoid-GPT,
# and the underlying runner skips shard outputs that already exist.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1_v100elastic768}"
export METHODS="${METHODS:-any2track humanoid_gpt}"
export SPLITS="${SPLITS:-amass_test_fixed600,lafan1_fixed600,wild_clean_fixed600}"
export TOTAL_SHARDS="${TOTAL_SHARDS:-768}"
export LOCAL_SHARDS="${LOCAL_SHARDS:-8}"
export SHARD_BASE_OFFSET="${SHARD_BASE_OFFSET:-0}"
if [[ -z "${INDEX:-}" && -z "${JOB_RANK:-}" && "${REQUIRE_TAIJI_INDEX:-1}" == "1" ]]; then
  echo "[table2-baselines-v100elastic] ERROR: missing Taiji INDEX/JOB_RANK; refusing to duplicate shard ranges across hosts." >&2
  exit 7
fi
export NODE_RANK="${NODE_RANK:-${INDEX:-${JOB_RANK:-0}}}"
export SHARD_START="${SHARD_START:-$((SHARD_BASE_OFFSET + NODE_RANK * LOCAL_SHARDS))}"
export FORCE_EVAL="${FORCE_EVAL:-0}"
export SKIP_BUILD="${SKIP_BUILD:-1}"
export HGPT_DEVICE="${HGPT_DEVICE:-cuda:0}"
export HGPT_TIMEOUT_S="${HGPT_TIMEOUT_S:-14400}"
export PHYSFLOW_HGPT_VENV="${PHYSFLOW_HGPT_VENV:-/dev/shm/hgpt_venv311_gpu}"
export PHYSFLOW_HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-${PHYSFLOW_HGPT_VENV}/bin/python}"
export PHYSFLOW_HGPT_ORT_PACKAGE="${PHYSFLOW_HGPT_ORT_PACKAGE:-onnxruntime<1.24}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROTOCOL_ROOT}/logs"
HOST_TAG="$(hostname)_rank${NODE_RANK}_s${SHARD_START}_n${LOCAL_SHARDS}"
LOG="${PROTOCOL_ROOT}/logs/table2_baselines_v100elastic_${HOST_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "[table2-baselines-v100elastic] start $(date)"
echo "[table2-baselines-v100elastic] host=$(hostname) index=${INDEX:-unset} node_rank=${NODE_RANK}"
echo "[table2-baselines-v100elastic] protocol_root=${PROTOCOL_ROOT}"
echo "[table2-baselines-v100elastic] total_shards=${TOTAL_SHARDS} local_shards=${LOCAL_SHARDS} shard_start=${SHARD_START}"
echo "[table2-baselines-v100elastic] methods=${METHODS}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

bash scripts/embodied/run_table2_unified_released_baselines_shards.sh

echo "[table2-baselines-v100elastic] aggregating with allow-missing"
python3 scripts/embodied/aggregate_table2_unified_protocol.py --protocol-root "${PROTOCOL_ROOT}" --allow-missing || true

echo "[table2-baselines-v100elastic] done $(date)"
