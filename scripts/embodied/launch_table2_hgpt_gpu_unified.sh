#!/usr/bin/env bash
# Thin launcher for Taiji nodes. Keep the remote command short and put all
# Humanoid-GPT GPU runtime details here.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1}"
export METHODS="${METHODS:-humanoid_gpt}"
export SPLITS="${SPLITS:-lafan1_fixed600,wild_clean_fixed600,amass_fixed600}"
export TOTAL_SHARDS="${TOTAL_SHARDS:-32}"
export LOCAL_SHARDS="${LOCAL_SHARDS:-8}"
export SHARD_START="${SHARD_START:-0}"
export PHYSFLOW_HGPT_VENV="${PHYSFLOW_HGPT_VENV:-/dev/shm/hgpt_venv311_gpu}"
export PHYSFLOW_HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-${PHYSFLOW_HGPT_VENV}/bin/python}"
export PHYSFLOW_HGPT_ORT_PACKAGE="${PHYSFLOW_HGPT_ORT_PACKAGE:-onnxruntime-gpu==1.18.1}"
export HGPT_DEVICE="${HGPT_DEVICE:-cuda:0}"
export HGPT_TIMEOUT_S="${HGPT_TIMEOUT_S:-14400}"

cd "${PROJECT_ROOT}"
exec bash scripts/embodied/run_table2_unified_released_baselines_shards.sh
