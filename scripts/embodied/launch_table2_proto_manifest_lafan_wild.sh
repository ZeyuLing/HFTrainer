#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1}"
export SPLITS="${SPLITS:-lafan1_fixed600 wild_clean_fixed600}"
export SHARD_START="${SHARD_START:-0}"
export LOCAL_SHARDS="${LOCAL_SHARDS:-32}"
export TOTAL_SHARDS="${TOTAL_SHARDS:-32}"
export FORCE_EVAL="${FORCE_EVAL:-1}"
export FORCE_CONVERT="${FORCE_CONVERT:-1}"
export FORCE_PACK="${FORCE_PACK:-1}"
export RUN_NODE_SETUP="${RUN_NODE_SETUP:-1}"
export PHYSFLOW_TRACKER_PYTHON_CMD="${PHYSFLOW_TRACKER_PYTHON_CMD:-/root/physflow_isaacgym_py38_cu118/bin/python}"

cd "${PROJECT_ROOT}"
exec bash scripts/embodied/run_table2_unified_proto_shards.sh
