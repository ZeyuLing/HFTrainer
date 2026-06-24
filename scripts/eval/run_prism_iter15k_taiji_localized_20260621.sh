#!/usr/bin/env bash
# Taiji multi-host entrypoint for strict official-272 iter15000 generation.
# Uses Taiji INDEX/NODE_LIST to assign disjoint 8-GPU shard blocks and localizes
# the 27GB iter15000 checkpoint on each host before launching workers.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

NGPU="${NGPU:-8}"
HOST_RANK="${INDEX:-0}"
if [[ -n "${NODE_LIST:-}" ]]; then
  HOST_COUNT="$(python3 - <<'PY'
import os
nodes = [x for x in os.environ.get("NODE_LIST", "").split(",") if x]
print(len(nodes) or 1)
PY
)"
else
  HOST_COUNT="${MACHINE_NUM:-1}"
fi

export VARIANT=iter15k_no_kt_no_kafs
export LOCALIZE_CKPT=1
export NGPU
export TOTAL_SHARDS="${TOTAL_SHARDS:-$((HOST_COUNT * NGPU))}"
export SHARD_BASE="${SHARD_BASE:-$((HOST_RANK * NGPU))}"

RUN_ROOT="${RUN_ROOT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_kafs_kt_compare_20260621}"
mkdir -p "$RUN_ROOT/iter15k_no_kt_no_kafs"
LOG="$RUN_ROOT/iter15k_no_kt_no_kafs/taiji_iter15k_host${HOST_RANK}_of${HOST_COUNT}.log"

{
  echo "[taiji-iter15k] $(date -Is) host_rank=$HOST_RANK host_count=$HOST_COUNT total_shards=$TOTAL_SHARDS shard_base=$SHARD_BASE ngpu=$NGPU"
  bash scripts/eval/run_prism_kafs_kt_compare_gen_20260621.sh
} > "$LOG" 2>&1

