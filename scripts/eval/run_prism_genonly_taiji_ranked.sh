#!/usr/bin/env bash
# Taiji multi-host wrapper for run_prism_genonly_param.sh.
# Computes global shard ids from Taiji INDEX/NODE_LIST (or JOB_RANK/JOB_COUNT)
# and then delegates all generation details to the existing resumable worker.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

NGPU=${NGPU:-8}

if [ -n "${JOB_COUNT:-}" ]; then
  HOST_RANK=${JOB_RANK:-0}
  MACHINE_NUM=$JOB_COUNT
elif [ -n "${NODE_LIST:-}" ]; then
  HOST_RANK=${INDEX:-0}
  MACHINE_NUM=$(python3 -c "import os; print(len([x for x in os.environ['NODE_LIST'].split(',') if x]))" 2>/dev/null || echo 1)
else
  HOST_RANK=${INDEX:-0}
  MACHINE_NUM=${MACHINE_NUM:-1}
fi

TOTAL_SHARDS=${TOTAL_SHARDS:-$((MACHINE_NUM * NGPU))}
SHARD_BASE=${SHARD_BASE:-$((HOST_RANK * NGPU))}
export NGPU TOTAL_SHARDS SHARD_BASE

echo "[ranked-gen] host=$HOST_RANK/$MACHINE_NUM ngpu=$NGPU total=$TOTAL_SHARDS base=$SHARD_BASE"
bash scripts/eval/run_prism_genonly_param.sh
