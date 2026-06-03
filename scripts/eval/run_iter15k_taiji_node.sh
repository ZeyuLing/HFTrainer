#!/bin/bash
# Multi-host Taiji driver for parallel iter_15000 generation (text cross-attn is
# always mask-free / Wan-style, hardcoded in the pipeline).
# Designed to run ALONGSIDE the lzy_debug_machine_2 tmux run, which occupies
# shard group 0 (shards [0,8) of NSHARDS). This task's nodes occupy the shard
# groups AFTER machine_2, so the union is disjoint. skip-existing dedupes
# against whatever machine_2 already produced in the shared output dir.
#
# Taiji provides per-host env:
#   INDEX      = node rank within this task (0-based)
#   NODE_LIST  = comma-separated host list (len => number of nodes in this task)
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MN="$(python3 -c "import os;print(len(os.environ.get('NODE_LIST','x').split(',')))")"
NODE_IDX="${INDEX:-0}"

# machine_2 = shard group 0; this task's MN nodes = groups 1..MN.
export NGPU=8
export MODE="${MODE:-none}"
export NSHARDS=$(( (MN + 1) * 8 ))
export SHARD_START=$(( (NODE_IDX + 1) * 8 ))

echo "[taiji-infer] INDEX=$NODE_IDX MN=$MN NSHARDS=$NSHARDS SHARD_START=$SHARD_START NGPU=$NGPU"
bash scripts/eval/run_iter15k_node.sh
