#!/usr/bin/env bash
# Multi-node (4 host x 8 A100-40GB = 32 GPU) T2M-only launcher for the Taiji
# INTERACTIVE keyframe instance. Run this on EACH of host 0..3 with NODE_RANK
# set to that host's index. Uses the ISOLATED shared venv (.venv_t2m_a100) so
# mmengine/overrides/transformers/mmcv-lite come from the venv (visible on every
# host via cephfs) while torch/accelerate come from the container image.
#
# NCCL is left on auto-detect (IB + socket): taiji_dist_train.sh runs PhysFlow
# multinode the same way, so no NCCL_SOCKET_IFNAME / NCCL_IB_HCA override needed.
#
# Usage (per host):
#   bash scripts/eval/launch_t2m_mn.sh <NODE_RANK>
# env overrides: NNODES(4) GPUS(8) MASTER_ADDR(30.72.67.8) MASTER_PORT(29500)
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
VENV="$ROOT/.venv_t2m_a100"
PY="$VENV/bin/python"
CONFIG=configs/hymotion_m2m/hymotion_m2m_t2m_only_from_lite_046b.py

NODE_RANK="${1:?usage: launch_t2m_mn.sh <NODE_RANK>}"
NNODES="${NNODES:-4}"
GPUS="${GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-30.72.67.8}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_PROCESSES=$((NNODES * GPUS))
LOGF="$ROOT/work_dirs/hymotion_m2m_t2m_only_from_lite/mn32_rank${NODE_RANK}.log"
mkdir -p "$(dirname "$LOGF")"

if [ ! -x "$PY" ]; then echo "VENV_PYTHON_MISSING $PY"; exit 3; fi

# Stop any prior single-node (8-GPU) run / stale ranks on this host.
tmux kill-session -t t2m 2>/dev/null || true
pkill -f 'accelerate' 2>/dev/null || true
pkill -f 'tools/train.py' 2>/dev/null || true
sleep 2

tmux new-session -d -s t2m \
  "cd $ROOT && \
   export PYTHONPATH=$ROOT HFTRAINER_SKIP_AUTOREGISTER=0 \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          PYTHONFAULTHANDLER=1 NCCL_DEBUG=WARN && \
   $PY -m accelerate.commands.launch \
     --num_machines=$NNODES --num_processes=$NUM_PROCESSES --machine_rank=$NODE_RANK \
     --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
     --mixed_precision=no --dynamo_backend=no \
     --rdzv_backend=static \
     tools/train.py $CONFIG > $LOGF 2>&1"
sleep 3
echo "--- tmux sessions ---"
tmux ls 2>&1
echo "MN_LAUNCHED rank=$NODE_RANK/$NNODES procs=$NUM_PROCESSES master=$MASTER_ADDR:$MASTER_PORT log=$LOGF py=$PY"
