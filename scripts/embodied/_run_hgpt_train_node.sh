#!/usr/bin/env bash
# Thin wrapper to launch the PhysFlow HGPT single-GPU training as a fully
# detached (setsid) background job on a Taiji node, so it survives the
# taiji_exec PTY session closing. Logs to work_dirs/physflow_overfit100_hgpt/.
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
mkdir -p work_dirs/physflow_overfit100_hgpt
export PHYSFLOW_HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-/dev/shm/hgpt_venv311/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG=work_dirs/physflow_overfit100_hgpt/train_node2.log
setsid bash scripts/embodied/physflow_hgpt_single_gpu_start.sh \
    configs/physflow/physflow_overfit100_hgpt.py \
    </dev/null >"$LOG" 2>&1 &
disown
echo "launched pid $! -> $LOG"
