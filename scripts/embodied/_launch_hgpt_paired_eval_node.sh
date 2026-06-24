#!/usr/bin/env bash
# Detached launcher for the paired HGPT eval (survives the taiji_exec PTY close).
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
mkdir -p work_dirs/physflow_overfit100_hgpt
LOG=work_dirs/physflow_overfit100_hgpt/paired_eval.log
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    setsid bash scripts/embodied/_run_hgpt_paired_eval_node.sh "$@" \
    </dev/null >"$LOG" 2>&1 &
disown
echo "launched paired-eval pid $! -> $LOG"
