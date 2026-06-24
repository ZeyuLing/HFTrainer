#!/usr/bin/env bash
# Detached launcher for the overfit viz data producer (survives taiji_exec PTY).
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
mkdir -p work_dirs/physflow_overfit_eval
LOG=work_dirs/physflow_overfit_eval/viz_node.log
setsid bash scripts/embodied/_run_overfit_viz_node.sh "$@" \
    </dev/null >"$LOG" 2>&1 &
disown
echo "launched overfit-viz pid $! -> $LOG"
