#!/usr/bin/env bash
# Detached launcher for the co-evolution smoke (survives taiji_exec PTY close).
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
mkdir -p work_dirs/physflow_coevolve_smoke
LOG=work_dirs/physflow_coevolve_smoke/orchestrator.log
setsid bash scripts/embodied/_run_coevolve_smoke_node.sh "$@" \
    </dev/null >"$LOG" 2>&1 &
disown
echo "launched coevolve-smoke pid $! -> $LOG"
