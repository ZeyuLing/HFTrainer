#!/bin/bash
# Final phase: Quality checking + report generation
# Run AFTER both machines have finished all repairs

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUTDIR=output/cjgame_repair_eval
LOGFILE="$OUTDIR/logs/final_report_$(date +%Y%m%d_%H%M%S).log"

echo "=== Final Report Phase: Starting at $(date) ===" | tee "$LOGFILE"

# Wait for both machines to finish
echo "Waiting for all repair phases to complete..." | tee -a "$LOGFILE"
while [ ! -f "$OUTDIR/logs/machine1_m2m_done.flag" ] || [ ! -f "$OUTDIR/logs/machine2_m2m_done.flag" ]; do
    sleep 30
    echo "  Still waiting... $(date)" | tee -a "$LOGFILE"
done
echo "All repairs complete at $(date)" | tee -a "$LOGFILE"

# Run quality checker + report generation
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_cjgame_repair.py \
    --max-samples 0 \
    --device cuda:0 \
    --skip-mogendit \
    --skip-m2m \
    2>&1 | tee -a "$LOGFILE"

echo "REPORT_DONE" > "$OUTDIR/logs/report_done.flag"
echo "=== Final Report Phase: Complete at $(date) ===" | tee -a "$LOGFILE"
