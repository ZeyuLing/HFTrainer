#!/bin/bash
# Launch CJGame repair eval on debug machine 2
# Waits for MoGenDIT masks from machine 1, then runs the other half of M2M configs

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUTDIR=output/cjgame_repair_eval
LOGFILE="$OUTDIR/logs/machine2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTDIR/logs"

echo "=== Machine 2: Starting at $(date) ===" | tee "$LOGFILE"

# Wait for MoGenDIT masks from machine 1
echo "Waiting for MoGenDIT masks from machine 1..." | tee -a "$LOGFILE"
while [ ! -f "$OUTDIR/logs/machine1_mogendit_done.flag" ]; do
    sleep 30
    echo "  Still waiting... $(date)" | tee -a "$LOGFILE"
done
echo "MoGenDIT masks ready at $(date)" | tee -a "$LOGFILE"

# Phase 2: M2M configs (globalrot variants)
echo "Phase 2: M2M repair (globalrot configs)" | tee -a "$LOGFILE"
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_cjgame_repair.py \
    --max-samples 0 \
    --device cuda:0 \
    --skip-mogendit \
    --skip-checker \
    --skip-report \
    --m2m-configs uncond_fm_man_globalrot uncond_jit_man_globalrot caption_fm_man_globalrot caption_jit_man_globalrot \
    2>&1 | tee -a "$LOGFILE"

echo "Phase 2 done at $(date)" | tee -a "$LOGFILE"
echo "M2M_PART2_DONE" > "$OUTDIR/logs/machine2_m2m_done.flag"

echo "=== Machine 2: All phases complete at $(date) ===" | tee -a "$LOGFILE"
