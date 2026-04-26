#!/bin/bash
# Launch CJGame repair eval on debug machine 1
# Runs MoGenDIT first (adaptive masks + denoise/ada_denoise repair)
# Then runs half the M2M configs

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUTDIR=output/cjgame_repair_eval
LOGFILE="$OUTDIR/logs/machine1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTDIR/logs"

echo "=== Machine 1: Starting at $(date) ===" | tee "$LOGFILE"

# Phase 1: MoGenDIT (masks + repair) on GPU 0
echo "Phase 1: MoGenDIT (masks + denoise + ada_denoise)" | tee -a "$LOGFILE"
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_cjgame_repair.py \
    --max-samples 0 \
    --device cuda:0 \
    --skip-m2m \
    --skip-checker \
    --skip-report \
    2>&1 | tee -a "$LOGFILE"

echo "Phase 1 done at $(date)" | tee -a "$LOGFILE"
echo "MOGENDIT_DONE" > "$OUTDIR/logs/machine1_mogendit_done.flag"

# Phase 2: M2M configs (first 4 _man configs)
echo "Phase 2: M2M repair (uncond_fm_man, uncond_jit_man, caption_fm_man, caption_jit_man)" | tee -a "$LOGFILE"
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_cjgame_repair.py \
    --max-samples 0 \
    --device cuda:0 \
    --skip-mogendit \
    --skip-checker \
    --skip-report \
    --m2m-configs uncond_fm_man uncond_jit_man caption_fm_man caption_jit_man \
    2>&1 | tee -a "$LOGFILE"

echo "Phase 2 done at $(date)" | tee -a "$LOGFILE"
echo "M2M_PART1_DONE" > "$OUTDIR/logs/machine1_m2m_done.flag"

echo "=== Machine 1: All phases complete at $(date) ===" | tee -a "$LOGFILE"
