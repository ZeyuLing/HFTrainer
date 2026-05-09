#!/bin/bash
# Run KIMODO on the M2M v2 tasks where it is missing from the dashboard DB.
# KIMODO driver only supports constraint-style tasks (E2-E8, E10, E14-16).
# Missing from DB: E14 (A, B, C) — that's the only gap in constraint tasks.
# E1 (unconstrained T2M) and E13 (multi-prompt) are not KIMODO-compatible.
# E9 (motion repair) and E15 (prepend) are also not compatible (no adaptive mask / prepend mechanism).
#
# KIMODO saves NPZs by default and reads captions from the datalist;
# `--use-caption yes/no` toggles text conditioning.
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUT_ROOT=work_dirs/kimodo_missing_20260421
mkdir -p "$OUT_ROOT"

CUDA_VISIBLE_DEVICES=2 python3 tools/run_kimodo_all_tasks.py \
    --tasks E14 --max-samples 50 \
    --use-caption yes \
    --output-dir "$OUT_ROOT/kimodo_caption_e14" \
    > "$OUT_ROOT/kimodo_caption_e14.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 tools/run_kimodo_all_tasks.py \
    --tasks E14 --max-samples 50 \
    --use-caption no \
    --output-dir "$OUT_ROOT/kimodo_uncond_e14" \
    > "$OUT_ROOT/kimodo_uncond_e14.log" 2>&1 &

echo "Launched KIMODO E14 (caption + uncond)"
wait
echo "ALL DONE"
