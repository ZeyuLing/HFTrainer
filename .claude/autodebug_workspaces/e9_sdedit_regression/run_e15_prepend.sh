#!/bin/bash
# E15 new semantics (2026-04-21): prepend transition from start pose P.
# 4 models × E15 (3 settings A/B/C) = 4 jobs.

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
OUT_ROOT=work_dirs/e15_prepend_20260421
mkdir -p $OUT_ROOT

CUDA_VISIBLE_DEVICES=0 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E15 --max-samples 50 --save-npz \
    --output-dir $OUT_ROOT/ul > $OUT_ROOT/ul.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_global --tasks E15 --max-samples 50 --save-npz \
    --output-dir $OUT_ROOT/ug > $OUT_ROOT/ug.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E15 --max-samples 50 --save-npz \
    --use-rewritten --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cl > $OUT_ROOT/cl.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E15 --max-samples 50 --save-npz \
    --use-rewritten --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cg > $OUT_ROOT/cg.log 2>&1 &

echo "Started 4 jobs: 4 models × E15 (prepend to start pose)"
wait
echo "ALL DONE"
