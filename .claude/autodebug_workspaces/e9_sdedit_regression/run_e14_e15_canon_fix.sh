#!/bin/bash
# Re-run E14 + E15 with canonicalize_segment fix (Y preserved).
# 4 models × 2 tasks = 8 jobs on 8 GPUs.

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
OUT_ROOT=work_dirs/e14_e15_canon_fix_20260421
mkdir -p $OUT_ROOT

# E14 (4 models)
CUDA_VISIBLE_DEVICES=0 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E14 --max-samples 50 --save-npz \
    --output-dir $OUT_ROOT/ul_e14 > $OUT_ROOT/ul_e14.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_global --tasks E14 --max-samples 50 --save-npz \
    --output-dir $OUT_ROOT/ug_e14 > $OUT_ROOT/ug_e14.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E14 --max-samples 50 --save-npz \
    --use-rewritten --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cl_e14 > $OUT_ROOT/cl_e14.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E14 --max-samples 50 --save-npz \
    --use-rewritten --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cg_e14 > $OUT_ROOT/cg_e14.log 2>&1 &

# E15 (4 models)
CUDA_VISIBLE_DEVICES=4 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E15 --max-samples 50 --save-npz \
    --output-dir $OUT_ROOT/ul_e15 > $OUT_ROOT/ul_e15.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_global --tasks E15 --max-samples 50 --save-npz \
    --output-dir $OUT_ROOT/ug_e15 > $OUT_ROOT/ug_e15.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E15 --max-samples 50 --save-npz \
    --use-rewritten --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cl_e15 > $OUT_ROOT/cl_e15.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E15 --max-samples 50 --save-npz \
    --use-rewritten --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cg_e15 > $OUT_ROOT/cg_e15.log 2>&1 &

echo "Started 8 jobs: 4 models × (E14, E15) with canonicalize_segment Y-preserving fix"
wait
echo "ALL DONE"
