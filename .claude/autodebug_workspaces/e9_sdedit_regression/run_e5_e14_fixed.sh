#!/bin/bash
# Re-run E5 (new XZ-only mask) + E14 (new non-stitching backend) for all 4 models.
# 4 models × 2 tasks = 8 jobs, 1 per GPU.

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
OUT_ROOT=work_dirs/e5_e14_fixed_20260421
mkdir -p $OUT_ROOT

# GPU 0-3: uncond_{local,global} + caption_{local,global} × E5
CUDA_VISIBLE_DEVICES=0 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E5 --max-samples 50 --save-npz \
    --output-dir $OUT_ROOT/ul_e5 > $OUT_ROOT/ul_e5.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_global --tasks E5 --max-samples 50 --save-npz \
    --output-dir $OUT_ROOT/ug_e5 > $OUT_ROOT/ug_e5.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E5 --max-samples 50 --save-npz \
    --use-rewritten --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cl_e5 > $OUT_ROOT/cl_e5.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E5 --max-samples 50 --save-npz \
    --use-rewritten --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cg_e5 > $OUT_ROOT/cg_e5.log 2>&1 &

# GPU 4-7: 4 models × E14
CUDA_VISIBLE_DEVICES=4 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E14 --max-samples 50 --save-npz \
    --output-dir $OUT_ROOT/ul_e14 > $OUT_ROOT/ul_e14.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_global --tasks E14 --max-samples 50 --save-npz \
    --output-dir $OUT_ROOT/ug_e14 > $OUT_ROOT/ug_e14.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E14 --max-samples 50 --save-npz \
    --use-rewritten --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cl_e14 > $OUT_ROOT/cl_e14.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E14 --max-samples 50 --save-npz \
    --use-rewritten --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cg_e14 > $OUT_ROOT/cg_e14.log 2>&1 &

echo "Started 8 jobs: 4 models × (E5 XZ mask, E14 no-stitch)"
wait
echo "ALL DONE"
