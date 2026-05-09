#!/bin/bash
# Re-run all caption_* model evals with text-guidance-scale=2.0 (reduced from default 5.0
# which was OOD due to cond_mask_prob=0.1 training).
# Uses REWRITTEN captions + saves NPZ for 3D viz.

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
OUT_ROOT=work_dirs/caption_scale2_20260421
mkdir -p $OUT_ROOT

# GPU 0-3: caption_local × 4 splits
CUDA_VISIBLE_DEVICES=0 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E1 E2 E3 --max-samples 50 \
    --use-rewritten --save-npz --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cl_0 > $OUT_ROOT/cl_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E4 E5 E6 --max-samples 50 \
    --use-rewritten --save-npz --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cl_1 > $OUT_ROOT/cl_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E7 E8 E10 --max-samples 50 \
    --use-rewritten --save-npz --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cl_2 > $OUT_ROOT/cl_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E13 E15 E16 --max-samples 50 \
    --use-rewritten --save-npz --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cl_3 > $OUT_ROOT/cl_3.log 2>&1 &

# GPU 4-7: caption_global × 4 splits
CUDA_VISIBLE_DEVICES=4 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E1 E2 E3 --max-samples 50 \
    --use-rewritten --save-npz --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cg_0 > $OUT_ROOT/cg_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E4 E5 E6 --max-samples 50 \
    --use-rewritten --save-npz --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cg_1 > $OUT_ROOT/cg_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E7 E8 E10 --max-samples 50 \
    --use-rewritten --save-npz --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cg_2 > $OUT_ROOT/cg_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E13 E15 E16 --max-samples 50 \
    --use-rewritten --save-npz --text-guidance-scale 2.0 \
    --output-dir $OUT_ROOT/cg_3 > $OUT_ROOT/cg_3.log 2>&1 &

echo "Started 8 jobs: caption × 12 tasks with CFG scale=2.0 + --save-npz + --use-rewritten"
wait
echo "ALL 8 JOBS DONE"
