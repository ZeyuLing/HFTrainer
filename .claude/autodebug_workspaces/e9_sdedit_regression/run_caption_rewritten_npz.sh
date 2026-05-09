#!/bin/bash
# Re-run all caption_* model evals using REWRITTEN captions.
# uncond_* results are unaffected by --use-rewritten (they don't use caption).
# 12 caption-aware tasks: E1 E2 E3 E4 E5 E6 E7 E8 E10 E13 E15 E16
# 2 caption models × 12 tasks = 24 jobs, distributed across 8 GPUs.
# Each GPU handles 3 (model, task) pairs sequentially.
#
# IMPORTANT: --save-npz is required so dashboard 3D viewer can render.
# See motion_annot_web/eval_dashboard/CLAUDE.md.

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
OUT_ROOT=work_dirs/caption_rewritten_npz_20260421
mkdir -p $OUT_ROOT

# GPU 0: caption_local {E1,E2,E3}
CUDA_VISIBLE_DEVICES=0 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E1 E2 E3 --max-samples 50 \
    --use-rewritten --save-npz --output-dir $OUT_ROOT/cl_0 \
    > $OUT_ROOT/cl_0.log 2>&1 &

# GPU 1: caption_local {E4,E5,E6}
CUDA_VISIBLE_DEVICES=1 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E4 E5 E6 --max-samples 50 \
    --use-rewritten --save-npz --output-dir $OUT_ROOT/cl_1 \
    > $OUT_ROOT/cl_1.log 2>&1 &

# GPU 2: caption_local {E7,E8,E10}
CUDA_VISIBLE_DEVICES=2 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E7 E8 E10 --max-samples 50 \
    --use-rewritten --save-npz --output-dir $OUT_ROOT/cl_2 \
    > $OUT_ROOT/cl_2.log 2>&1 &

# GPU 3: caption_local {E13,E15,E16}
CUDA_VISIBLE_DEVICES=3 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E13 E15 E16 --max-samples 50 \
    --use-rewritten --save-npz --output-dir $OUT_ROOT/cl_3 \
    > $OUT_ROOT/cl_3.log 2>&1 &

# GPU 4: caption_global {E1,E2,E3}
CUDA_VISIBLE_DEVICES=4 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E1 E2 E3 --max-samples 50 \
    --use-rewritten --save-npz --output-dir $OUT_ROOT/cg_0 \
    > $OUT_ROOT/cg_0.log 2>&1 &

# GPU 5: caption_global {E4,E5,E6}
CUDA_VISIBLE_DEVICES=5 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E4 E5 E6 --max-samples 50 \
    --use-rewritten --save-npz --output-dir $OUT_ROOT/cg_1 \
    > $OUT_ROOT/cg_1.log 2>&1 &

# GPU 6: caption_global {E7,E8,E10}
CUDA_VISIBLE_DEVICES=6 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E7 E8 E10 --max-samples 50 \
    --use-rewritten --save-npz --output-dir $OUT_ROOT/cg_2 \
    > $OUT_ROOT/cg_2.log 2>&1 &

# GPU 7: caption_global {E13,E15,E16}
CUDA_VISIBLE_DEVICES=7 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E13 E15 E16 --max-samples 50 \
    --use-rewritten --save-npz --output-dir $OUT_ROOT/cg_3 \
    > $OUT_ROOT/cg_3.log 2>&1 &

echo "Started 8 jobs for caption models with REWRITTEN captions + --save-npz."
echo "Tasks: E1 E2 E3 E4 E5 E6 E7 E8 E10 E13 E15 E16"
wait
echo "ALL 8 JOBS DONE"
