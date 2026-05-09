#!/bin/bash
# Rerun E8 + E14 for uncond_local/global with --save-npz (250 samples were
# without gen_motion_path in DB because previous E8/E14 reruns didn't save NPZ).
# 2 models × 2 tasks (E8, E14) = 4 jobs on 4 GPUs.

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
OUT_ROOT=work_dirs/uncond_e8_e14_npz_20260421
mkdir -p $OUT_ROOT

# GPU 0: uncond_local E8
CUDA_VISIBLE_DEVICES=0 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E8 --max-samples 50 \
    --save-npz --output-dir $OUT_ROOT/ul_e8 \
    > $OUT_ROOT/ul_e8.log 2>&1 &

# GPU 1: uncond_local E14
CUDA_VISIBLE_DEVICES=1 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E14 --max-samples 50 \
    --save-npz --output-dir $OUT_ROOT/ul_e14 \
    > $OUT_ROOT/ul_e14.log 2>&1 &

# GPU 2: uncond_global E8
CUDA_VISIBLE_DEVICES=2 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_global --tasks E8 --max-samples 50 \
    --save-npz --output-dir $OUT_ROOT/ug_e8 \
    > $OUT_ROOT/ug_e8.log 2>&1 &

# GPU 3: uncond_global E14
CUDA_VISIBLE_DEVICES=3 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_global --tasks E14 --max-samples 50 \
    --save-npz --output-dir $OUT_ROOT/ug_e14 \
    > $OUT_ROOT/ug_e14.log 2>&1 &

echo "Started 4 jobs: uncond_local/global × E8/E14 with --save-npz."
wait
echo "ALL 4 JOBS DONE"
