#!/bin/bash
# Rerun missing tasks: E1, E4, E6, E7, E8, E10, E13 × 4 models.
# These tasks never had uncond/caption runs in dashboard. 8-GPU parallel.

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
OUT_ROOT=work_dirs/missing_tasks_rerun_20260421
mkdir -p $OUT_ROOT

# Split tasks into 2 groups of 3-4 per GPU slot.
# Each model → 2 GPU slots (group A: E1 E4 E6; group B: E7 E8 E10 E13)
# 4 models × 2 groups = 8 GPUs

# GPU 0: uncond_local on {E1,E4,E6}
CUDA_VISIBLE_DEVICES=0 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E1 E4 E6 --max-samples 50 \
    --output-dir $OUT_ROOT/uncond_local_A \
    > $OUT_ROOT/uncond_local_A.log 2>&1 &

# GPU 1: uncond_local on {E7,E8,E10,E13}
CUDA_VISIBLE_DEVICES=1 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E7 E8 E10 E13 --max-samples 50 \
    --output-dir $OUT_ROOT/uncond_local_B \
    > $OUT_ROOT/uncond_local_B.log 2>&1 &

# GPU 2: uncond_global on {E1,E4,E6}
CUDA_VISIBLE_DEVICES=2 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_global --tasks E1 E4 E6 --max-samples 50 \
    --output-dir $OUT_ROOT/uncond_global_A \
    > $OUT_ROOT/uncond_global_A.log 2>&1 &

# GPU 3: uncond_global on {E7,E8,E10,E13}
CUDA_VISIBLE_DEVICES=3 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_global --tasks E7 E8 E10 E13 --max-samples 50 \
    --output-dir $OUT_ROOT/uncond_global_B \
    > $OUT_ROOT/uncond_global_B.log 2>&1 &

# GPU 4: caption_local on {E1,E4,E6}
CUDA_VISIBLE_DEVICES=4 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E1 E4 E6 --max-samples 50 \
    --output-dir $OUT_ROOT/caption_local_A \
    > $OUT_ROOT/caption_local_A.log 2>&1 &

# GPU 5: caption_local on {E7,E8,E10,E13}
CUDA_VISIBLE_DEVICES=5 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E7 E8 E10 E13 --max-samples 50 \
    --output-dir $OUT_ROOT/caption_local_B \
    > $OUT_ROOT/caption_local_B.log 2>&1 &

# GPU 6: caption_global on {E1,E4,E6}
CUDA_VISIBLE_DEVICES=6 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E1 E4 E6 --max-samples 50 \
    --output-dir $OUT_ROOT/caption_global_A \
    > $OUT_ROOT/caption_global_A.log 2>&1 &

# GPU 7: caption_global on {E7,E8,E10,E13}
CUDA_VISIBLE_DEVICES=7 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E7 E8 E10 E13 --max-samples 50 \
    --output-dir $OUT_ROOT/caption_global_B \
    > $OUT_ROOT/caption_global_B.log 2>&1 &

echo "Started 8 parallel jobs for missing tasks (E1,E4,E6,E7,E8,E10,E13)"
wait
echo "ALL DONE"
