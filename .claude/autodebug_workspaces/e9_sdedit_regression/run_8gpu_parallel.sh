#!/bin/bash
# 8-GPU parallel eval: 4 models × 7 tasks = 28 jobs, distributed to 8 GPUs.
# Strategy: group by model to reuse the loaded checkpoint.
# Each GPU gets a contiguous slice so checkpoint is loaded once per slice.

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

OUT_ROOT=work_dirs/full_rerun_20260421
mkdir -p $OUT_ROOT

# (gpu, model, tasks-list) assignments:
# uncond_local has 7 tasks → split into 2 GPUs (4 + 3)
# uncond_global has 7 tasks → split into 2 GPUs (4 + 3)
# caption_local has 7 tasks → split into 2 GPUs (4 + 3)
# caption_global has 7 tasks → split into 2 GPUs (4 + 3)
# Total 8 GPUs used.

# GPU 0: uncond_local on E2 E3 E5 E9
CUDA_VISIBLE_DEVICES=0 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E2 E3 E5 E9 --max-samples 50 \
    --output-dir $OUT_ROOT/uncond_local_0 --save-npz \
    > $OUT_ROOT/uncond_local_0.log 2>&1 &
PID0=$!

# GPU 1: uncond_local on E14 E15 E16
CUDA_VISIBLE_DEVICES=1 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local --tasks E14 E15 E16 --max-samples 50 \
    --output-dir $OUT_ROOT/uncond_local_1 --save-npz \
    > $OUT_ROOT/uncond_local_1.log 2>&1 &
PID1=$!

# GPU 2: uncond_global on E2 E3 E5 E9
CUDA_VISIBLE_DEVICES=2 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_global --tasks E2 E3 E5 E9 --max-samples 50 \
    --output-dir $OUT_ROOT/uncond_global_0 --save-npz \
    > $OUT_ROOT/uncond_global_0.log 2>&1 &
PID2=$!

# GPU 3: uncond_global on E14 E15 E16
CUDA_VISIBLE_DEVICES=3 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_global --tasks E14 E15 E16 --max-samples 50 \
    --output-dir $OUT_ROOT/uncond_global_1 --save-npz \
    > $OUT_ROOT/uncond_global_1.log 2>&1 &
PID3=$!

# GPU 4: caption_local on E2 E3 E5 E9
CUDA_VISIBLE_DEVICES=4 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E2 E3 E5 E9 --max-samples 50 \
    --output-dir $OUT_ROOT/caption_local_0 --save-npz \
    > $OUT_ROOT/caption_local_0.log 2>&1 &
PID4=$!

# GPU 5: caption_local on E14 E15 E16
CUDA_VISIBLE_DEVICES=5 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_local --tasks E14 E15 E16 --max-samples 50 \
    --output-dir $OUT_ROOT/caption_local_1 --save-npz \
    > $OUT_ROOT/caption_local_1.log 2>&1 &
PID5=$!

# GPU 6: caption_global on E2 E3 E5 E9
CUDA_VISIBLE_DEVICES=6 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E2 E3 E5 E9 --max-samples 50 \
    --output-dir $OUT_ROOT/caption_global_0 --save-npz \
    > $OUT_ROOT/caption_global_0.log 2>&1 &
PID6=$!

# GPU 7: caption_global on E14 E15 E16
CUDA_VISIBLE_DEVICES=7 python3 tools/eval_m2m_v2_all_tasks.py \
    --models caption_global --tasks E14 E15 E16 --max-samples 50 \
    --output-dir $OUT_ROOT/caption_global_1 --save-npz \
    > $OUT_ROOT/caption_global_1.log 2>&1 &
PID7=$!

echo "Started 8 parallel jobs:"
echo "  GPU 0: uncond_local  {E2,E3,E5,E9}    PID=$PID0"
echo "  GPU 1: uncond_local  {E14,E15,E16}    PID=$PID1"
echo "  GPU 2: uncond_global {E2,E3,E5,E9}    PID=$PID2"
echo "  GPU 3: uncond_global {E14,E15,E16}    PID=$PID3"
echo "  GPU 4: caption_local {E2,E3,E5,E9}    PID=$PID4"
echo "  GPU 5: caption_local {E14,E15,E16}    PID=$PID5"
echo "  GPU 6: caption_global {E2,E3,E5,E9}   PID=$PID6"
echo "  GPU 7: caption_global {E14,E15,E16}   PID=$PID7"

# Wait for all
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7
echo "ALL 8 JOBS DONE"
