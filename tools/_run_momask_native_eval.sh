#!/bin/bash
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

LOG=work_dirs/momask_eval/logs/momask_native_eval.log
mkdir -p work_dirs/momask_eval/logs
echo "=== GT-only eval ===" > "$LOG"
python3 tools/eval_momask_native_h3d263.py \
    --recon_root work_dirs/momask_eval/h3d263_test_recon \
    --src_h3d272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
    --momask_root ref_repo/Momask/momask-codes \
    --mode gt-only \
    --num_repeats 5 \
    --output work_dirs/momask_eval/momask_native_gt_only.json \
    >> "$LOG" 2>&1

echo "=== Pred eval ===" >> "$LOG"
python3 tools/eval_momask_native_h3d263.py \
    --recon_root work_dirs/momask_eval/h3d263_test_recon \
    --src_h3d272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
    --momask_root ref_repo/Momask/momask-codes \
    --mode pred \
    --pred_dir work_dirs/momask_eval/momask_pred_263_v2 \
    --num_repeats 20 \
    --output work_dirs/momask_eval/momask_native_pred.json \
    >> "$LOG" 2>&1
echo "=== END ===" >> "$LOG"
