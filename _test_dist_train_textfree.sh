#!/bin/bash
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:$PYTHONPATH

LOG=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/_debug_dist_train.log

# 8-GPU distributed training test with Medium config
accelerate launch --num_processes 8 --mixed_precision no tools/train.py \
    configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_textfree_m.py \
    --work-dir work_dirs/_debug_textfree_m_dist_test \
    --cfg-options train_cfg.max_epochs=1 \
    default_hooks.logger.iter_interval=5 \
    > $LOG 2>&1 &

PID=$!
echo "Training PID: $PID"

# Wait up to 5 minutes
for i in $(seq 1 10); do
    sleep 30
    echo "=== Check $i ($(date)) ==="
    tail -3 $LOG
    if grep -q "step \[50/" $LOG 2>/dev/null; then
        echo "Training running! Showing loss trend:"
        grep "loss=" $LOG | head -15
        kill $PID 2>/dev/null
        echo "DIST_TRAINING_OK"
        exit 0
    fi
    if ! kill -0 $PID 2>/dev/null; then
        echo "Process died"
        tail -40 $LOG
        exit 1
    fi
done

echo "Timeout"
kill $PID 2>/dev/null
tail -50 $LOG
