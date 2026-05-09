#!/bin/bash
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:$PYTHONPATH

LOG=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/_debug_train.log

# Quick 1-GPU training test
python3 tools/train.py configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_textfree_s.py \
    --work-dir work_dirs/_debug_textfree_s_test \
    --cfg-options train_cfg.max_epochs=1 \
    train_dataloader.batch_size=8 \
    train_dataloader.num_workers=2 \
    default_hooks.logger.iter_interval=5 \
    > $LOG 2>&1 &

PID=$!
echo "Training PID: $PID"

# Wait up to 5 minutes, check log every 30 seconds
for i in $(seq 1 10); do
    sleep 30
    echo "=== Check $i ($(date)) ==="
    tail -5 $LOG
    # Check if training iterations are appearing
    if grep -q "iter.*loss" $LOG 2>/dev/null; then
        echo "Training iterations detected!"
        # Show some loss values
        grep "iter.*loss" $LOG | head -20
        kill $PID 2>/dev/null
        echo "TRAINING_OK"
        exit 0
    fi
    # Check if process died
    if ! kill -0 $PID 2>/dev/null; then
        echo "Process died"
        tail -30 $LOG
        exit 1
    fi
done

echo "Timeout waiting for training to start"
kill $PID 2>/dev/null
tail -50 $LOG
