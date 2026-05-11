#!/bin/bash
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:$PYTHONPATH
OUT=work_dirs/eval_e14_ep2490_20260511

# 让 eval 脚本加载 epoch_2490 而非最新 checkpoint
mkdir -p work_dirs/uncond_local_ep2490_tmp
ln -snf $(pwd)/work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_2490 work_dirs/uncond_local_ep2490_tmp/checkpoint-epoch_2490
export _EVAL_WORK_DIR__UNCOND_LOCAL=work_dirs/uncond_local_ep2490_tmp

for S in 0 1 2 3 4; do
    SEED=$((0xE4A10000 + S * 0x100000))
    echo "=== Seed $S (ep2490) ==="
    python3 scripts/eval/eval_m2m_v2_all_tasks.py \
        --models uncond_local --tasks E14 --settings M L \
        --max-samples 100 --save-npz --replacement-guidance skip_last \
        --seed-base $SEED --output-dir $OUT/s${S} 2>&1 | tail -3
done
echo "EP2490_M1_DONE"
