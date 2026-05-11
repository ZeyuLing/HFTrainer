#!/bin/bash
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:$PYTHONPATH
OUT=work_dirs/eval_e8d_multiseed_20260511
for S in 0 1 2; do
    SEED=$((0xE4A10000 + S * 0x100000))
    echo "=== Seed $S ==="
    python3 scripts/eval/eval_m2m_v2_all_tasks.py \
        --models uncond_local --tasks E8 --settings D \
        --max-samples 100 --save-npz --replacement-guidance skip_last \
        --seed-base $SEED --output-dir $OUT/s${S} 2>&1 | tail -3
done
echo "M1_DONE"
