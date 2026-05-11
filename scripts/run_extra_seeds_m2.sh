#!/bin/bash
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:$PYTHONPATH
OUT=work_dirs/eval_e14_multiseed_20260511

# Extra seeds 8,9 for E14-M
for S in 8 9; do
    SEED=$((0xE4A10000 + S * 0x100000))
    echo "=== Seed $S ==="
    python3 scripts/eval/eval_m2m_v2_all_tasks.py \
        --models uncond_local --tasks E14 --settings M \
        --max-samples 100 --save-npz --replacement-guidance skip_last \
        --seed-base $SEED --output-dir $OUT/s${S} 2>&1 | tail -3
done
echo "EXTRA_M2_DONE"
