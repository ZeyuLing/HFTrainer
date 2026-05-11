#!/bin/bash
# Machine 1: run seeds 0,1,2 for E14-M
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:$PYTHONPATH
OUT=work_dirs/eval_e14_multiseed_20260511

for S in 0 1 2; do
    SEED=$((0xE4A10000 + S * 0x100000))
    SDIR=$OUT/s${S}
    echo "=== Seed $S (base=0x$(printf '%X' $SEED)) -> $SDIR ==="
    python3 scripts/eval/eval_m2m_v2_all_tasks.py \
        --models uncond_local \
        --tasks E14 \
        --settings M \
        --max-samples 100 \
        --save-npz \
        --replacement-guidance skip_last \
        --seed-base $SEED \
        --output-dir $SDIR \
        2>&1 | tail -5
    echo "--- seed $S done ---"
done
echo "MACHINE1_DONE"
