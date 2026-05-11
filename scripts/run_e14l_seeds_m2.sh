#!/bin/bash
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:$PYTHONPATH
OUT=work_dirs/eval_e14L_multiseed_20260511

for S in 3 4; do
    SEED=$((0xE4A10000 + S * 0x100000))
    SDIR=$OUT/s${S}
    echo "=== Seed $S -> $SDIR ==="
    python3 scripts/eval/eval_m2m_v2_all_tasks.py \
        --models uncond_local \
        --tasks E14 \
        --settings L \
        --max-samples 100 \
        --save-npz \
        --replacement-guidance skip_last \
        --seed-base $SEED \
        --output-dir $SDIR \
        2>&1 | tail -3
    echo "--- seed $S done ---"
done
echo "M2_DONE"
