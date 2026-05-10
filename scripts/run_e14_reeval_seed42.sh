#!/bin/bash
# E14 re-eval with seed offset to get a different sample set
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Run with --seed-offset so we get different random results
python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local \
    --tasks E14 \
    --settings M \
    --max-samples 100 \
    --save-npz \
    --replacement-guidance skip_last \
    --seed 42 \
    --output-dir work_dirs/eval_e14_uncond_local_reeval_seed42_20260510 \
    2>&1 | tee /tmp/e14_reeval_seed42.log

echo "DONE: exit code $?"
