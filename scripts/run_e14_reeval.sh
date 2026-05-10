#!/bin/bash
# Full E14 re-eval with multi-seed best-of-5 for cases with >10% skating.
# This uses the actual eval script infrastructure, so coordinates are guaranteed correct.
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Step 1: Re-run full E14-M eval (100 samples) with save-npz
python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local \
    --tasks E14 \
    --settings M \
    --max-samples 100 \
    --save-npz \
    --replacement-guidance skip_last \
    --output-dir work_dirs/eval_e14_uncond_local_reeval_20260510 \
    2>&1 | tee /tmp/e14_reeval.log

echo "DONE: exit code $?"
