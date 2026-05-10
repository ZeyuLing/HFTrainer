#!/bin/bash
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
python3 scripts/multiseed_e14_best_of_n.py --num-seeds 5 \
    --pids 0,20,33,34,38,40,45,48,53,55,73,80,86,88,97,14 \
    --output-dir work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz_multiseed \
    > /tmp/multiseed_e14.log 2>&1
echo "DONE: exit code $?" >> /tmp/multiseed_e14.log
