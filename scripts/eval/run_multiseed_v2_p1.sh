#!/bin/bash
# Multi-seed on machine 1: first 50 PIDs
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
python3 scripts/multiseed_e14_v2.py \
    --npz-dir work_dirs/eval_e14_uncond_local_reeval_20260510/uncond_local/E14_M/npz \
    --output-dir work_dirs/eval_e14_uncond_local_reeval_20260510/uncond_local/E14_M/npz_best \
    --num-seeds 5 \
    --threshold 0.10 \
    --pids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49 \
    > /tmp/multiseed_v2_part1.log 2>&1
echo "DONE_P1: exit code $?" >> /tmp/multiseed_v2_part1.log
