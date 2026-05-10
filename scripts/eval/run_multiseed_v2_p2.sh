#!/bin/bash
# Multi-seed on machine 2: PIDs 50-99
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
python3 scripts/multiseed_e14_v2.py \
    --npz-dir work_dirs/eval_e14_uncond_local_reeval_20260510/uncond_local/E14_M/npz \
    --output-dir work_dirs/eval_e14_uncond_local_reeval_20260510/uncond_local/E14_M/npz_best \
    --num-seeds 5 \
    --threshold 0.10 \
    --pids 50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99 \
    > /tmp/multiseed_v2_part2.log 2>&1
echo "DONE_P2: exit code $?" >> /tmp/multiseed_v2_part2.log
