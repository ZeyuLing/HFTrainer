#!/bin/bash
# Run multi-seed on first half of bad PIDs on machine 1
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
python3 scripts/multiseed_e14_best_of_n.py --num-seeds 5 \
    --pids 0,1,2,3,4,6,7,8,9,10,12,14,15,16,18,20,21,24,25,26,27,28,30,32,33,34,35,36,37,38 \
    --output-dir work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz_multiseed \
    > /tmp/multiseed_e14_part1.log 2>&1
echo "DONE_PART1: exit code $?" >> /tmp/multiseed_e14_part1.log
