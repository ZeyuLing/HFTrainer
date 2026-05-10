#!/bin/bash
# Run multi-seed on second half of bad PIDs on machine 2
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
python3 scripts/multiseed_e14_best_of_n.py --num-seeds 5 \
    --pids 40,44,45,46,47,48,49,50,52,53,54,55,57,58,60,62,63,65,66,67,68,69,70,71,72,73,74,75,77,79,80,82,83,84,85,86,87,88,90,92,95,96,97,98,99 \
    --output-dir work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz_multiseed \
    > /tmp/multiseed_e14_part2.log 2>&1
echo "DONE_PART2: exit code $?" >> /tmp/multiseed_e14_part2.log
