#!/bin/bash
# Launch all 4 KAFS modes as background nohup processes, each on its own GPU.
# This script is meant to be run INSIDE the taiji container.
# It avoids using wait/& in complex ways — just spawns nohup jobs.

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
mkdir -p work_dirs/prism_kafs_ablation

nohup env CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/launch_kafs_single.py none 200 \
    > work_dirs/prism_kafs_ablation/log_none.txt 2>&1 &
P1=$!

nohup env CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/launch_kafs_single.py depth_driven 200 \
    > work_dirs/prism_kafs_ablation/log_depth_driven.txt 2>&1 &
P2=$!

nohup env CUDA_VISIBLE_DEVICES=2 python3 scripts/eval/launch_kafs_single.py uniform 200 \
    > work_dirs/prism_kafs_ablation/log_uniform.txt 2>&1 &
P3=$!

nohup env CUDA_VISIBLE_DEVICES=3 python3 scripts/eval/launch_kafs_single.py random 200 \
    > work_dirs/prism_kafs_ablation/log_random.txt 2>&1 &
P4=$!

echo "Launched 4 KAFS jobs: PIDs $P1 $P2 $P3 $P4"
echo "$P1 $P2 $P3 $P4" > work_dirs/prism_kafs_ablation/pids.txt
