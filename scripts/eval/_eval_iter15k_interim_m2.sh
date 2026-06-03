#!/bin/bash
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT=$REPO/outputs/evaluation/prism_paper_iter15000_vaefix
SESS=iter15k_eval
mkdir -p "$ROOT/_eval_logs"
tmux kill-session -t "$SESS" 2>/dev/null
tmux new-session -d -s "$SESS" \
  "cd $REPO && export PYTHONPATH=$REPO && CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/compute_kafs_metrics.py --kafs-dir $ROOT/h3d --modes none --anno-file data/annotation/test_hml3d.json --rewritten-caption-file data/annotation/test_hml3d_rewritten.json --data-dir data/motionhub --chunk-size 64 --n-repeats 20 --gpu 0 > $ROOT/_eval_logs/h3d_interim.log 2>&1"
sleep 2
tmux ls 2>&1 | grep iter15k
