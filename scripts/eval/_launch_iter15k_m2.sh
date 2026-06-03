#!/bin/bash
# Remote launcher for iter_15000 VAE-fixed reproduction on a single 8-GPU node.
# Runs inside a tmux session so it survives the taiji exec session closing.
# Generates H3D then MH (rewritten captions, mode=none) using the *_iter15k config
# (wanmo_vae2d_aug VAE). Resumable via --skip-existing.
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT=$REPO/outputs/evaluation/prism_paper_iter15000_vaefix
SESS=iter15k_gen
mkdir -p "$ROOT/_logs"
tmux kill-session -t "$SESS" 2>/dev/null
tmux new-session -d -s "$SESS" \
  "cd $REPO && export PYTHONPATH=$REPO && NSHARDS=8 SHARD_START=0 NGPU=8 MODE=none bash scripts/eval/run_iter15k_node.sh > $ROOT/_logs/node_main.log 2>&1"
sleep 2
echo "--- tmux sessions ---"
tmux ls 2>&1
