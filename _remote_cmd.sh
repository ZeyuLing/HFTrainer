#!/bin/bash
set -e
BASE=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/cjgame_repair_eval
mkdir -p "$BASE/adaptive_masks"
mkdir -p "$BASE/logs"
echo "SETUP OK $(date)" > "$BASE/logs/setup_done.txt"
ls "$BASE/"
