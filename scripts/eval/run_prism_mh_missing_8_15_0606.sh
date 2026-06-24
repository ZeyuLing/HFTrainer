#!/bin/bash
# Resume missing MotionHub shards 8..15 for PRISM KT-spectral epoch-7.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:${PYTHONPATH:-}

COMMON_ENV=(
  CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py
  CKPT=work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_7
  MODE=depth_driven
  ANNO=data/annotation/test_motionhub_t2m.json
  DATA=data/motionhub
  OUT=outputs/evaluation/prism_kt_spectral_epoch7_rw/mh
  NSHARDS=32
  NUM_INFER=50
)

env "${COMMON_ENV[@]}" SHARD_START=8 NGPU=4 bash scripts/eval/run_gen_node.sh
env "${COMMON_ENV[@]}" SHARD_START=12 NGPU=4 bash scripts/eval/run_gen_node.sh

echo "DONE_MISSING_8_15 $(date)"
