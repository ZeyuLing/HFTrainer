#!/bin/bash
# Launch the BABEL checkpoint comparison on the already-running Taiji debug node.
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:$PWD/ref_repo/MotionStreamer/MotionStreamer:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

export PHASE=gen
export OUT_ROOT=outputs/evaluation/babel_seq/ckpt_compare_20260615_m2
export METHODS="kt_latest iter15000"
export NUM_GPUS=8
export MACHINE_NUM=1
mkdir -p "$OUT_ROOT/_logs"
nohup bash scripts/eval/run_prism_babel_checkpoint_compare.sh \
  > "$OUT_ROOT/_logs/driver_machine2.log" 2>&1 &
echo $! > "$OUT_ROOT/_logs/driver_machine2.pid"
echo "started pid=$(cat "$OUT_ROOT/_logs/driver_machine2.pid") out=$OUT_ROOT"
