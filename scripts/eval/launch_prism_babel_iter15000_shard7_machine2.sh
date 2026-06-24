#!/bin/bash
# Restart the missing iter15000 BABEL shard on GPU7 of lzy_debug_machine_2.
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:$PWD/ref_repo/MotionStreamer/MotionStreamer:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

OUT=outputs/evaluation/babel_seq/ckpt_compare_20260615_m2/iter15000_gen
mkdir -p "$OUT/logs"
nohup bash -lc "CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval/gen_prism_babel_seq.py \
  --config configs/prism/prism_1b_tp2m_multiframe_iter15k.py \
  --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
  --manifest outputs/evaluation/babel_seq/common_valid_manifest.jsonl \
  --output-dir $OUT \
  --num-inference-steps 50 \
  --guidance-scale 5.0 \
  --ar-cond-frames 5 \
  --kafs-mode none \
  --rewrite-captions \
  --num-shards 8 \
  --shard-idx 7 \
  --skip-existing" \
  > "$OUT/logs/gen_h0_g7_restart.log" 2>&1 &
echo $! > "$OUT/logs/gen_h0_g7_restart.pid"
echo "started shard7 pid=$(cat "$OUT/logs/gen_h0_g7_restart.pid")"
