#!/usr/bin/env bash
# One-off self-test of the periodic-eval script on the existing smoke checkpoint.
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export HF_HOME=checkpoints/kimodo
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${OFT_EVAL_GPU:-1}
echo "[evaltest] $(date) GPU=$CUDA_VISIBLE_DEVICES host=$(hostname)"
python3 scripts/embodied/physflow_periodic_eval.py \
  --config configs/physflow/physflow_online_adv_smoke.py \
  --eval-corpus configs/experiments/physflow_kimodo_g1/physflow_text_eval.jsonl \
  --num-prompts 2 --gen-batch 2 \
  --ckpt work_dirs/physflow_online_adv_smoke/checkpoint-iter_2
echo "[evaltest] done code=$? $(date)"
