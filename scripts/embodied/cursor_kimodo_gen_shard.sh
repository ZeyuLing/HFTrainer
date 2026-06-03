#!/usr/bin/env bash
# Generate-only KIMODO-G1 worker for one prompt-bank shard on one GPU.
# Shares OUT with other shards; already-generated CSVs are skipped, distinct
# prompt ids => no write collisions. Convert is done once after all shards.
set +e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

PY="${PHYSFLOW_KIMODO_PY:-python3}"        # KIMODO needs py3.10+
PROMPT_BANK="${PHYSFLOW_PROMPT_BANK:?set PHYSFLOW_PROMPT_BANK}"
OUT="${PHYSFLOW_OUT:-output/physflow_kimodo_g1/overfit100_pool}"
GPU="${PHYSFLOW_GPU:?set PHYSFLOW_GPU}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "[shard gpu=$GPU bank=$PROMPT_BANK] $(date) start"
"$PY" scripts/embodied/physflow_kimodo_g1_runner.py --mode generate \
  --output-dir "$OUT" --prompt-bank "$PROMPT_BANK" --prompt-split train \
  --max-prompts 100 --samples-per-prompt 1 \
  --kimodo-model Kimodo-G1-RP-v1 --diffusion-steps 100 \
  --seed 42 --cfg-type separated --cfg-weight 2.0 2.0 \
  --local-cache --require-ready --robot-json-subsample 1
echo "[shard gpu=$GPU] generate exit=$? $(date)"
echo "SHARD_DONE gpu=$GPU"
