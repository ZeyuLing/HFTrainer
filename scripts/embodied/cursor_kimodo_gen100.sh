#!/usr/bin/env bash
# Generate ~100 KIMODO-G1 motions from the HumanML3D prompt bank, then convert
# to ProtoMotions .motion. Detached; logs to a shared-FS file we can poll.
set +e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

PY="${PHYSFLOW_KIMODO_PY:-python3}"   # KIMODO needs py3.10+, NOT the isaacgym py38 env
PROMPT_BANK=configs/experiments/physflow_kimodo_g1/prompt_bank_humanml3d_overfit100.jsonl
OUT="${PHYSFLOW_OUT:-output/physflow_kimodo_g1/overfit100_pool}"
GPU="${PHYSFLOW_GPU:-2}"
MAXP="${PHYSFLOW_MAX_PROMPTS:-100}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"

COMMON=( --output-dir "$OUT" --prompt-bank "$PROMPT_BANK" --prompt-split train
         --max-prompts "$MAXP" --samples-per-prompt 1
         --kimodo-model Kimodo-G1-RP-v1 --diffusion-steps 100
         --seed 42 --cfg-type separated --cfg-weight 2.0 2.0
         --local-cache --require-ready --robot-json-subsample 1 )

echo "[gen] $(date) start on GPU $GPU"
"$PY" scripts/embodied/physflow_kimodo_g1_runner.py --mode generate "${COMMON[@]}"
echo "[gen] generate exit=$? $(date)"

echo "[gen] converting CSV -> .motion"
"$PY" scripts/embodied/physflow_kimodo_g1_runner.py --mode convert "${COMMON[@]}"
echo "[gen] convert exit=$? $(date)"

echo "[gen] proto motion count: $(ls "$OUT"/proto/*.motion 2>/dev/null | wc -l)"
echo "ALL_GEN_DONE $(date)"
