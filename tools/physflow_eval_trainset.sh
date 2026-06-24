#!/usr/bin/env bash
# Deterministic TRAIN-SET paired eval for the PhysFlow overfit experiment.
# Runs two arms on the SAME 100 training prompts, scored by the frozen judge
# tracker + simulation-free kinematic metrics:
#   (1) base      : un-optimized KIMODO-G1   (control)
#   (2) optimized : a1 anchor=1.0 checkpoint (overfit result)
# Co-located inside an already-running tlinux4 vermo container (single GPU).
set -eo pipefail

CKPT="${1:?usage: physflow_eval_trainset.sh <ckpt_dir|base> <tag> [gpu_id]}"
TAG="${2:?need tag}"
GPU="${3:-6}"

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PATH=/usr/local/bin:$PATH
export HF_HOME="$PWD/checkpoints/kimodo"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="$PWD/checkpoints/kimodo/text_encoders"
export PHYSFLOW_CONVERT_PYTHON=python3
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export CUDA_VISIBLE_DEVICES="${GPU}"

OUT="work_dirs/physflow_overfit_eval/${TAG}"
MAN="work_dirs/physflow_overfit_eval/manifest_${TAG}"
mkdir -p "$OUT" "$MAN"

echo "[eval] arm=${TAG} ckpt=${CKPT} gpu=${GPU}"
python3 scripts/embodied/physflow_coevolve_viz.py \
  --config configs/physflow/physflow_overfit100.py \
  --ckpt "${CKPT}" \
  --eval-corpus configs/experiments/physflow_kimodo_g1/prompt_bank_humanml3d_overfit100.jsonl \
  --feature-dir data/kimodo_text_feature/kimodo_g1_llm2vec_overfit100 \
  --split train \
  --num-prompts 100 \
  --diffusion-steps 20 \
  --gen-batch 8 \
  --seed 0 \
  --out-dir "${OUT}" \
  --manifest-dir "${MAN}" \
  --iteration 0
echo "[eval] DONE arm=${TAG} -> ${OUT}"
