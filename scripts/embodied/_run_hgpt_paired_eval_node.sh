#!/usr/bin/env bash
# Paired base-vs-optimized PhysFlow eval under the Humanoid-GPT judge, on a
# Taiji node. Greedy (single-sample, fixed seed) generation on the SAME prompts,
# scored by the SAME HGPT worker, so the only thing that differs is the
# generator weights (base KIMODO-G1 vs a RAFT checkpoint).
#
# Usage (on node):
#   CUDA_VISIBLE_DEVICES=1 bash scripts/embodied/_run_hgpt_paired_eval_node.sh \
#       [checkpoint_dir]
# Results append to work_dirs/physflow_overfit100_hgpt/physflow_eval_metrics.jsonl
set -eo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PATH=/usr/local/bin:$PATH
export HF_HOME="$PWD/checkpoints/kimodo"
export HUGGINGFACE_HUB_CACHE="$PWD/checkpoints/kimodo/hub"
export TRANSFORMERS_CACHE="$PWD/checkpoints/kimodo/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="$PWD/checkpoints/kimodo/text_encoders"
export PYTHONPATH="$PWD/ref_repo/KIMODO/kimodo:${PYTHONPATH:-}"
export PHYSFLOW_HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-/dev/shm/hgpt_venv311/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

CFG=configs/physflow/physflow_overfit100_hgpt.py
CKPT="${1:-work_dirs/physflow_overfit100_hgpt/checkpoint-iter_300}"
COMMON=(--config "$CFG"
        --eval-corpus configs/experiments/physflow_kimodo_g1/prompt_bank_humanml3d_overfit100.jsonl
        --split train --num-prompts 100 --min-frames 60 --max-frames 120 --seed 0)

echo "=== BASE (pre-RAFT KIMODO-G1) eval ==="
python3.10 scripts/embodied/physflow_periodic_eval.py "${COMMON[@]}" --base
echo "=== OPTIMIZED eval: $CKPT ==="
python3.10 scripts/embodied/physflow_periodic_eval.py "${COMMON[@]}" --ckpt "$CKPT"
echo "=== PAIRED EVAL DONE ==="
