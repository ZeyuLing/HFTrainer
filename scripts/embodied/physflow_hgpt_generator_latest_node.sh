#!/usr/bin/env bash
# HumanoidGPT frozen-judge generator run with qpos replay export.
set -euo pipefail

REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$REPO"

GENCKPT="${GENCKPT:-work_dirs/physflow_verify_hymotion_g1_base/checkpoint-iter_130000}"
WORK_DIR="${WORK_DIR:-work_dirs/physflow_coevolve_hgpt_hymotion132k/generator_half}"
POOL="${POOL:-work_dirs/physflow_coevolve_hgpt_hymotion132k/qpos_pool}"
MAX_ITERS="${MAX_ITERS:-300}"

echo "[hgpt-generator] $(date) host=$(hostname)"
echo "[hgpt-generator] genckpt=$GENCKPT work_dir=$WORK_DIR pool=$POOL max_iters=$MAX_ITERS"
nvidia-smi || true
if [[ ! -e "$GENCKPT" ]]; then
  echo "[hgpt-generator] FATAL: GENCKPT does not exist: $GENCKPT" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-checkpoints/kimodo}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 tools/train.py configs/physflow/physflow_coevo_humanoidgpt_g1.py \
  --work-dir "$WORK_DIR" \
  --load-from "$GENCKPT" --load-scope model \
  --cfg-options \
  "train_cfg.max_iters=$MAX_ITERS" \
  "trainer.tracker_qpos_pool_dir=$POOL" \
  "default_hooks.checkpoint.interval=100" \
  "default_hooks.checkpoint.max_keep_ckpts=4"

echo "[hgpt-generator] done $(date)"
