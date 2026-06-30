#!/usr/bin/env bash
# Table 2 generator + Humanoid-GPT judge launcher.
set -euo pipefail

REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "$REPO"

CONFIG="${CONFIG:-configs/physflow/table2_g1_generator_humanoidgpt.py}"
WORK_DIR="${WORK_DIR:-work_dirs/table2_g1_generator_humanoidgpt}"
GENCKPT="${GENCKPT:-work_dirs/hymotion_g1_t2m_38dim/checkpoint-iter_339000}"
MAX_ITERS="${MAX_ITERS:-3000}"
PHYSFLOW_HGPT_VENV="${PHYSFLOW_HGPT_VENV:-/dev/shm/hgpt_venv311}"
TRAIN_PY="${TRAIN_PY:-python3.10}"

echo "[table2-hgpt-generator] start $(date)"
echo "[table2-hgpt-generator] host=$(hostname)"
echo "[table2-hgpt-generator] config=$CONFIG"
echo "[table2-hgpt-generator] work_dir=$WORK_DIR"
echo "[table2-hgpt-generator] genckpt=$GENCKPT max_iters=$MAX_ITERS"
nvidia-smi || true

if [[ ! -e "$GENCKPT" ]]; then
  echo "[table2-hgpt-generator] FATAL: missing GENCKPT=$GENCKPT" >&2
  exit 2
fi

if ! command -v "$TRAIN_PY" >/dev/null 2>&1; then
  TRAIN_PY=python3
fi

export HF_HOME="${HF_HOME:-$REPO/checkpoints/kimodo}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="${TEXT_ENCODERS_DIR:-$REPO/checkpoints/kimodo/text_encoders}"
export PYTHONPATH="$REPO/ref_repo/KIMODO/kimodo:$REPO:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "[table2-hgpt-generator] building/checking HGPT judge venv at $PHYSFLOW_HGPT_VENV"
HGPT_PYTHON="$(
  PHYSFLOW_HGPT_VENV="$PHYSFLOW_HGPT_VENV" \
    bash scripts/embodied/physflow_hgpt_node_setup.sh | tail -1
)"
export PHYSFLOW_HGPT_PYTHON="$HGPT_PYTHON"
echo "[table2-hgpt-generator] hgpt_python=$PHYSFLOW_HGPT_PYTHON train_py=$TRAIN_PY"

"$TRAIN_PY" -c "from mmengine.config import Config; Config.fromfile('$CONFIG'); print('[table2-hgpt-generator] config OK')"

exec "$TRAIN_PY" tools/train.py "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --load-from "$GENCKPT" --load-scope model \
  --cfg-options \
  "train_cfg.max_iters=$MAX_ITERS" \
  "trainer.tracker_qpos_pool_dir=$WORK_DIR/qpos_pool" \
  "default_hooks.checkpoint.interval=150" \
  "default_hooks.checkpoint.max_keep_ckpts=8"
