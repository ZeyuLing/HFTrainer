#!/usr/bin/env bash
# Single-GPU PhysFlow training with the Humanoid-GPT tracking JUDGE, for a
# Taiji vermo node (py3.10 trainer) + a py3.11 HGPT judge venv built by
# scripts/embodied/physflow_hgpt_node_setup.sh.
#
# Usage (on the node):
#   bash scripts/embodied/physflow_hgpt_node_setup.sh   # once, builds judge venv
#   CUDA_VISIBLE_DEVICES=0 bash scripts/embodied/physflow_hgpt_single_gpu_start.sh \
#       configs/physflow/physflow_overfit100_hgpt.py
set -eo pipefail

CONFIG="${1:?usage: physflow_hgpt_single_gpu_start.sh <config.py> [extra train.py args...]}"
shift || true

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PATH=/usr/local/bin:$PATH

# KIMODO offline HF env (same as physflow_mn_start.sh); text encoder is dummy /
# features are precomputed, so the 8B encoder is never loaded during training.
export HF_HOME="$PWD/checkpoints/kimodo"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="$PWD/checkpoints/kimodo/text_encoders"

# kimodo package is not pip-installed in the image -> use the in-repo copy.
export PYTHONPATH="$PWD/ref_repo/KIMODO/kimodo:${PYTHONPATH:-}"

# HGPT judge worker (jax/mujoco-mjx in its own py3.11 venv on node-local disk).
export PHYSFLOW_HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-/dev/shm/hgpt_venv311/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "[hgpt-train] config=$CONFIG"
echo "[hgpt-train] hgpt_python=$PHYSFLOW_HGPT_PYTHON gpu=$CUDA_VISIBLE_DEVICES"
if [[ ! -x "$PHYSFLOW_HGPT_PYTHON" ]]; then
    echo "[hgpt-train] ERROR: HGPT judge venv missing; run physflow_hgpt_node_setup.sh first" >&2
    exit 1
fi
python3.10 -c "from mmengine.config import Config; Config.fromfile('$CONFIG')" && echo "[hgpt-train] config OK"

exec python3.10 tools/train.py "$CONFIG" "$@"
