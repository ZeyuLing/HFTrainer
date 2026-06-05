#!/usr/bin/env bash
# Per-node bootstrap + multi-node launch for PhysFlow online-adversarial
# training on FRESH vermo Taiji containers.
#
# Runs on EVERY MPI pod (the Taiji job sets exec_start_in_all_mpi_pods=true):
# each node bootstraps the MuJoCo/convert deps that the vermo image does not
# ship, sets the KIMODO offline-HF env, then hands off to taiji_dist_train.sh
# which reads NODE_LIST/NODE_NUM/CHIEF_IP/INDEX and runs `accelerate launch`
# for this node's rank.
#
# Usage (from the Taiji start_cmd):
#   cd <repo> && bash tools/physflow_mn_start.sh <config.py> --auto-resume
set -eo pipefail

CONFIG="${1:?usage: physflow_mn_start.sh <config.py> [extra train.py args...]}"
shift || true

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# /usr/local/bin holds python3.10 + accelerate in the vermo image.
export PATH=/usr/local/bin:$PATH

# HF offline: the LLM2Vec 8B text encoder is cached on shared cephfs; point all
# HF cache vars at it BEFORE python starts, and use a flat dir of local
# snapshot symlinks (TEXT_ENCODERS_DIR) so transformers treats the model id as
# a local path and skips the offline-mode-aborting network revision check.
export HF_HOME="$PWD/checkpoints/kimodo"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="$PWD/checkpoints/kimodo/text_encoders"

# csv->.motion converter: vermo has no IsaacGym py3.8 venv (that only exists on
# the hand-built debug machine), and the converter does not actually need
# IsaacGym, so point it at the container python.
export PHYSFLOW_CONVERT_PYTHON=python3

# CPU thread caps: 8 ranks per node share 96 cores; without a cap the 8 parallel
# MuJoCo/convert scorers oversubscribe and slow every rank down.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# vermo ships KIMODO + LLM2Vec + torch but NOT the MuJoCo tracker-scoring deps.
# Install them from the internal PyPI mirror (idempotent: skips if importable).
# dm_control + typer -> csv->.motion converter; mujoco + onnxruntime -> rollout.
python3 -c "import mujoco, onnxruntime, dm_control, typer" 2>/dev/null || {
  echo "[mn] installing mujoco onnxruntime dm_control typer from internal mirror ..."
  python3 -m pip install --quiet mujoco onnxruntime dm_control typer 2>&1 | tail -3 | sed 's/^/[mn] pip /'
}
python3 -c "import mujoco, onnxruntime, dm_control; print('[mn] deps OK: mujoco', mujoco.__version__, 'onnxruntime', onnxruntime.__version__)"

# Config sanity gate: fail fast on this node before the NCCL rendezvous so a
# config typo doesn't hang the whole job waiting for a crashed peer.
python3 -c "from mmengine.config import Config; Config.fromfile('$CONFIG')"
echo "[mn] config OK: $CONFIG"

echo "[mn] handing off to taiji_dist_train.sh (node INDEX=${INDEX:-?} of NODE_LIST=${NODE_LIST:-?})"
exec bash tools/taiji_dist_train.sh "$CONFIG" "$@"
