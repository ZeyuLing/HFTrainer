#!/usr/bin/env bash
# Single-GPU PhysFlow launch for CO-LOCATING a small overfit run inside an
# already-running tlinux4 vermo container. Use when the scheduler keeps handing
# fresh single-V100 jobs tlinux3 nodes (bad NVIDIA driver -> guard abort): exec
# into a known-good container and run the tiny job here, sharing one GPU.
#
# Usage (inside the container):
#   bash tools/physflow_single_gpu_start.sh <config.py> [gpu_id] [extra train.py args]
set -eo pipefail

CONFIG="${1:?usage: physflow_single_gpu_start.sh <config.py> [gpu_id] [extra args...]}"
shift || true
GPU="${1:-7}"; shift || true

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PATH=/usr/local/bin:$PATH
export HF_HOME="$PWD/checkpoints/kimodo"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="$PWD/checkpoints/kimodo/text_encoders"
export PHYSFLOW_CONVERT_PYTHON=python3
# lower CPU caps than the multi-rank launcher: we are sharing the box with the
# host job's 8 MuJoCo scorers, so keep this co-located run modest.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export CUDA_VISIBLE_DEVICES="${GPU}"

python3 - <<'PY' 2>/dev/null || {
import mujoco, onnxruntime, dm_control, typer
from packaging.version import Version
assert Version(onnxruntime.__version__) >= Version("1.23.0"), onnxruntime.__version__
PY
  echo "[single] installing/upgrading mujoco onnxruntime dm_control typer ..."
  python3 -m pip install --quiet -U mujoco "onnxruntime>=1.23,<1.24" dm_control typer packaging 2>&1 | tail -3 | sed 's/^/[single] pip /'
}
python3 -c "from mmengine.config import Config; Config.fromfile('$CONFIG')"
echo "[single] launching $CONFIG on GPU ${GPU} (single process)"
exec python3 tools/train.py "$CONFIG" "$@"
