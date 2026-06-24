#!/usr/bin/env bash
# Run one HYMotion G1 checkpoint quick-eval on a Taiji V100 node.
set -euo pipefail

REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"

CONFIG="${EVAL_CONFIG:-configs/physflow/hymotion_g1_t2m_38dim_long.py}"
CKPT="${EVAL_CKPT:?set EVAL_CKPT}"
OUT="${EVAL_OUT:?set EVAL_OUT}"
NUM_SAMPLES="${EVAL_NUM_SAMPLES:-24}"
MAX_ITEMS="${EVAL_MAX_ITEMS:-4096}"
SAMPLE_STEPS="${EVAL_SAMPLE_STEPS:-30}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SEED="${EVAL_SEED:-20260615}"
SCORE_GT="${EVAL_SCORE_GT:---score-gt}"

echo "[hymotion-g1-eval-node] $(date) host=$(hostname)"
echo "[hymotion-g1-eval-node] config=${CONFIG}"
echo "[hymotion-g1-eval-node] ckpt=${CKPT}"
echo "[hymotion-g1-eval-node] out=${OUT}"

nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[hymotion-g1-eval-node] host CUDA driver version: ${CUDA_DRV:-unknown}"
if [ -n "$CUDA_DRV" ] && awk "BEGIN{exit !($CUDA_DRV < 11.4)}"; then
  echo "[hymotion-g1-eval-node] FATAL_BAD_NODE: CUDA driver $CUDA_DRV < 11.4. Aborting fast for reschedule."
  exit 42
fi
echo "[hymotion-g1-eval-node] driver gate OK (>=11.4)"

ln -sfn "$ENVDIR/isaacgym" /root/isaacgym
ln -sfn "$ENVDIR/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38 python38-devel gcc gcc-c++ make 2>&1 | tail -1 || true

PY38RT="$ENVDIR/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  cp -a "$PY38RT/bin/python3.8" /usr/bin/python3.8
  cp -a "$PY38RT/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "$PY38RT/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi

export PATH=/usr/local/bin:$PATH
export PROJECT_ROOT="$REPO"
export PYTHONPATH="$REPO:$REPO/ref_repo/ProtoMotions:${PYTHONPATH:-}"
export MUJOCO_GL=disable
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PHYSFLOW_CONVERT_PYTHON=/root/physflow_isaacgym_py38_cu118/bin/python
export PIP_DEFAULT_TIMEOUT=30

PY310="${PY310:-/usr/local/bin/python3}"
if [ ! -x "$PY310" ]; then
  PY310="$(command -v python3)"
fi
timeout 300 "$PY310" -m pip install --quiet \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -1 || true

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY310" \
  scripts/embodied/eval_hymotion_g1_checkpoint_frozen.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --out "$OUT" \
  --num-samples "$NUM_SAMPLES" \
  --max-items "$MAX_ITEMS" \
  --sample-steps "$SAMPLE_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  $SCORE_GT

echo "[hymotion-g1-eval-node] done $(date)"
