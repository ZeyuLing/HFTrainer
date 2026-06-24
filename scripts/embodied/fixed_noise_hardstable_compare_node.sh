#!/usr/bin/env bash
# Build a G1 mesh fixed-noise dashboard for base130k vs hardstable Any2Track.
set -euo pipefail

REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"

CONFIG="${HARDSTABLE_CONFIG:-configs/physflow/verify_hymotion_g1_any2track_130k_hardstable_0620.py}"
BASE_CKPT="${HARDSTABLE_BASE_CKPT:-work_dirs/physflow_verify_hymotion_g1_base/checkpoint-iter_130000}"
OPT_CKPT="${HARDSTABLE_OPT_CKPT:-work_dirs/physflow_verify_hymotion_g1_any2track_130k_hardstable_0620/checkpoint-iter_6000}"
RUN_ROOT="${HARDSTABLE_RUN_ROOT:-output/physflow_fixed_noise_hardstable_any2track_compare_0621}"
VIZ_DIR="${HARDSTABLE_VIZ_DIR:-output/physflow_visualizations/hardstable_any2track_fixed_noise}"
NUM_SAMPLES="${HARDSTABLE_NUM_SAMPLES:-24}"
MAX_ITEMS="${HARDSTABLE_MAX_ITEMS:-4096}"
SAMPLE_STEPS="${HARDSTABLE_SAMPLE_STEPS:-30}"
SEED="${HARDSTABLE_SEED:-20260615}"

echo "[hardstable-fixed-noise] $(date) host=$(hostname)"
echo "[hardstable-fixed-noise] config=${CONFIG}"
echo "[hardstable-fixed-noise] base=${BASE_CKPT}"
echo "[hardstable-fixed-noise] opt=${OPT_CKPT}"
echo "[hardstable-fixed-noise] run_root=${RUN_ROOT}"
echo "[hardstable-fixed-noise] viz_dir=${VIZ_DIR}"

nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[hardstable-fixed-noise] host CUDA driver version: ${CUDA_DRV:-unknown}"
if [ "${CUDA_DRV:-}" != "11.4" ]; then
  echo "[hardstable-fixed-noise] FATAL_BAD_NODE: need CUDA driver 11.4, got ${CUDA_DRV:-unknown}"
  exit 42
fi
echo "[hardstable-fixed-noise] driver gate OK (==11.4)"

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
  scripts/embodied/build_hymotion_g1_fixed_noise_generator_compare.py \
  --config "$CONFIG" \
  --run "base130k=${BASE_CKPT}" \
  --run "hardstable6000=${OPT_CKPT}" \
  --out "$RUN_ROOT" \
  --num-samples "$NUM_SAMPLES" \
  --max-items "$MAX_ITEMS" \
  --sample-steps "$SAMPLE_STEPS" \
  --seed "$SEED"

PHYSFLOW_FIXED_NOISE_RUN_ROOT="$RUN_ROOT" \
PHYSFLOW_FIXED_NOISE_OUT_DIR="$VIZ_DIR" \
PHYSFLOW_FIXED_NOISE_BASE_KEY=base130k \
PHYSFLOW_FIXED_NOISE_PROTO_KEY=hardstable6000 \
PHYSFLOW_FIXED_NOISE_TITLE="hardstable_any2track_6000 fixed-noise four-way" \
  "$PY310" scripts/embodied/build_fixed_noise_proto2k_fourway_dashboard.py

echo "[hardstable-fixed-noise] page=${REPO}/${VIZ_DIR}/index.html"
echo "[hardstable-fixed-noise] done $(date)"
