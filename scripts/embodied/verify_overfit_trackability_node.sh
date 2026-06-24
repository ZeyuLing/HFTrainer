#!/usr/bin/env bash
# Run the co-evolution overfit TRACKABILITY verification on a Taiji node:
# generate the 8 overfit prompts with the round-0 generator checkpoint, roll
# each out under the FROZEN judge, and report per-prompt completion/fall +
# keep the robot rollout JSON for visualization.
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"
echo "[verify-track] $(date) host=$(hostname)"

# py3.8 env (only needed for CSV->.motion convert via PHYSFLOW_CONVERT_PYTHON)
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

# py3.10 judge deps (in-process MuJoCo + ONNX scoring)
export PIP_DEFAULT_TIMEOUT=30
timeout 300 /usr/local/bin/python3 -m pip install --quiet \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -1 || true

export PATH=/usr/local/bin:$PATH
export PYTHONPATH="$REPO:$REPO/ref_repo/ProtoMotions:${PYTHONPATH:-}"
export MUJOCO_GL=disable
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

CKPT="${1:-work_dirs/physflow_coevolve_overfit/overfit_g1_co/gen/r0/checkpoint-iter_40}"
OUT="${2:-output/coevo_overfit_track}"
/usr/local/bin/python3 scripts/embodied/verify_overfit_trackability.py \
  --config configs/physflow/physflow_coevo_overfit_g1.py \
  --checkpoint "$CKPT" \
  --anno data/annotation/_coevo_overfit8.json \
  --out "$OUT" --num-samples 4 --guidance 2.0
echo "[verify-track] exit=$?"
