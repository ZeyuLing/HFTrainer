#!/usr/bin/env bash
# Visualize the LATEST co-evolution generator (judgestart run, round-2 ckpt):
#   1) generate the 8 overfit prompts (x4 candidates), roll out under the FROZEN
#      judge, score trackability  (verify_overfit_trackability.py)
#   2) build the 2-column embodied_viz manifest (generated kinematic FK vs judge
#      rollout)                    (build_coevo_track_viz.py)
# Both steps run in the py3.10 judge env (in-process MuJoCo+ONNX, MUJOCO_GL=disable).
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"
echo "[viz-latest] $(date) host=$(hostname)"

CKPT="${1:-work_dirs/physflow_coevolve_overfit/overfit_g1_judgestart/gen/r2/checkpoint-iter_40}"
OUT="${2:-output/coevo_track_latest}"

# py3.8 env (CSV->.motion convert via PHYSFLOW_CONVERT_PYTHON)
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

export PIP_DEFAULT_TIMEOUT=30
timeout 300 /usr/local/bin/python3 -m pip install --quiet \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -1 || true

export PATH=/usr/local/bin:$PATH
export PYTHONPATH="$REPO:$REPO/ref_repo/ProtoMotions:${PYTHONPATH:-}"
export MUJOCO_GL=disable
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "[viz-latest] step 1: generate + frozen-judge rollout from $CKPT"
/usr/local/bin/python3 scripts/embodied/verify_overfit_trackability.py \
  --config configs/physflow/physflow_coevo_overfit_g1.py \
  --checkpoint "$CKPT" \
  --anno data/annotation/_coevo_overfit8.json \
  --out "$OUT" --num-samples 4 --guidance 2.0
echo "[viz-latest] step1 exit=$?"

echo "[viz-latest] step 2: build manifest"
/usr/local/bin/python3 scripts/embodied/build_coevo_track_viz.py \
  --track-dir "$OUT" --out-dir "$OUT/viz"
echo "[viz-latest] step2 exit=$?  manifest=$OUT/viz/manifest.json"
