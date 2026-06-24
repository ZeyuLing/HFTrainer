#!/usr/bin/env bash
# Score the 80 held-out AGILE clips under the FROZEN g1-bones judge to find
# head-room (clips the baseline tracker drops). Runs in the py3.10 judge env;
# CSV->.motion conversion uses the py3.8 IsaacGym env (PHYSFLOW_CONVERT_PYTHON).
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"
echo "[heldout-score] $(date) host=$(hostname)"

OUT="${1:-output/heldout_frozen_score}"
# Optional: re-score under a CO-EVOLVED round's exported tracker ONNX. When set,
# the make-or-break paired comparison vs the frozen baseline on the SAME 80 clips.
SCORE_ONNX="${SCORE_ONNX:-${2:-}}"

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
export PHYSFLOW_CONVERT_PYTHON=/root/physflow_isaacgym_py38_cu118/bin/python

ONNX_ARG=()
if [ -n "$SCORE_ONNX" ]; then
  echo "[heldout-score] scoring held-out agile clips under CO-EVOLVED tracker: $SCORE_ONNX -> $OUT"
  ONNX_ARG=(--onnx "$SCORE_ONNX")
else
  echo "[heldout-score] scoring held-out agile clips under FROZEN judge -> $OUT"
fi
/usr/local/bin/python3 scripts/embodied/score_heldout_frozen.py \
  --heldout data/annotation/_heldout_agile.json \
  --out "$OUT" "${ONNX_ARG[@]}"
echo "[heldout-score] exit=$?  result=$OUT/heldout_score.json"
