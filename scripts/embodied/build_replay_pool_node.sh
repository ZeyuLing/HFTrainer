#!/usr/bin/env bash
# Build the anti-forgetting replay set (agile-inclusive real G1 motions) and inject
# .motion files into the running co-evolution trainee pool. py3.10 for encode/decode,
# py3.8 IsaacGym env for CSV->.motion conversion. Run on Taiji, MUJOCO_GL=disable.
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"
echo "[replay-build] $(date) host=$(hostname)"

POOL="${1:-work_dirs/physflow_coevolve_formal/formal_ours/pool}"

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

echo "[replay-build] injecting replay into pool: $POOL"
/usr/local/bin/python3 scripts/embodied/build_replay_pool.py \
  --pool "$POOL" \
  --scan 6000 --topk-agile 300 --n-random 120
echo "[replay-build] exit=$?"
