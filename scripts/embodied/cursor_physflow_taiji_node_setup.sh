#!/usr/bin/env bash
# PhysFlow dedicated Taiji node bootstrap.
#
# The KIMODO generation env (py3.10 + einops/mujoco/onnxruntime) ships inside the
# vermo:latest image and resolves the kimodo package via PYTHONPATH from the
# cephfs repo, so it needs no setup. The IsaacGym tracker-training env, however,
# was installed container-local under /root on the debug machine and is NOT in
# the image. We staged it once to cephfs (physflow_env/) and symlink it back to
# the SAME /root paths here so the venv shebangs and the editable isaacgym
# install resolve correctly.
set -u
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
ISAAC_SRC="$ENVDIR/isaacgym"
VENV_SRC="$ENVDIR/physflow_isaacgym_py38_cu118"
ISAAC_DST=/root/isaacgym
VENV_DST=/root/physflow_isaacgym_py38_cu118
MARKER=/root/physflow_node_ready

echo "[setup] $(date) host=$(hostname)"

# Symlink staged env back to the original /root paths (idempotent).
ln -sfn "$ISAAC_SRC" "$ISAAC_DST"
ln -sfn "$VENV_SRC" "$VENV_DST"
echo "[setup] linked $ISAAC_DST -> $ISAAC_SRC"
echo "[setup] linked $VENV_DST -> $VENV_SRC"

cd "$REPO" || { echo "[setup] FATAL: repo not on cephfs"; exit 1; }

# --- Provision base python3.8 if missing (vermo:latest may lack it) ---
# The staged isaacgym venv uses /usr/bin/python3.8 as its base interpreter.
# Some image revisions ship without it, so restore it from the cephfs-staged
# runtime (binary + libpython + stdlib) into the container-local /usr.
PY38RT="$ENVDIR/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[setup] base python3.8 broken/missing -> restoring from $PY38RT"
  cp -a "$PY38RT/bin/python3.8" /usr/bin/python3.8
  cp -a "$PY38RT/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  # The real stdlib lives in /usr/lib64/python3.8 on el8 (lib only has site-packages).
  rsync -a "$PY38RT/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi

# --- Provision py3.10 KIMODO scoring deps if missing (not in vermo:latest) ---
# mujoco/onnxruntime are needed by the runner's import chain (G1 tracker scoring).
# KIMODO uses its own VENDORED llm2vec wrapper, so the pip llm2vec is NOT needed.
if ! /usr/local/bin/python3 -c "import mujoco, onnxruntime" >/dev/null 2>&1; then
  echo "[setup] installing py3.10 deps mujoco/onnxruntime"
  /usr/local/bin/python3 -m pip install --no-cache-dir --root-user-action=ignore \
    mujoco==3.8.1 onnxruntime==1.23.2 >/dev/null 2>&1 || echo "[setup] WARN pip deps failed"
fi

# --- Verify IsaacGym training env ---
ISAAC_OK=0
if "$VENV_DST/bin/python" -c "import isaacgym, torch; print('[verify] isaacgym ok, torch', torch.__version__)" 2>&1; then
  ISAAC_OK=1
else
  echo "[verify] ISAACGYM IMPORT FAILED"
fi

# --- Verify KIMODO generation env (py3.10 + cephfs repo via PYTHONPATH) ---
KIMODO_OK=0
if PYTHONPATH="$REPO/ref_repo/KIMODO/kimodo:$REPO/ref_repo/ProtoMotions:$REPO" \
   /usr/local/bin/python3 -c "import kimodo, einops, mujoco, onnxruntime; print('[verify] kimodo+deps ok')" 2>&1; then
  KIMODO_OK=1
else
  echo "[verify] KIMODO IMPORT FAILED"
fi

# --- GPU sanity ---
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | head -8

if [ "$ISAAC_OK" = 1 ] && [ "$KIMODO_OK" = 1 ]; then
  echo "PHYSFLOW_NODE_READY" | tee "$MARKER"
else
  echo "PHYSFLOW_NODE_DEGRADED isaac=$ISAAC_OK kimodo=$KIMODO_OK" | tee "$MARKER"
fi

echo "[setup] entering keep-alive (sleep infinity); drive the pipeline via taiji_client exec + tmux"
sleep infinity
