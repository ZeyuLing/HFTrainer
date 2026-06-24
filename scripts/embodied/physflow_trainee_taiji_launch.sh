#!/usr/bin/env bash
# PhysFlow co-evolution TRAINEE launcher for a fresh Taiji node.
#
# Rehydrates the py3.8 IsaacGym tracker-training env that was staged to cephfs
# (physflow_env/) back onto the container-local /root paths the venv expects,
# then runs the decoupled trainee round-runner against the LIVE generator pool
# (which the separately-running generator job keeps filling with gen+GT motions).
set -u
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
WORK=$REPO/work_dirs/physflow_online_adv_g1_38dim

echo "[trainee-launch] $(date) host=$(hostname)"

# 1) symlink staged env back to /root (idempotent)
ln -sfn "$ENVDIR/isaacgym" /root/isaacgym
ln -sfn "$ENVDIR/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118

# 2) provision base python3.8 + dev headers + build toolchain.
#    IsaacGym's gymtorch is a JIT-compiled C++/CUDA extension, so it needs
#    Python.h (python38-devel) and ninja/gcc at first import. The staged
#    py38_runtime only carries the runtime stdlib, so install devel via dnf
#    (matches the original env-build recipe) and keep the staged copy as a
#    runtime fallback.
PY38RT="$ENVDIR/py38_runtime"
echo "[trainee-launch] installing python38-devel + build toolchain via dnf"
dnf install -y python38 python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -3 \
  || echo "[trainee-launch] WARN dnf install failed/partial"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[trainee-launch] restoring base python3.8 from $PY38RT"
  cp -a "$PY38RT/bin/python3.8" /usr/bin/python3.8
  cp -a "$PY38RT/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "$PY38RT/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi
# Confirm Python.h is present (gymtorch JIT will fail without it).
PYH=$(/usr/bin/python3.8 -c "import sysconfig;print(sysconfig.get_path('include'))" 2>/dev/null)
echo "[trainee-launch] python3.8 include dir: $PYH ; Python.h: $([ -f "$PYH/Python.h" ] && echo FOUND || echo MISSING)"
# nvcc visibility for the CUDA half of gymtorch.
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -2 || echo "[trainee-launch] WARN nvcc not on PATH (gymtorch CUDA build may fail)"

# 3) verify the tracker env imports before committing the GPU
/root/physflow_isaacgym_py38_cu118/bin/python -c \
  "import isaacgym, torch; print('[trainee-launch] isaacgym ok, torch', torch.__version__)" 2>&1 | tail -3

cd "$REPO" || { echo "[trainee-launch] FATAL: repo not on cephfs"; exit 1; }
mkdir -p "$WORK/trainee_rounds"

# 4) run the decoupled trainee round-runner (driver is py3.10; it spawns the
#    py3.8 isaacgym train_agent via --tracker-python).
CUDA_VISIBLE_DEVICES=0 /usr/local/bin/python3 scripts/embodied/physflow_trainee_round_runner.py \
  --pool-dir "$WORK/tracker_motion_pool" \
  --out-root "$WORK/trainee_rounds" \
  --warmstart-ckpt output/physflow_kimodo_g1/checkpoints/g1_xyvel_partial_warmstart.ckpt \
  --tracker-python /root/physflow_isaacgym_py38_cu118/bin/python \
  --experiment ref_repo/ProtoMotions/examples/experiments/mimic/physflow_g1_xy_offset.py \
  --num-envs 1024 --batch-size 8192 --epochs-per-round 30 \
  --pool-sample 800 --recent-frac 0.5 \
  --min-motions 24 --min-new 16 --max-rounds 100 --poll-sec 120
echo "[trainee-launch] runner exited code=$?"
