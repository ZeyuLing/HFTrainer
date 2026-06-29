#!/usr/bin/env bash
# PhysFlow LEARNABILITY-FRONTIER (direction-B) co-evolution OVERFIT on one Taiji
# node. Validates the redesigned loop end-to-end on 8 fixed prompts BEFORE the
# formal run: GENERATOR (py3.10 flow-matching RAFT, scored by a Q+T judge
# ensemble) -> per-judge frontier select + regret SFT target + frontier pool
# export -> TRAINEE (py3.8 IsaacGym PPO on the frontier pool) -> JUDGE SYNC.
# Watch the new telemetry: n_frontier_mean (>0 from round 1) and sel_trainee_compl
# (should sit below 0.9: the generator is targeting the trainee's failure band).
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"
echo "[overfit-frontier] $(date) host=$(hostname)"

# ---- driver gate FIRST: IsaacGym PhysX needs host CUDA driver >= 11.4 ----
nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[overfit-frontier] host CUDA driver version: ${CUDA_DRV:-unknown}"
if [ -n "$CUDA_DRV" ] && awk "BEGIN{exit !($CUDA_DRV < 11.4)}"; then
  echo "[overfit-frontier] FATAL_BAD_NODE: CUDA driver $CUDA_DRV < 11.4. Aborting fast for reschedule."
  exit 42
fi
echo "[overfit-frontier] driver gate OK (>=11.4)"

# --- 1) rehydrate py3.8 IsaacGym env ---
ln -sfn "$ENVDIR/isaacgym" /root/isaacgym
ln -sfn "$ENVDIR/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38 python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -2 || echo "[overfit-frontier] WARN dnf partial"
PY38RT="$ENVDIR/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[overfit-frontier] restoring base python3.8 from $PY38RT"
  cp -a "$PY38RT/bin/python3.8" /usr/bin/python3.8
  cp -a "$PY38RT/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "$PY38RT/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi
/root/physflow_isaacgym_py38_cu118/bin/python -c \
  "import isaacgym, torch; print('[overfit-frontier] isaacgym ok torch', torch.__version__)" 2>&1 | tail -3

# --- 2) py3.10 judge deps ---
export PIP_DEFAULT_TIMEOUT=30
timeout 300 /usr/local/bin/python3 -m pip install --quiet \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -2 \
  || echo "[overfit-frontier] WARN py310 judge dep install partial"

export PATH=/usr/local/bin:$PATH
export PYTHONPATH="$REPO:$REPO/ref_repo/ProtoMotions:${PYTHONPATH:-}"
export MUJOCO_GL=disable
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# --- 3) run the orchestrator: 3 rounds, tiny budget, anchor judge (Q frozen + T) ---
/usr/local/bin/python3 scripts/embodied/physflow_coevolve_orchestrator.py \
  --arm-name overfit_frontier_g1 \
  --judge-mode anchor --anchor-alpha 0.8 \
  --num-rounds 3 \
  --gen-iters 40 \
  --trainee-epochs 12 \
  --num-envs 512 --batch-size 4096 \
  --gpu 0 \
  --gen-config configs/physflow/physflow_coevo_overfit_frontier_g1.py \
  --gen-init-ckpt work_dirs/hymotion_g1_t2m_38dim_scene_clean_minus_heldout/checkpoint-g1base \
  --trainee-init-ckpt ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
  --trainee-exp "$REPO/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py" \
  --py310 /usr/local/bin/python3 \
  --py38 /root/physflow_isaacgym_py38_cu118/bin/python \
  --root work_dirs/physflow_coevolve_overfit_frontier
echo "[overfit-frontier] orchestrator exit=$?"
