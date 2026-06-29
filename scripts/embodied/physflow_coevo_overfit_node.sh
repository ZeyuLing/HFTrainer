#!/usr/bin/env bash
# PhysFlow co-evolution OVERFIT on a single Taiji node.
#
# Validates the FULL closed loop end-to-end on 8 fixed prompts BEFORE any formal
# run, via the true-co-evolution orchestrator (judge sync each round):
#   GENERATOR (py3.10 flow-matching RAFT vs judge) -> qpos decode -> JUDGE
#   (in-process MuJoCo + ONNX) -> accept-filter -> reward SFT (+GT mix) -> pool
#   -> TRAINEE (py3.8 IsaacGym PPO) -> JUDGE SYNC (export trainee ONNX -> next
#   round's judge, blended with the frozen anchor).
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"
echo "[overfit-co] $(date) host=$(hostname)"

# --- 1) rehydrate py3.8 IsaacGym env (trainee PPO + CSV->.motion convert + ONNX export) ---
ln -sfn "$ENVDIR/isaacgym" /root/isaacgym
ln -sfn "$ENVDIR/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
echo "[overfit-co] installing python38-devel + build toolchain via dnf"
dnf install -y python38 python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -2 || echo "[overfit-co] WARN dnf partial"
PY38RT="$ENVDIR/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[overfit-co] restoring base python3.8 from $PY38RT"
  cp -a "$PY38RT/bin/python3.8" /usr/bin/python3.8
  cp -a "$PY38RT/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "$PY38RT/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi
/root/physflow_isaacgym_py38_cu118/bin/python -c \
  "import isaacgym, torch; print('[overfit-co] isaacgym ok torch', torch.__version__)" 2>&1 | tail -3

# --- 2) install py3.10 judge deps (in-process MuJoCo + ONNX scoring for generator) ---
export PIP_DEFAULT_TIMEOUT=30
timeout 300 /usr/local/bin/python3 -m pip install --quiet \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -2 \
  || echo "[overfit-co] WARN py310 judge dep install partial"

# --- 3) generator-side env (orchestrator appends HFT to PYTHONPATH; keep PROTO too) ---
export PATH=/usr/local/bin:$PATH
export PYTHONPATH="$REPO:$REPO/ref_repo/ProtoMotions:${PYTHONPATH:-}"
# The G1 judge (run_g1_rl_tracker_export.simulate_and_export) is PHYSICS-ONLY
# (mj_step, no mujoco.Renderer), so it needs NO GL backend. MUJOCO_GL=egl is
# both unnecessary AND fatal on these headless Taiji nodes (egl_ext.py raises
# AttributeError: 'NoneType' has no 'eglQueryString' at mujoco import). Use
# 'disable' -- matches the convert step + trainee/export envs.
export MUJOCO_GL=disable
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# --- 4) run the orchestrator: 3 rounds, tiny gen/trainee budget, anchor judge ---
/usr/local/bin/python3 scripts/embodied/physflow_coevolve_orchestrator.py \
  --arm-name overfit_g1_judgestart \
  --judge-mode anchor --anchor-alpha 0.5 \
  --num-rounds 3 \
  --gen-iters 40 \
  --trainee-epochs 12 \
  --num-envs 512 --batch-size 4096 \
  --gpu 0 \
  --gen-config configs/physflow/physflow_coevo_overfit_g1.py \
  --gen-init-ckpt work_dirs/hymotion_g1_t2m_38dim_scene_clean_minus_heldout/checkpoint-g1base \
  --trainee-init-ckpt ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
  --trainee-exp data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
  --py310 /usr/local/bin/python3 \
  --py38 /root/physflow_isaacgym_py38_cu118/bin/python \
  --root work_dirs/physflow_coevolve_overfit
echo "[overfit-co] orchestrator exit=$?"
