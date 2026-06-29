#!/usr/bin/env bash
# PhysFlow AGILE hard-overfit with frozen-policy actor anchor.
# Keeps Direction-B generator adversarial/frontier logic, but stabilizes the
# BeyondMimic/AMP trainee so PPO fine-tuning does not drift away from the SOTA
# released tracker across rounds.
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
ARM=hardovf_frontier_gtreplay_anchor100k
ROOT=work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_anchor
SRC_ARM=work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_cons/hardovf_frontier_gtreplay_cons
FROZEN_CKPT=$REPO/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt
cd "$REPO"
echo "[hardovf-anchor] $(date) host=$(hostname)"

# ---- driver gate FIRST: IsaacGym PhysX needs host CUDA driver >= 11.4 ----
nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[hardovf-anchor] host CUDA driver version: ${CUDA_DRV:-unknown}"
if [ -n "$CUDA_DRV" ] && awk "BEGIN{exit !($CUDA_DRV < 11.4)}"; then
  echo "[hardovf-anchor] FATAL_BAD_NODE: CUDA driver $CUDA_DRV < 11.4. Aborting fast for reschedule."
  exit 42
fi
echo "[hardovf-anchor] driver gate OK (>=11.4)"

# --- 1) rehydrate py3.8 IsaacGym env ---
ln -sfn "$ENVDIR/isaacgym" /root/isaacgym
ln -sfn "$ENVDIR/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38 python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -2 || echo "[hardovf-anchor] WARN dnf partial"
PY38RT="$ENVDIR/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[hardovf-anchor] restoring base python3.8 from $PY38RT"
  cp -a "$PY38RT/bin/python3.8" /usr/bin/python3.8
  cp -a "$PY38RT/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "$PY38RT/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi
/root/physflow_isaacgym_py38_cu118/bin/python -c \
  "import isaacgym, torch; print('[hardovf-anchor] isaacgym ok torch', torch.__version__)" 2>&1 | tail -3

# --- 2) py3.10 judge deps ---
export PIP_DEFAULT_TIMEOUT=30
timeout 300 /usr/local/bin/python3 -m pip install --quiet \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -2 \
  || echo "[hardovf-anchor] WARN py310 judge dep install partial"

export PATH=/usr/local/bin:$PATH
export PYTHONPATH="$REPO:$REPO/ref_repo/ProtoMotions:${PYTHONPATH:-}"
export MUJOCO_GL=disable
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# Reuse the already-completed r0 generator and r0 pool from the conservative
# run. Round0 generator uses only the frozen judge, so the generated pool is
# identical for this trainee-stabilization ablation.
ARM_DIR="$ROOT/$ARM"
if [ ! -d "$ARM_DIR/gen/r0/checkpoint-iter_50" ] && [ -d "$SRC_ARM/gen/r0/checkpoint-iter_50" ]; then
  echo "[hardovf-anchor] bootstrap r0 generator/pool from $SRC_ARM"
  mkdir -p "$ARM_DIR/gen" "$ARM_DIR/pool"
  rsync -a "$SRC_ARM/gen/r0/" "$ARM_DIR/gen/r0/"
  rsync -a "$SRC_ARM/trainee/r0_snap/"*.motion "$ARM_DIR/pool/"
fi

# --- 3) orchestrator: 3 rounds is enough to see whether r1/r2 stay above frozen.
/usr/local/bin/python3 scripts/embodied/physflow_coevolve_orchestrator.py \
  --arm-name "$ARM" \
  --judge-mode anchor --anchor-alpha 0.8 \
  --num-rounds 3 \
  --gen-iters 50 \
  --trainee-epochs 20 \
  --num-envs 512 --batch-size 4096 \
  --gpu 0 \
  --gen-config configs/physflow/physflow_coevo_hardovf_frontier_g1.py \
  --gen-init-ckpt work_dirs/hymotion_g1_t2m_38dim_scene_clean_minus_heldout/checkpoint-g1base \
  --trainee-init-ckpt ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
  --trainee-exp "$REPO/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py" \
  --trainee-overrides "agent.model.actor_optimizer.lr=2e-6,agent.model.critic_optimizer.lr=1e-5,agent.model.discriminator_optimizer.lr=1e-5,agent.num_mini_epochs=1,agent.gradient_clip_val=10.0,agent.policy_anchor.enabled=True,agent.policy_anchor.coeff=100000.0,agent.policy_anchor.reference_checkpoint=$FROZEN_CKPT,agent.policy_anchor.freeze_buffers=True" \
  --py310 /usr/local/bin/python3 \
  --py38 /root/physflow_isaacgym_py38_cu118/bin/python \
  --root "$ROOT"
echo "[hardovf-anchor] orchestrator exit=$?"
