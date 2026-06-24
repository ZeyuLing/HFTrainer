#!/usr/bin/env bash
# ProtoMotions co-evolution from the latest converged HYMotion-G1 generator.
set -euo pipefail

REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"

GENCKPT="${GENCKPT:-work_dirs/physflow_verify_hymotion_g1_base/checkpoint-iter_130000}"
ROOT="${ROOT:-work_dirs/physflow_coevolve_proto_hymotion132k_e5c2}"
ARM="${ARM:-proto_hymotion132k_frontier_e5c2}"
NROUNDS="${NROUNDS:-3}"
GEN_ITERS="${GEN_ITERS:-120}"
TR_EPOCHS="${TR_EPOCHS:-5}"
NUM_ENVS="${NUM_ENVS:-512}"
BATCH="${BATCH:-4096}"
ACTION_DISTILL_COEFF="${ACTION_DISTILL_COEFF:-2.0}"

PROTO_ROOT="$REPO/hftrainer/models/motion/physflow/trackers/protomotions/vendor"
TR_CKPT="$PROTO_ROOT/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt"
TR_EXP="$PROTO_ROOT/data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py"

echo "[proto-coevo] $(date) host=$(hostname)"
echo "[proto-coevo] genckpt=$GENCKPT arm=$ARM root=$ROOT rounds=$NROUNDS gen_iters=$GEN_ITERS tr_epochs=$TR_EPOCHS coeff=$ACTION_DISTILL_COEFF"
nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true
if [[ ! -e "$GENCKPT" ]]; then
  echo "[proto-coevo] FATAL: GENCKPT does not exist: $GENCKPT" >&2
  exit 2
fi
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[proto-coevo] host CUDA driver version: ${CUDA_DRV:-unknown}"
if [ -n "$CUDA_DRV" ] && awk "BEGIN{exit !($CUDA_DRV < 11.4)}"; then
  echo "[proto-coevo] FATAL_BAD_NODE: CUDA driver $CUDA_DRV < 11.4"
  exit 42
fi

ln -sfn "$ENVDIR/isaacgym" /root/isaacgym
ln -sfn "$ENVDIR/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38 python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -1 || true
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
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -1 \
  || echo "[proto-coevo] WARN py310 judge dep install partial"

export PATH=/usr/local/bin:$PATH
export PYTHONPATH="$REPO:$PROTO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL=disable
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

/usr/local/bin/python3 scripts/embodied/physflow_coevolve_orchestrator.py \
  --arm-name "$ARM" \
  --judge-mode anchor --anchor-alpha 0.8 \
  --num-rounds "$NROUNDS" \
  --gen-iters "$GEN_ITERS" \
  --trainee-epochs "$TR_EPOCHS" \
  --trainee-restart-each-round \
  --trainee-snapshot-mode base-plus-latest \
  --num-envs "$NUM_ENVS" --batch-size "$BATCH" \
  --gpu 0 \
  --gen-config configs/physflow/physflow_coevo_frontier_g1.py \
  --gen-init-ckpt "$GENCKPT" \
  --trainee-init-ckpt "$TR_CKPT" \
  --trainee-exp "$TR_EXP" \
  --trainee-overrides "agent.model.actor_optimizer.lr=2e-6,agent.model.critic_optimizer.lr=1e-5,agent.model.discriminator_optimizer.lr=1e-5,agent.num_mini_epochs=1,agent.gradient_clip_val=10.0,agent.action_distill.enabled=True,agent.action_distill.coeff=${ACTION_DISTILL_COEFF},agent.action_distill.reference_checkpoint=$TR_CKPT" \
  --py310 /usr/local/bin/python3 \
  --py38 /root/physflow_isaacgym_py38_cu118/bin/python \
  --root "$ROOT"

STATE_JSONL="$ROOT/$ARM/state.jsonl"
if [[ -s "$STATE_JSONL" ]] && grep -Eq '"event": "[^"]*_failed"' "$STATE_JSONL"; then
  echo "[proto-coevo] FATAL: failed event found in $STATE_JSONL" >&2
  tail -20 "$STATE_JSONL" >&2
  exit 1
fi

echo "[proto-coevo] done $(date)"
