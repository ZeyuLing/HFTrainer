#!/usr/bin/env bash
# Restart-sliding AGILE hard-overfit with behavior-level action distillation.
#
# By default this reproduces the r2-only repair run. Set START_ROUND=0 and
# BOOTSTRAP_FROM_SRC=0 to run a fresh full closed-loop arm under the same
# conservative/action-distilled tracker updates.
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
ADISTILL_TAG="${ADISTILL_TAG:-adistill1}"
ACTION_DISTILL_COEFF="${ACTION_DISTILL_COEFF:-1.0}"
START_ROUND="${START_ROUND:-2}"
NUM_ROUNDS="${NUM_ROUNDS:-3}"
BOOTSTRAP_FROM_SRC="${BOOTSTRAP_FROM_SRC:-1}"
BOOTSTRAP_UP_TO_ROUND="${BOOTSTRAP_UP_TO_ROUND:-}"
TRAINEE_SNAPSHOT_MODE="${TRAINEE_SNAPSHOT_MODE:-base-plus-latest}"
TRAINEE_EPOCHS="${TRAINEE_EPOCHS:-20}"
EXTRA_TRAINEE_OVERRIDES="${EXTRA_TRAINEE_OVERRIDES:-}"
EXTRA_GEN_CFG_OPTIONS="${EXTRA_GEN_CFG_OPTIONS:-}"
EXTRA_GEN_CFG_OPTIONS_BY_ROUND="${EXTRA_GEN_CFG_OPTIONS_BY_ROUND:-}"
FIXED_REPLAY_ANNO="${FIXED_REPLAY_ANNO:-}"
FIXED_REPLAY_BANK="${FIXED_REPLAY_BANK:-}"
FIXED_REPLAY_PREFIX="${FIXED_REPLAY_PREFIX:-fixed_}"
ARM="hardovf_frontier_gtreplay_restart_sliding_${ADISTILL_TAG}"
ROOT="work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart_sliding_${ADISTILL_TAG}"
SRC_ARM="${SRC_ARM:-work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart/hardovf_frontier_gtreplay_restart}"
RELEASE_CKPT=ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt
cd "$REPO"
echo "[hardovf-restart-sliding-adistill] $(date) host=$(hostname)"
echo "[hardovf-restart-sliding-adistill] tag=$ADISTILL_TAG coeff=$ACTION_DISTILL_COEFF start_round=$START_ROUND num_rounds=$NUM_ROUNDS bootstrap=$BOOTSTRAP_FROM_SRC snapshot_mode=$TRAINEE_SNAPSHOT_MODE trainee_epochs=$TRAINEE_EPOCHS"
if [ -n "$BOOTSTRAP_UP_TO_ROUND" ]; then
  echo "[hardovf-restart-sliding-adistill] bootstrap_up_to_round=$BOOTSTRAP_UP_TO_ROUND"
fi
if [ -n "$EXTRA_GEN_CFG_OPTIONS" ]; then
  echo "[hardovf-restart-sliding-adistill] extra_gen_cfg_options=$EXTRA_GEN_CFG_OPTIONS"
fi
if [ -n "$EXTRA_GEN_CFG_OPTIONS_BY_ROUND" ]; then
  echo "[hardovf-restart-sliding-adistill] extra_gen_cfg_options_by_round=$EXTRA_GEN_CFG_OPTIONS_BY_ROUND"
fi
if [ -n "$EXTRA_TRAINEE_OVERRIDES" ]; then
  echo "[hardovf-restart-sliding-adistill] extra_overrides=$EXTRA_TRAINEE_OVERRIDES"
fi

nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[hardovf-restart-sliding-adistill] host CUDA driver version: ${CUDA_DRV:-unknown}"
if [ -n "$CUDA_DRV" ] && awk "BEGIN{exit !($CUDA_DRV < 11.4)}"; then
  echo "[hardovf-restart-sliding-adistill] FATAL_BAD_NODE: CUDA driver $CUDA_DRV < 11.4. Aborting fast for reschedule."
  exit 42
fi
echo "[hardovf-restart-sliding-adistill] driver gate OK (>=11.4)"

ln -sfn "$ENVDIR/isaacgym" /root/isaacgym
ln -sfn "$ENVDIR/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38 python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -2 || echo "[hardovf-restart-sliding-adistill] WARN dnf partial"
PY38RT="$ENVDIR/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[hardovf-restart-sliding-adistill] restoring base python3.8 from $PY38RT"
  cp -a "$PY38RT/bin/python3.8" /usr/bin/python3.8
  cp -a "$PY38RT/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "$PY38RT/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi
/root/physflow_isaacgym_py38_cu118/bin/python -c \
  "import isaacgym, torch; print('[hardovf-restart-sliding-adistill] isaacgym ok torch', torch.__version__)" 2>&1 | tail -3
/root/physflow_isaacgym_py38_cu118/bin/python -m py_compile \
  ref_repo/ProtoMotions/protomotions/agents/ppo/config.py \
  ref_repo/ProtoMotions/protomotions/agents/ppo/agent.py

export PIP_DEFAULT_TIMEOUT=30
timeout 300 /usr/local/bin/python3 -m pip install --quiet \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -2 \
  || echo "[hardovf-restart-sliding-adistill] WARN py310 judge dep install partial"

export PATH=/usr/local/bin:$PATH
export PYTHONPATH="$REPO:$REPO/ref_repo/ProtoMotions:${PYTHONPATH:-}"
export MUJOCO_GL=disable
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

ARM_DIR="$ROOT/$ARM"
if [ "$BOOTSTRAP_FROM_SRC" = "1" ]; then
  if [ -n "$BOOTSTRAP_UP_TO_ROUND" ]; then
    if [ ! -d "$SRC_ARM/gen/r${BOOTSTRAP_UP_TO_ROUND}/checkpoint-iter_50" ] || [ ! -f "$SRC_ARM/judge_onnx/r${BOOTSTRAP_UP_TO_ROUND}/unified_pipeline.onnx" ] || [ ! -d "$SRC_ARM/trainee/r${BOOTSTRAP_UP_TO_ROUND}_snap" ]; then
      echo "[hardovf-restart-sliding-adistill] FATAL missing partial source arm artifacts through r${BOOTSTRAP_UP_TO_ROUND}: $SRC_ARM"
      exit 5
    fi
  elif [ ! -d "$SRC_ARM/gen/r2/checkpoint-iter_50" ] || [ ! -f "$SRC_ARM/judge_onnx/r1/unified_pipeline.onnx" ]; then
    echo "[hardovf-restart-sliding-adistill] FATAL missing completed source arm artifacts: $SRC_ARM"
    exit 5
  fi
fi

if [ "$BOOTSTRAP_FROM_SRC" = "1" ] && [ -n "$BOOTSTRAP_UP_TO_ROUND" ] && [ ! -f "$ARM_DIR/.partial_bootstrap_round_${BOOTSTRAP_UP_TO_ROUND}" ]; then
  echo "[hardovf-restart-sliding-adistill] partial bootstrap through r${BOOTSTRAP_UP_TO_ROUND} from $SRC_ARM"
  mkdir -p "$ARM_DIR/gen" "$ARM_DIR/judge_onnx" "$ARM_DIR/trainee" "$ARM_DIR/pool"
  rm -rf "$ARM_DIR/pool"
  mkdir -p "$ARM_DIR/pool"
  for ((r=0; r<=BOOTSTRAP_UP_TO_ROUND; r++)); do
    rsync -a "$SRC_ARM/gen/r${r}/" "$ARM_DIR/gen/r${r}/"
    rsync -a "$SRC_ARM/judge_onnx/r${r}/" "$ARM_DIR/judge_onnx/r${r}/"
    rsync -a "$SRC_ARM/trainee/r${r}_snap/" "$ARM_DIR/trainee/r${r}_snap/"
  done
  # Reconstruct the cumulative generator pool at the bootstrap boundary from
  # the source snapshot. Fixed replay files live only in snapshots, not in the
  # generator pool, so exclude the configured prefix when present.
  shopt -s nullglob
  for m in "$SRC_ARM/trainee/r${BOOTSTRAP_UP_TO_ROUND}_snap/"*.motion; do
    bn=$(basename "$m")
    if [ -n "$FIXED_REPLAY_PREFIX" ] && [[ "$bn" == "$FIXED_REPLAY_PREFIX"* ]]; then
      continue
    fi
    cp -a "$m" "$ARM_DIR/pool/$bn"
  done
  shopt -u nullglob
  touch "$ARM_DIR/.partial_bootstrap_round_${BOOTSTRAP_UP_TO_ROUND}"
elif [ "$BOOTSTRAP_FROM_SRC" = "1" ] && [ -z "$BOOTSTRAP_UP_TO_ROUND" ] && [ ! -d "$ARM_DIR/gen/r2/checkpoint-iter_50" ]; then
  echo "[hardovf-restart-sliding-adistill] bootstrap generator/pool/judges/snaps from $SRC_ARM"
  mkdir -p "$ARM_DIR"
  rsync -a "$SRC_ARM/gen/" "$ARM_DIR/gen/"
  rsync -a "$SRC_ARM/pool/" "$ARM_DIR/pool/"
  mkdir -p "$ARM_DIR/judge_onnx" "$ARM_DIR/trainee"
  rsync -a "$SRC_ARM/judge_onnx/r0/" "$ARM_DIR/judge_onnx/r0/"
  rsync -a "$SRC_ARM/judge_onnx/r1/" "$ARM_DIR/judge_onnx/r1/"
  rsync -a "$SRC_ARM/trainee/r0_snap/" "$ARM_DIR/trainee/r0_snap/"
  rsync -a "$SRC_ARM/trainee/r1_snap/" "$ARM_DIR/trainee/r1_snap/"
fi

EXTRA_SNAPSHOT_ARGS=()
if [ -n "$FIXED_REPLAY_ANNO" ] || [ -n "$FIXED_REPLAY_BANK" ]; then
  if [ -z "$FIXED_REPLAY_BANK" ]; then
    FIXED_REPLAY_BANK="$ROOT/fixed_replay_bank"
  fi
  if ! find "$FIXED_REPLAY_BANK" -maxdepth 1 -name '*.motion' -print -quit 2>/dev/null | grep -q .; then
    if [ -z "$FIXED_REPLAY_ANNO" ]; then
      echo "[hardovf-restart-sliding-adistill] FATAL fixed replay bank missing and FIXED_REPLAY_ANNO empty: $FIXED_REPLAY_BANK"
      exit 6
    fi
    echo "[hardovf-restart-sliding-adistill] building fixed replay bank anno=$FIXED_REPLAY_ANNO out=$FIXED_REPLAY_BANK"
    /usr/local/bin/python3 scripts/embodied/build_fixed_g1_replay_bank.py \
      --anno "$FIXED_REPLAY_ANNO" \
      --out "$FIXED_REPLAY_BANK" \
      --prefix "$FIXED_REPLAY_PREFIX" \
      --overwrite
  fi
  echo "[hardovf-restart-sliding-adistill] fixed_replay_bank=$FIXED_REPLAY_BANK"
  EXTRA_SNAPSHOT_ARGS+=(--trainee-extra-motion-dir "$FIXED_REPLAY_BANK")
  EXTRA_SNAPSHOT_ARGS+=(--trainee-extra-motion-prefix "$FIXED_REPLAY_PREFIX")
fi

/usr/local/bin/python3 scripts/embodied/physflow_coevolve_orchestrator.py \
  --arm-name "$ARM" \
  --judge-mode anchor --anchor-alpha 0.8 \
  --start-round "$START_ROUND" \
  --num-rounds "$NUM_ROUNDS" \
  --gen-iters 50 \
  --trainee-epochs "$TRAINEE_EPOCHS" \
  --trainee-restart-each-round \
  --trainee-snapshot-mode "$TRAINEE_SNAPSHOT_MODE" \
  --num-envs 512 --batch-size 4096 \
  --gpu 0 \
  --gen-config configs/physflow/physflow_coevo_hardovf_frontier_g1.py \
  --gen-cfg-options "$EXTRA_GEN_CFG_OPTIONS" \
  --gen-cfg-options-by-round "$EXTRA_GEN_CFG_OPTIONS_BY_ROUND" \
  --gen-init-ckpt work_dirs/hymotion_g1_t2m_38dim_scene_clean_minus_heldout/checkpoint-g1base \
  --trainee-init-ckpt "$RELEASE_CKPT" \
  --trainee-exp "$REPO/ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py" \
  --trainee-overrides "agent.model.actor_optimizer.lr=2e-6,agent.model.critic_optimizer.lr=1e-5,agent.model.discriminator_optimizer.lr=1e-5,agent.num_mini_epochs=1,agent.gradient_clip_val=10.0,agent.action_distill.enabled=True,agent.action_distill.coeff=${ACTION_DISTILL_COEFF},agent.action_distill.reference_checkpoint=$REPO/$RELEASE_CKPT${EXTRA_TRAINEE_OVERRIDES:+,$EXTRA_TRAINEE_OVERRIDES}" \
  --py310 /usr/local/bin/python3 \
  --py38 /root/physflow_isaacgym_py38_cu118/bin/python \
  --root "$ROOT" \
  "${EXTRA_SNAPSHOT_ARGS[@]}"
echo "[hardovf-restart-sliding-adistill] orchestrator exit=$?"
