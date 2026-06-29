#!/usr/bin/env bash
# PhysFlow LEARNABILITY-FRONTIER (direction-B) co-evolution on a Taiji node.
#
# The redesign: the generator's best-of-N no longer picks the EASIEST-to-track
# candidate (which gave the trainee nothing to learn). Instead, with a frozen
# quality certifier Q + the live trainee T in the SAME judge ensemble (anchor
# mode), it (a) pulls the SFT target toward the regret-max VALID candidate
# (Q-trackable, hardest for T) and (b) exports the learnability-frontier set
# (Q-valid AND T-struggles-but-learnable) to the trainee pool. This is what can
# make the tracker genuinely improve.
#
# Ablation arms (run in parallel on one 8x V100 node):
#   gpu0  frontier_ours      frontier_mode config + anchor judge  -> the method
#   gpu1  frontier_baseline  easiest-export config + FROZEN judge -> the prior
#                            method exactly as it ran (best-of-N = easiest, single
#                            frozen judge); this is the loop that degraded the
#                            tracker, the honest "before direction-B" comparison.
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"
echo "[frontier-co] $(date) host=$(hostname) rank=${INDEX:-0}/${HOST_NUM:-1}"

# ---- driver gate FIRST (before any env rehydrate) so a bad node fails in ~10s ----
nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[frontier-co] host CUDA driver version: ${CUDA_DRV:-unknown}"
if [ -n "$CUDA_DRV" ] && awk "BEGIN{exit !($CUDA_DRV < 11.4)}"; then
  echo "[frontier-co] FATAL_BAD_NODE: CUDA driver $CUDA_DRV < 11.4 (IsaacGym PhysX GPU pipeline unavailable). Aborting fast for reschedule."
  exit 42
fi
echo "[frontier-co] driver gate OK (>=11.4)"

# --- 1) rehydrate py3.8 IsaacGym env (trainee PPO + CSV->.motion + ONNX export) ---
ln -sfn "$ENVDIR/isaacgym" /root/isaacgym
ln -sfn "$ENVDIR/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38 python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -1 || echo "[frontier-co] WARN dnf partial"
PY38RT="$ENVDIR/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[frontier-co] restoring base python3.8 from $PY38RT"
  cp -a "$PY38RT/bin/python3.8" /usr/bin/python3.8
  cp -a "$PY38RT/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "$PY38RT/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi
/root/physflow_isaacgym_py38_cu118/bin/python -c \
  "import isaacgym, torch; print('[frontier-co] isaacgym ok torch', torch.__version__)" 2>&1 | tail -2

# --- 2) py3.10 judge deps (in-process MuJoCo + ONNX scoring for generator) ---
export PIP_DEFAULT_TIMEOUT=30
timeout 300 /usr/local/bin/python3 -m pip install --quiet \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -1 \
  || echo "[frontier-co] WARN py310 judge dep install partial"

export PATH=/usr/local/bin:$PATH
PROTO_ROOT="$REPO/hftrainer/models/motion/physflow/trackers/protomotions/vendor"
export PYTHONPATH="$REPO:$PROTO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL=disable
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

NROUNDS="${NROUNDS:-10}"
GEN_ITERS="${GEN_ITERS:-120}"
TR_EPOCHS="${TR_EPOCHS:-40}"
NUM_ENVS="${NUM_ENVS:-512}"
BATCH="${BATCH:-4096}"
ROOT=work_dirs/physflow_coevolve_frontier
GENCKPT=work_dirs/hymotion_g1_t2m_38dim_scene_clean_minus_heldout/checkpoint-g1base
TR_CKPT="$PROTO_ROOT/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt"
TR_EXP="$PROTO_ROOT/data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py"

# anchor-alpha 0.8: frozen Q dominates the COMBINED score (used only as the
# round-0 cold-start fallback before a trainee exists); the frontier logic reads
# per_judge["frozen"/"trainee"] directly so alpha is not otherwise critical.
run_arm () {  # $1=gpu $2=arm-name $3=gen-config $4=judge-mode
  local gpu="$1" name="$2" cfg="$3" mode="$4"
  echo "[frontier-co] launch arm=$name gpu=$gpu cfg=$cfg mode=$mode rounds=$NROUNDS gen_iters=$GEN_ITERS tr_epochs=$TR_EPOCHS envs=$NUM_ENVS"
  /usr/local/bin/python3 scripts/embodied/physflow_coevolve_orchestrator.py \
    --arm-name "$name" \
    --judge-mode "$mode" --anchor-alpha 0.8 \
    --num-rounds "$NROUNDS" \
    --gen-iters "$GEN_ITERS" \
    --trainee-epochs "$TR_EPOCHS" \
    --num-envs "$NUM_ENVS" --batch-size "$BATCH" \
    --gpu "$gpu" \
    --gen-config "$cfg" \
    --gen-init-ckpt "$GENCKPT" \
    --trainee-init-ckpt "$TR_CKPT" \
    --trainee-exp "$TR_EXP" \
    --py310 /usr/local/bin/python3 \
    --py38 /root/physflow_isaacgym_py38_cu118/bin/python \
    --root "$ROOT" > "$REPO/${name}.orch.log" 2>&1 &
  echo "  pid=$! log=$REPO/${name}.orch.log"
}

# Which arms to run on THIS node (space separated): "ours", "baseline".
RUN_ARMS="${RUN_ARMS:-ours baseline}"
gpu=0
for a in $RUN_ARMS; do
  case "$a" in
    ours)     run_arm "$gpu" frontier_ours     configs/physflow/physflow_coevo_frontier_g1.py anchor ;;
    baseline) run_arm "$gpu" frontier_baseline configs/physflow/physflow_coevo_formal_g1.py   frozen ;;
    *) echo "[frontier-co] unknown arm $a"; continue ;;
  esac
  gpu=$((gpu+1))
  sleep 150
done

echo "[frontier-co] all arms launched; waiting..."
wait
echo "[frontier-co] all arms exited"
