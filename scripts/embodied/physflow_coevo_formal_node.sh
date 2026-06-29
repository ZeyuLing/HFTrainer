#!/usr/bin/env bash
# PhysFlow FORMAL co-evolution on a Taiji node (8x V100).
#
# The orchestrator is SINGLE-GPU serial (generator phase then trainee phase per
# round), so we fill the node by running several independent arms in parallel,
# each pinned to its own GPU.  This yields the main result + the paper ablations
# in one shot.  The 80 held-out AGILE clips are EXCLUDED from every arm's prompt
# bank (data/annotation/train_g1_t2m_emb_minus_heldout_scene_clean.json) so the
# tracker key-capability eval (frozen baseline vs co-evolved) is truly held out,
# and no training arm learns flat-ground-invalid platform/stair/object-support clips.
#
# Arms (default single-node set):
#   gpu0  formal_ours       anchor judge (alpha .5) + judge warm-start + GT-mix
#                           -> THE deliverable: co-evolved tracker for held-out
#   gpu1  formal_frozengen  frozen judge (no judge sync) -> generator baseline
#                           (does the co-evolution feedback help the generator?)
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ENVDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env
cd "$REPO"
echo "[formal-co] $(date) host=$(hostname) rank=${INDEX:-0}/${HOST_NUM:-1}"

# ---- driver gate FIRST (before any env rehydrate) so a bad node fails in ~10s ----
# IsaacGym PhysX needs host CUDA driver >= 11.4; older-driver hosts fall back to
# CPU pipeline and the trainee crashes ("Must enable GPU pipeline").  Some
# AILab_DHA V100 hosts still ship CUDA 11.0.
nvidia-smi --query-gpu=index,name,driver_version --format=csv 2>&1 | head -10 || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[formal-co] host CUDA driver version: ${CUDA_DRV:-unknown}"
if [ -n "$CUDA_DRV" ] && awk "BEGIN{exit !($CUDA_DRV < 11.4)}"; then
  echo "[formal-co] FATAL_BAD_NODE: CUDA driver $CUDA_DRV < 11.4 (IsaacGym PhysX GPU pipeline unavailable). Aborting fast for reschedule."
  exit 42
fi
echo "[formal-co] driver gate OK (>=11.4)"

# --- 1) rehydrate py3.8 IsaacGym env (trainee PPO + CSV->.motion + ONNX export) ---
ln -sfn "$ENVDIR/isaacgym" /root/isaacgym
ln -sfn "$ENVDIR/physflow_isaacgym_py38_cu118" /root/physflow_isaacgym_py38_cu118
dnf install -y python38 python38-devel gcc gcc-c++ make ninja-build 2>&1 | tail -1 || echo "[formal-co] WARN dnf partial"
PY38RT="$ENVDIR/py38_runtime"
if ! /usr/bin/python3.8 -c "import encodings" >/dev/null 2>&1; then
  echo "[formal-co] restoring base python3.8 from $PY38RT"
  cp -a "$PY38RT/bin/python3.8" /usr/bin/python3.8
  cp -a "$PY38RT/lib64/libpython3.8.so.1.0" /usr/lib64/libpython3.8.so.1.0
  ln -sfn libpython3.8.so.1.0 /usr/lib64/libpython3.8.so
  rsync -a "$PY38RT/lib64/python3.8/" /usr/lib64/python3.8/
  ldconfig 2>/dev/null || true
fi
/root/physflow_isaacgym_py38_cu118/bin/python -c \
  "import isaacgym, torch; print('[formal-co] isaacgym ok torch', torch.__version__)" 2>&1 | tail -2

# --- 2) py3.10 judge deps (in-process MuJoCo + ONNX scoring for generator) ---
export PIP_DEFAULT_TIMEOUT=30
timeout 300 /usr/local/bin/python3 -m pip install --quiet \
  mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -1 \
  || echo "[formal-co] WARN py310 judge dep install partial"

export PATH=/usr/local/bin:$PATH
PROTO_ROOT="$REPO/hftrainer/models/motion/physflow/trackers/protomotions/vendor"
export PYTHONPATH="$REPO:$PROTO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL=disable
# Each arm spawns CPU-heavy MuJoCo judge subprocesses; cap threads so 2+ arms on
# one node don't thrash the CPUs.
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

NROUNDS="${NROUNDS:-10}"
GEN_ITERS="${GEN_ITERS:-120}"
TR_EPOCHS="${TR_EPOCHS:-40}"
# 512 envs / 4096 batch is the V100-16GB-safe setting validated by the overfit
# run; 1024 envs OOMs IsaacGym at sim-creation on these nodes (illegal mem access).
NUM_ENVS="${NUM_ENVS:-512}"
BATCH="${BATCH:-4096}"
ROOT=work_dirs/physflow_coevolve_formal
GENCFG=configs/physflow/physflow_coevo_formal_g1.py
GENCKPT=work_dirs/hymotion_g1_t2m_38dim_scene_clean_minus_heldout/checkpoint-g1base
TR_CKPT="$PROTO_ROOT/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt"
TR_EXP="$PROTO_ROOT/data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py"

run_arm () {  # $1=gpu $2=arm-name $3=judge-mode
  local gpu="$1" name="$2" mode="$3"
  echo "[formal-co] launch arm=$name gpu=$gpu mode=$mode rounds=$NROUNDS gen_iters=$GEN_ITERS tr_epochs=$TR_EPOCHS envs=$NUM_ENVS"
  /usr/local/bin/python3 scripts/embodied/physflow_coevolve_orchestrator.py \
    --arm-name "$name" \
    --judge-mode "$mode" --anchor-alpha 0.5 \
    --num-rounds "$NROUNDS" \
    --gen-iters "$GEN_ITERS" \
    --trainee-epochs "$TR_EPOCHS" \
    --num-envs "$NUM_ENVS" --batch-size "$BATCH" \
    --gpu "$gpu" \
    --gen-config "$GENCFG" \
    --gen-init-ckpt "$GENCKPT" \
    --trainee-init-ckpt "$TR_CKPT" \
    --trainee-exp "$TR_EXP" \
    --py310 /usr/local/bin/python3 \
    --py38 /root/physflow_isaacgym_py38_cu118/bin/python \
    --root "$ROOT" > "$REPO/${name}.orch.log" 2>&1 &
  echo "  pid=$! log=$REPO/${name}.orch.log"
}

# Which arms to run on THIS node (space separated): "ours", "frozengen".
# Default both; set RUN_ARMS=ours to run the single main arm (matches the proven
# single-arm overfit setup exactly, eliminating any two-IsaacGym init race).
RUN_ARMS="${RUN_ARMS:-ours frozengen}"
gpu=0
for a in $RUN_ARMS; do
  case "$a" in
    ours)      run_arm "$gpu" formal_ours      anchor ;;
    frozengen) run_arm "$gpu" formal_frozengen frozen ;;
    *) echo "[formal-co] unknown arm $a"; continue ;;
  esac
  gpu=$((gpu+1))
  sleep 150
done

echo "[formal-co] all arms launched; waiting..."
wait
echo "[formal-co] all arms exited"
