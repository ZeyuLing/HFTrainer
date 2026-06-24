#!/usr/bin/env bash
# Any2Track closed-loop: generator round -> qpos pool -> Any2Track update -> next judge.
set -euo pipefail

REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "$REPO"

GENCKPT="${GENCKPT:-work_dirs/physflow_verify_hymotion_g1_base/checkpoint-iter_130000}"
ROOT="${ROOT:-work_dirs/physflow_coevolve_any2track_hymotion132k}"
ARM="${ARM:-any2track_hymotion132k_closedloop}"
NROUNDS="${NROUNDS:-2}"
GEN_ITERS="${GEN_ITERS:-120}"
A2T_TIMESTEPS="${A2T_TIMESTEPS:-200000000}"
NUM_GPUS="${NUM_GPUS:-8}"
ADV_PROB="${ADV_PROB:-0.35}"
A2T_ADV_MAX_FILES="${A2T_ADV_MAX_FILES:-96}"
A2T_ADV_SELECTION_STRATEGY="${A2T_ADV_SELECTION_STRATEGY:-evenly}"
A2T_ADV_SELECTION_SEED="${A2T_ADV_SELECTION_SEED:-0}"

ANY_ONNX="${ANY_ONNX:-$REPO/hftrainer/models/motion/physflow/trackers/any2track/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/model.onnx}"
ANY_CONFIG="${ANY_CONFIG:-$REPO/hftrainer/models/motion/physflow/trackers/any2track/storage/logs/dagger/general_tracker_lafan1_v2/config.json}"

mkdir -p "$ROOT/$ARM"
echo "[any2track-coevo] $(date) host=$(hostname)"
echo "[any2track-coevo] genckpt=$GENCKPT root=$ROOT arm=$ARM rounds=$NROUNDS gen_iters=$GEN_ITERS timesteps=$A2T_TIMESTEPS"
nvidia-smi || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
MIN_CUDA_DRV="${MIN_CUDA_DRV:-11.4}"
MAX_CUDA_DRV="${MAX_CUDA_DRV:-}"
echo "[any2track-coevo] host CUDA driver version: ${CUDA_DRV:-unknown}; required >= ${MIN_CUDA_DRV}${MAX_CUDA_DRV:+ and <= ${MAX_CUDA_DRV}}"
if [ -n "${CUDA_DRV:-}" ] && awk "BEGIN{exit !(${CUDA_DRV} < ${MIN_CUDA_DRV})}"; then
  echo "[any2track-coevo] FATAL_BAD_NODE: CUDA driver ${CUDA_DRV} < ${MIN_CUDA_DRV}; use a CUDA11.4 V100 node for OpenTrack." >&2
  exit 86
fi
if [ -n "${CUDA_DRV:-}" ] && [ -n "${MAX_CUDA_DRV:-}" ] && awk "BEGIN{exit !(${CUDA_DRV} > ${MAX_CUDA_DRV})}"; then
  echo "[any2track-coevo] FATAL_BAD_NODE: CUDA driver ${CUDA_DRV} > ${MAX_CUDA_DRV}; use a CUDA11.4 V100 node for OpenTrack." >&2
  exit 87
fi
if [[ ! -e "$GENCKPT" ]]; then
  echo "[any2track-coevo] FATAL: GENCKPT does not exist: $GENCKPT" >&2
  exit 2
fi

if command -v uv >/dev/null 2>&1; then
  UV_CMD=(uv)
else
  python3 -m pip install --user -q uv
  UV_CMD=(python3 -m uv)
fi

OPEN_VENV_DIR="${OPENTRACK_VENV_DIR:-${UV_PROJECT_ENVIRONMENT:-.venv}}"
if [[ "$OPEN_VENV_DIR" == /* ]]; then
  OPEN_VENV_PATH="$OPEN_VENV_DIR"
else
  OPEN_VENV_PATH="$REPO/ref_repo/OpenTrack/$OPEN_VENV_DIR"
fi
export UV_PROJECT_ENVIRONMENT="$OPEN_VENV_PATH"
export GLI_PATH="${GLI_PATH:-$REPO/ref_repo/OpenTrack}"
OPENTRACK_CUDA_FLAVOR="${OPENTRACK_CUDA_FLAVOR:-cuda11}"
CUDA_LIB_DIRS=()
for d in /usr/local/cuda*/extras/CUPTI/lib64 /usr/local/cuda*/lib64; do
  [[ -d "${d}" ]] && CUDA_LIB_DIRS+=("${d}")
done
if [[ "${#CUDA_LIB_DIRS[@]}" -gt 0 ]]; then
  CUDA_LD_PATH="$(IFS=:; echo "${CUDA_LIB_DIRS[*]}")"
  export LD_LIBRARY_PATH="${CUDA_LD_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "[any2track-coevo] LD_LIBRARY_PATH prepended with ${CUDA_LD_PATH}"
fi

if [[ "${SKIP_UV_SYNC:-0}" == "1" ]]; then
  echo "[any2track-coevo] SKIP_UV_SYNC=1; reusing $OPEN_VENV_PATH"
elif [[ -x "$OPEN_VENV_PATH/bin/python" ]] && "$OPEN_VENV_PATH/bin/python" - <<'PY' >/dev/null 2>&1
import track_mj, torch, mujoco, jax  # noqa: F401

assert hasattr(jax.config, "define_bool_state")
assert hasattr(jax.random, "KeyArray")
PY
then
  echo "[any2track-coevo] OpenTrack venv import check passed; skip env sync"
elif [[ "$OPENTRACK_CUDA_FLAVOR" == "cuda11" ]]; then
  echo "[any2track-coevo] preparing OpenTrack CUDA11 env at $OPEN_VENV_PATH"
  PROJECT_ROOT="$REPO" OPENTRACK_ROOT="$REPO/ref_repo/OpenTrack" \
    bash scripts/embodied/prepare_opentrack_cuda11_env.sh "$OPEN_VENV_PATH"
else
  if [[ "$OPEN_VENV_PATH" == "$REPO/ref_repo/OpenTrack/.venv" ]]; then
    LOCK_FILE="$REPO/work_dirs/opentrack_uv_sync.lock"
  else
    LOCK_FILE="${OPEN_VENV_PATH}.sync.lock"
  fi
  mkdir -p "$(dirname "$LOCK_FILE")"
  echo "[any2track-coevo] acquiring uv sync lock: $LOCK_FILE"
  echo "[any2track-coevo] UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
  (
    cd ref_repo/OpenTrack
    flock "$LOCK_FILE" "${UV_CMD[@]}" sync -i https://pypi.org/simple
  )
fi
"$OPEN_VENV_PATH/bin/python" "$REPO/scripts/embodied/patch_opentrack_brax_uint64.py"

TRAIN_PYTHON="${TRAIN_PYTHON:-/usr/local/bin/python3}"
if [[ ! -x "$TRAIN_PYTHON" ]]; then
  TRAIN_PYTHON="$(command -v python3)"
fi
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
if [[ "$MUJOCO_GL" == "egl" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
else
  unset PYOPENGL_PLATFORM
fi
if ! "$TRAIN_PYTHON" - <<'PY' >/dev/null 2>&1
import mujoco
import onnxruntime
import dm_control
import typer
PY
then
  echo "[any2track-coevo] installing generator judge deps for $TRAIN_PYTHON"
  timeout 600 "$TRAIN_PYTHON" -m pip install --quiet \
    -i https://mirrors.tencent.com/pypi/simple --trusted-host mirrors.tencent.com \
    mujoco==3.9.0 onnxruntime==1.23.2 dm_control typer 2>&1 | tail -3
fi
"$TRAIN_PYTHON" - <<'PY'
import mujoco
import onnxruntime
import dm_control
import typer
print("[any2track-coevo] generator imports OK", "mujoco", mujoco.__version__, "onnxruntime", onnxruntime.__version__)
PY

export HF_HOME="${HF_HOME:-checkpoints/kimodo}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

CUR_GEN="$GENCKPT"
for r in $(seq 0 $((NROUNDS - 1))); do
  ROUND_DIR="$ROOT/$ARM/r${r}"
  GEN_WORK="$ROUND_DIR/gen"
  POOL="$ROUND_DIR/qpos_pool"
  mkdir -p "$GEN_WORK" "$POOL"
  echo "[any2track-coevo] round=$r judge=$ANY_ONNX"

  if ! find "$GEN_WORK" -maxdepth 1 -type d -name 'checkpoint-iter_*' -print -quit | grep -q .; then
    "$TRAIN_PYTHON" tools/train.py configs/physflow/physflow_coevo_any2track_g1.py \
      --work-dir "$GEN_WORK" \
      --load-from "$CUR_GEN" --load-scope model \
      --cfg-options \
      "train_cfg.max_iters=$GEN_ITERS" \
      "trainer.judge_onnx=$ANY_ONNX" \
      "trainer.any2track_config=$ANY_CONFIG" \
      "trainer.tracker_qpos_pool_dir=$POOL" \
      "default_hooks.checkpoint.interval=$GEN_ITERS" \
      "default_hooks.checkpoint.max_keep_ckpts=2"
  else
    echo "[any2track-coevo] generator round $r already has checkpoint; skip"
  fi

  NEXT_GEN="$(find "$GEN_WORK" -maxdepth 1 -type d -name 'checkpoint-iter_*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [ -n "$NEXT_GEN" ]; then
    CUR_GEN="$NEXT_GEN"
  fi
  npool=$(find "$POOL" -maxdepth 1 -name '*.npz' | wc -l)
  echo "[any2track-coevo] round=$r qpos_pool=$npool"
  if [ "$npool" -le 0 ]; then
    echo "[any2track-coevo] ERROR: empty qpos pool; cannot update Any2Track" >&2
    exit 3
  fi

  BASE_ONNX_REL="storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/model.onnx"
  BASE_CKPT_REL="storage/logs/dagger/general_tracker_lafan1_v2"
  if [[ "$ANY_ONNX" == "$REPO/ref_repo/OpenTrack/"* ]]; then
    BASE_ONNX_REL="${ANY_ONNX#$REPO/ref_repo/OpenTrack/}"
    BASE_CKPT_REL="${BASE_ONNX_REL%/checkpoints/model.onnx}"
  fi
  (
    source "$OPEN_VENV_PATH/bin/activate"
    export GLI_PATH="${GLI_PATH:-$REPO/ref_repo/OpenTrack}"
    TAG="${ARM}_r${r}" \
    ADV_SOURCE_DIR="$POOL" \
    ADV_KEYWORDS="" \
    ADV_MAX_FILES="$A2T_ADV_MAX_FILES" \
    ADV_SELECTION_STRATEGY="$A2T_ADV_SELECTION_STRATEGY" \
    ADV_SELECTION_SEED="$A2T_ADV_SELECTION_SEED" \
    ADV_PROB="$ADV_PROB" \
    NUM_GPUS="$NUM_GPUS" \
    NUM_TIMESTEPS="$A2T_TIMESTEPS" \
    STAGE_MODE=symlink \
    BASE_TEACHER_CKPT_DIR="$BASE_CKPT_REL" \
    BASE_TEACHER_ONNX_PATH="$BASE_ONNX_REL" \
    bash scripts/embodied/run_opentrack_physflow_adversarial.sh
  )

  NEW_ONNX="$(find ref_repo/OpenTrack/storage/logs/dagger -maxdepth 3 -path "*${ARM}_r${r}_dagger*/checkpoints/model.onnx" -print | sort | tail -1)"
  if [ -z "$NEW_ONNX" ] || [ ! -f "$NEW_ONNX" ]; then
    echo "[any2track-coevo] ERROR: missing updated Any2Track ONNX for round $r" >&2
    exit 4
  fi
  ANY_ONNX="$REPO/$NEW_ONNX"
  echo "[any2track-coevo] round=$r synced judge=$ANY_ONNX"
done

echo "[any2track-coevo] done $(date)"
