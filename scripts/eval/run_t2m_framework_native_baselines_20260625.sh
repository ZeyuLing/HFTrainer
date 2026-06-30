#!/usr/bin/env bash
# Framework-native rerun for the T2M baselines that previously used standalone
# wrappers. The three methods run sequentially to avoid CLEAN/race issues; each
# method shards internally over the available GPUs.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HFTRAINER_SKIP_AUTOREGISTER=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

BASE="outputs/evaluation/t2m/humanml3d_official_test"
STAGE_BASE="${STAGE_BASE:-$BASE/_framework_native_20260625}"
LOG_DIR="$STAGE_BASE/logs"
mkdir -p "$LOG_DIR"

TOTAL_SHARDS="${TOTAL_SHARDS:-${TJ_GPU_NUM:-6}}"
LOCAL_SHARDS="${LOCAL_SHARDS:-$TOTAL_SHARDS}"
NUM_GPUS="${NUM_GPUS:-$LOCAL_SHARDS}"
COMMON_ENV=(
  TOTAL_SHARDS="$TOTAL_SHARDS"
  LOCAL_SHARDS="$LOCAL_SHARDS"
  NUM_GPUS="$NUM_GPUS"
  CLEAN="${CLEAN:-1}"
  WORKERS="${WORKERS:-32}"
)

ensure_framework_deps() {
  local stamp="${DEPS_STAMP:-/tmp/hftrainer_framework_native_deps_v1.stamp}"
  if [[ -f "$stamp" ]]; then
    return 0
  fi
  local py_bin="${PY:-python3}"
  local missing
  missing="$("$py_bin" - <<'PY'
import importlib.util

checks = [
    ("mmengine", "mmengine>=0.10.0"),
    ("omegaconf", "omegaconf>=2.3"),
    ("tqdm", "tqdm"),
    ("einops", "einops>=0.7"),
    ("smplx", "smplx>=0.1.28"),
    ("chumpy", "chumpy>=0.70"),
    ("torchgeometry", "torchgeometry>=0.1.2"),
]
for module, package in checks:
    if importlib.util.find_spec(module) is None:
        print(package)
PY
)"
  if [[ -n "$missing" ]]; then
    echo "[deps] installing framework deps: $(tr '\n' ' ' <<<"$missing")"
    "$py_bin" -m pip install -q \
      -i https://mirrors.tencent.com/pypi/simple \
      --trusted-host mirrors.tencent.com \
      $missing
  else
    echo "[deps] framework deps importable"
  fi
  touch "$stamp"
}

run_flowmdm() {
  echo "[driver] flowmdm start $(date -Is)"
  env "${COMMON_ENV[@]}" \
    METHOD=flowmdm \
    RUN_TAG=flowmdm_framework_native_20260625 \
    HML_DIR="$STAGE_BASE/hml263/flowmdm" \
    M135_DIR="$STAGE_BASE/motion135/flowmdm" \
    MS272_DIR="$STAGE_BASE/ms272/flowmdm" \
    BATCH_SIZE=1 \
    FLOW_CHUNKED_ATT="${FLOW_CHUNKED_ATT:-1}" \
    bash scripts/eval/run_t2m_flowmdm_motionlab_clean_20260625.sh \
    2>&1 | tee "$LOG_DIR/flowmdm_driver.log"
  echo "[driver] flowmdm done $(date -Is)"
}

run_motionlab() {
  echo "[driver] motionlab start $(date -Is)"
  env "${COMMON_ENV[@]}" \
    METHOD=motionlab \
    RUN_TAG=motionlab_framework_native_20260625 \
    HML_DIR="$STAGE_BASE/hml263/motionlab" \
    M135_DIR="$STAGE_BASE/motion135/motionlab" \
    MS272_DIR="$STAGE_BASE/ms272/motionlab" \
    BATCH_SIZE="${MOTIONLAB_BATCH_SIZE:-16}" \
    MOTIONLAB_STAGE="${MOTIONLAB_STAGE:-demo}" \
    bash scripts/eval/run_t2m_flowmdm_motionlab_clean_20260625.sh \
    2>&1 | tee "$LOG_DIR/motionlab_driver.log"
  echo "[driver] motionlab done $(date -Is)"
}

run_motiongpt3() {
  echo "[driver] motiongpt3 start $(date -Is)"
  env "${COMMON_ENV[@]}" \
    RUN_TAG=motiongpt3_framework_native_20260625 \
    HML_DIR="$STAGE_BASE/hml263/motiongpt3" \
    M135_DIR="$STAGE_BASE/motion135/motiongpt3" \
    MS272_DIR="$STAGE_BASE/ms272/motiongpt3" \
    BATCH_SIZE="${MOTIONGPT3_BATCH_SIZE:-8}" \
    GUIDANCE_SCALE="${MOTIONGPT3_GUIDANCE_SCALE:-3.0}" \
    bash scripts/eval/run_t2m_motiongpt3_clean_20260624.sh \
    2>&1 | tee "$LOG_DIR/motiongpt3_driver.log"
  echo "[driver] motiongpt3 done $(date -Is)"
}

case "${METHODS:-flowmdm,motionlab,motiongpt3}" in
  *) ensure_framework_deps ;;
esac

case "${METHODS:-flowmdm,motionlab,motiongpt3}" in
  *flowmdm*) run_flowmdm ;;
esac
case "${METHODS:-flowmdm,motionlab,motiongpt3}" in
  *motionlab*) run_motionlab ;;
esac
case "${METHODS:-flowmdm,motionlab,motiongpt3}" in
  *motiongpt3*) run_motiongpt3 ;;
esac

echo "[driver] all requested methods done $(date -Is)"
