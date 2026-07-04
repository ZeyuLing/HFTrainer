#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TMR_ROOT="${TMR_ROOT:-${PROJECT_ROOT}/ref_repo/TMR}"
RUN_NAME="${RUN_NAME:-tmr_hymotion_g1_small_debug}"
REPRESENTATION="${REPRESENTATION:-g1_38d}"
INPUT_FORMAT="${INPUT_FORMAT:-g1}"
ANNO="${ANNO:-data/annotation/train_g1_t2m_emb_minus_heldout_scene_clean.json}"
SCENE_FILTER="${SCENE_FILTER:-hard}"
CAPTION_SOURCE="${CAPTION_SOURCE:-embedding}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_GPUS="${NUM_GPUS:-${TMRT_NUM_GPUS:-1}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/tmr_hymotion/${REPRESENTATION}/${RUN_NAME}}"
DATASET_DIR="${RUN_ROOT}/dataset"
STATS_DIR="${DATASET_DIR}/stats"
TMR_RUN_DIR="${RUN_ROOT}/checkpoints/tmr"
LOG_DIR="${RUN_ROOT}/logs"
METRICS_DIR="${RUN_ROOT}/metrics"
RETRIEVAL_DIR="${RUN_ROOT}/retrieval_cases"
DEVICE="${DEVICE:-cuda}"
REOCCUPY_AFTER="${REOCCUPY_AFTER:-0}"
BUILD_ONLY="${BUILD_ONLY:-0}"

mkdir -p "${LOG_DIR}" "${METRICS_DIR}" "${RETRIEVAL_DIR}" "${TMR_RUN_DIR}"

reoccupy() {
  if [[ "${REOCCUPY_AFTER}" == "1" && -f "${PROJECT_ROOT}/../occupy.py" ]]; then
    nohup python3 "${PROJECT_ROOT}/../occupy.py" --gpus all --mem-frac-of-free 0.50 \
      > "${LOG_DIR}/occupy_after.log" 2>&1 &
  fi
}
trap reoccupy EXIT

cd "${PROJECT_ROOT}"
echo "[HYMotion-TMR] project=${PROJECT_ROOT}"
echo "[HYMotion-TMR] run_root=${RUN_ROOT}"
echo "[HYMotion-TMR] representation=${REPRESENTATION} input_format=${INPUT_FORMAT}"

python3 scripts/embodied/build_hymotion_tmr_dataset.py \
  --anno "${ANNO}" \
  --out-root "${RUN_ROOT}" \
  --input-format "${INPUT_FORMAT}" \
  --representation "${REPRESENTATION}" \
  --scene-filter "${SCENE_FILTER}" \
  --caption-source "${CAPTION_SOURCE}" \
  --max-items "${MAX_ITEMS}" \
  2>&1 | tee "${LOG_DIR}/build_dataset.log"

if [[ "${BUILD_ONLY}" == "1" ]]; then
  echo "[HYMotion-TMR] BUILD_ONLY=1; skip text embeddings / train / retrieval"
  exit 0
fi

cd "${TMR_ROOT}"
export PYTHONPATH="${TMR_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

COMMON_OVERRIDES=(
  "data=humanml3d"
  "data.path=${DATASET_DIR}"
  "data.motion_loader.base_dir=${DATASET_DIR}/motions"
  "data.motion_loader.fps=30.0"
  "data.motion_loader.nfeats=$(python3 - <<PY
import json
print(json.load(open('${RUN_ROOT}/dataset_card.json'))['nfeats'])
PY
)"
  "data.motion_loader.normalizer.base_dir=${STATS_DIR}"
  "data.text_to_token_emb.path=${DATASET_DIR}"
  "data.text_to_sent_emb.path=${DATASET_DIR}"
)

if [[ ! -f "${DATASET_DIR}/token_embeddings/distilbert-base-uncased.npy" || \
      ! -f "${DATASET_DIR}/sent_embeddings/sentence-transformers/all-mpnet-base-v2.npy" ]]; then
  echo "[HYMotion-TMR] computing text embeddings"
  python3 prepare/text_embeddings.py \
    "${COMMON_OVERRIDES[@]}" \
    "device=${DEVICE}" \
    2>&1 | tee "${LOG_DIR}/text_embeddings.log"
else
  echo "[HYMotion-TMR] text embeddings already exist; skip"
fi

TRAIN_OVERRIDES=(
  "${COMMON_OVERRIDES[@]}"
  "run_dir=${TMR_RUN_DIR}"
  "data.preload=false"
  "dataloader.batch_size=${BATCH_SIZE}"
  "dataloader.num_workers=${NUM_WORKERS}"
  "trainer.max_epochs=${MAX_EPOCHS}"
  "trainer.devices=${NUM_GPUS}"
  "trainer.log_every_n_steps=25"
)
if [[ "${NUM_GPUS}" != "1" ]]; then
  TRAIN_OVERRIDES+=("++trainer.strategy=ddp")
fi

echo "[HYMotion-TMR] training TMR evaluator"
python3 train.py "${TRAIN_OVERRIDES[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

echo "[HYMotion-TMR] retrieval evaluation"
python3 retrieval.py \
  "run_dir=${TMR_RUN_DIR}" \
  "device=${DEVICE}" \
  "protocol=all" \
  "batch_size=256" \
  2>&1 | tee "${LOG_DIR}/retrieval.log"

if [[ -d "${TMR_RUN_DIR}/contrastive_metrics" ]]; then
  cp -a "${TMR_RUN_DIR}/contrastive_metrics/." "${METRICS_DIR}/"
fi
echo "[HYMotion-TMR] done: ${RUN_ROOT}"
