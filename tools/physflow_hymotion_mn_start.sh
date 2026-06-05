#!/usr/bin/env bash
# Bootstrap HYMotion PhysFlow training on Taiji, then launch distributed train.
#
# This wraps tools/physflow_mn_start.sh with a one-time text-feature precompute
# gate. Rank 0 shards the HYMotion corpus over local GPUs and writes one merged
# manifest; other nodes wait for the marker before entering accelerate.
set -eo pipefail

CONFIG="${1:?usage: physflow_hymotion_mn_start.sh <config.py> [extra train.py args...]}"
shift || true

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PATH=/usr/local/bin:$PATH
export HF_HOME="$PWD/checkpoints/kimodo"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODERS_DIR="$PWD/checkpoints/kimodo/text_encoders"
export PHYSFLOW_CONVERT_PYTHON=python3
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

CORPUS="${PHYSFLOW_TEXT_CORPUS:-configs/experiments/physflow_kimodo_g1/physflow_text_hymotion_g1_real_train.jsonl}"
TEXT_NS="${PHYSFLOW_TEXT_NS:-kimodo_g1_llm2vec_hymotion_real_train}"
FEATURE_ROOT="${PHYSFLOW_FEATURE_ROOT:-data/kimodo_text_feature}"
FEATURE_DIR="${FEATURE_ROOT}/${TEXT_NS}"
MARKER="${FEATURE_DIR}/.physflow_hymotion_features.ready"
N_SHARDS="${PHYSFLOW_TEXT_NSHARDS:-8}"
BATCH_SIZE="${PHYSFLOW_TEXT_BATCH_SIZE:-16}"

python3 -c "import mujoco, onnxruntime, dm_control, typer" 2>/dev/null || {
  echo "[hymotion-mn] installing mujoco onnxruntime dm_control typer ..."
  python3 -m pip install --quiet mujoco onnxruntime dm_control typer 2>&1 | tail -3 | sed 's/^/[hymotion-mn] pip /'
}

node_rank="${INDEX:-0}"
if [[ "${node_rank}" == "0" ]]; then
  mkdir -p "${FEATURE_DIR}"
  if [[ ! -s "${MARKER}" ]]; then
    echo "[hymotion-mn] precomputing HYMotion text features: corpus=${CORPUS} ns=${TEXT_NS} shards=${N_SHARDS}"
    rm -f "${FEATURE_DIR}"/manifest.shard*.jsonl "${FEATURE_DIR}"/extract.shard*.log "${FEATURE_DIR}/manifest.jsonl.tmp"
    pids=()
    for ((shard = 0; shard < N_SHARDS; shard++)); do
      gpu=$((shard % 8))
      log="${FEATURE_DIR}/extract.shard${shard}.log"
      CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/embodied/cursor_extract_kimodo_text_feature.py \
        --corpus "${CORPUS}" \
        --namespace "${TEXT_NS}" \
        --cache-dir "${FEATURE_ROOT}" \
        --text-encoder llm2vec \
        --device cuda \
        --batch-size "${BATCH_SIZE}" \
        --num-shards "${N_SHARDS}" \
        --shard-index "${shard}" \
        --manifest-name "manifest.shard${shard}.jsonl" \
        > "${log}" 2>&1 &
      pids+=("$!")
      echo "[hymotion-mn] shard=${shard} gpu=${gpu} pid=${pids[-1]} log=${log}"
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    tmp="${FEATURE_DIR}/manifest.jsonl.tmp"
    : > "${tmp}"
    for ((shard = 0; shard < N_SHARDS; shard++)); do
      cat "${FEATURE_DIR}/manifest.shard${shard}.jsonl" >> "${tmp}"
    done
    mv "${tmp}" "${FEATURE_DIR}/manifest.jsonl"
    entries=$(wc -l < "${FEATURE_DIR}/manifest.jsonl")
    echo "[hymotion-mn] merged manifest entries=${entries} path=${FEATURE_DIR}/manifest.jsonl"
    date > "${MARKER}"
    echo "[hymotion-mn] feature cache ready: ${MARKER}"
  else
    echo "[hymotion-mn] feature cache already ready: ${MARKER}"
  fi
else
  echo "[hymotion-mn] waiting for feature cache marker on node rank ${node_rank}: ${MARKER}"
  while [[ ! -s "${MARKER}" ]]; do
    sleep 60
  done
fi

exec bash tools/physflow_mn_start.sh "${CONFIG}" "$@"
