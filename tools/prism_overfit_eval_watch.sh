#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-work_dirs/prism_overfit100_kt_projected_t5cached_nofp16}"
SAMPLES="${2:-3}"
STEPS="${3:-50}"
GPU="${4:-0}"
CONFIG="${5:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_overfit100.py}"
EXPORT_POSITIONS="${6:-0}"
WATCH_SECONDS="${7:-300}"
DECODE_FRAMES="${8:-0}"

WATCH_DIR="${ROOT}/eval_watch"
mkdir -p "${WATCH_DIR}"

while true; do
  ckpt="$(find "${ROOT}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1 || true)"
  if [[ -n "${ckpt}" ]]; then
    name="$(basename "${ckpt}")"
    done_file="${WATCH_DIR}/${name}.done"
    failed_file="${WATCH_DIR}/${name}.failed"
    log_file="${WATCH_DIR}/${name}_${SAMPLES}x${STEPS}.log"
    out_file="${WATCH_DIR}/${name}_${SAMPLES}x${STEPS}.json"
    if [[ ! -s "${ckpt}/model.pt" || ! -s "${ckpt}/meta.pt" ]]; then
      echo "[$(date '+%F %T')] waiting for complete checkpoint ${ckpt}" | tee -a "${WATCH_DIR}/watch.log"
      sleep "${WATCH_SECONDS}"
      continue
    fi
    if [[ ! -f "${done_file}" ]]; then
      echo "[$(date '+%F %T')] evaluating ${ckpt}" | tee -a "${WATCH_DIR}/watch.log"
      extra_args=()
      positions_dir=""
      if [[ "${EXPORT_POSITIONS}" == "1" ]]; then
        positions_dir="${ROOT}/viewer_${name}_${SAMPLES}x${STEPS}"
        extra_args+=(--positions-dir "${positions_dir}")
      fi
      if [[ "${DECODE_FRAMES}" != "0" ]]; then
        extra_args+=(--decode-frames "${DECODE_FRAMES}")
      fi
      if CUDA_VISIBLE_DEVICES="${GPU}" python3 tools/eval_prism_overfit_cached_t5.py \
        --config "${CONFIG}" \
        --checkpoint "${ckpt}" \
        --num-samples "${SAMPLES}" \
        --batch-size 1 \
        --num-steps "${STEPS}" \
        --device cuda \
        --output "${out_file}" \
        "${extra_args[@]}" \
        > "${log_file}" 2>&1; then
        if [[ -n "${positions_dir}" ]]; then
          ln -sfn "$(basename "${positions_dir}")" "${ROOT}/viewer_latest_${SAMPLES}x${STEPS}"
        fi
        rm -f "${failed_file}"
        touch "${done_file}"
        echo "[$(date '+%F %T')] done ${ckpt}" | tee -a "${WATCH_DIR}/watch.log"
      else
        touch "${failed_file}"
        echo "[$(date '+%F %T')] failed ${ckpt}; see ${log_file}" | tee -a "${WATCH_DIR}/watch.log"
      fi
    fi
  fi
  sleep "${WATCH_SECONDS}"
done
