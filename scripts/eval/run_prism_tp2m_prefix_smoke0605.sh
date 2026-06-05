#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

COND_FRAMES="${COND_FRAMES:?set COND_FRAMES}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/evaluation/prism_tp2m_prefix_0605/h3d_smoke64_basecfg}"
CONFIG="${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}"
CHECKPOINT="${CHECKPOINT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_7}"

python3 scripts/eval/eval_prism_tp2m_prefix.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --condition-num-frames "${COND_FRAMES}" \
  --max-samples "${MAX_SAMPLES}" \
  --kafs-mode depth_driven \
  --skip-existing \
  --seed 42
