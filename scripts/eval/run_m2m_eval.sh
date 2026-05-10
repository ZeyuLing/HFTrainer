#!/bin/bash
# Run M2M checkpoint evaluation on Taiji GPU node
# Usage: bash tools/run_m2m_eval.sh [--all]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

cd "$PROJ_ROOT"

EXTRA_ARGS=""
if [[ "${1:-}" == "--all" ]]; then
    EXTRA_ARGS="--all --test-replacement"
fi

# Phase 1: Evaluate 4 main models (well-trained)
echo "=== Phase 1: Main models (post-bugfix, >50 epochs) ==="
python3 tools/eval_m2m_checkpoints.py \
    --models uncond_fm caption_fm uncond_jit caption_jit \
    --max-samples 50 \
    --num-steps 50 \
    --test-replacement \
    --output-dir work_dirs/m2m_eval_report/phase1_main \
    $EXTRA_ARGS

# Phase 2: Evaluate MAN (mask-aware noise) variants
echo ""
echo "=== Phase 2: Mask-Aware Noise (V4) models ==="
python3 tools/eval_m2m_checkpoints.py \
    --models uncond_fm_man caption_fm_man \
    --max-samples 50 \
    --num-steps 50 \
    --test-replacement \
    --output-dir work_dirs/m2m_eval_report/phase2_man \
    $EXTRA_ARGS

echo ""
echo "=== Evaluation complete ==="
echo "Results: work_dirs/m2m_eval_report/"
