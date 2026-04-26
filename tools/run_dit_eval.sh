#!/bin/bash
# Run DiT (text-free) checkpoint evaluation
# Usage: bash tools/run_dit_eval.sh [--variant fm_man|fm_man_globalrot|jit_man|jit_man_globalrot] [--size s|m|b|l]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

cd "$PROJ_ROOT"

VARIANT=""
SIZE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --size) SIZE="$2"; shift 2 ;;
        --test-replacement) EXTRA_ARGS="$EXTRA_ARGS --test-replacement"; shift ;;
        --max-samples) EXTRA_ARGS="$EXTRA_ARGS --max-samples $2"; shift 2 ;;
        --num-steps) EXTRA_ARGS="$EXTRA_ARGS --num-steps $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Build model list
ALL_VARIANTS=(fm_man fm_man_globalrot jit_man jit_man_globalrot)
ALL_SIZES=(s m b l)

if [[ -n "$VARIANT" ]]; then
    ALL_VARIANTS=("$VARIANT")
fi
if [[ -n "$SIZE" ]]; then
    ALL_SIZES=("$SIZE")
fi

MODELS=()
for v in "${ALL_VARIANTS[@]}"; do
    for sz in "${ALL_SIZES[@]}"; do
        MODELS+=("dit_${v}_${sz}")
    done
done

MODEL_LIST="${MODELS[*]}"
echo "=== DiT Evaluation ==="
echo "Models: ${MODEL_LIST}"
echo ""

# Phase 1: FM variants (local rotation)
FM_MODELS=()
for m in "${MODELS[@]}"; do
    if [[ "$m" == dit_fm_man_* ]] && [[ "$m" != *globalrot* ]]; then
        FM_MODELS+=("$m")
    fi
done
if [[ ${#FM_MODELS[@]} -gt 0 ]]; then
    echo "--- Phase 1: FM + MAN (local rot) ---"
    python3 tools/eval_m2m_checkpoints.py \
        --models ${FM_MODELS[*]} \
        --max-samples 50 \
        --num-steps 50 \
        --test-replacement \
        --output-dir work_dirs/dit_eval_report/fm_man \
        $EXTRA_ARGS
fi

# Phase 2: FM + GlobalRot variants
FM_GR_MODELS=()
for m in "${MODELS[@]}"; do
    if [[ "$m" == dit_fm_man_globalrot_* ]]; then
        FM_GR_MODELS+=("$m")
    fi
done
if [[ ${#FM_GR_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo "--- Phase 2: FM + MAN + GlobalRot ---"
    python3 tools/eval_m2m_checkpoints.py \
        --models ${FM_GR_MODELS[*]} \
        --max-samples 50 \
        --num-steps 50 \
        --test-replacement \
        --output-dir work_dirs/dit_eval_report/fm_man_globalrot \
        $EXTRA_ARGS
fi

# Phase 3: JiT variants (local rotation)
JIT_MODELS=()
for m in "${MODELS[@]}"; do
    if [[ "$m" == dit_jit_man_* ]] && [[ "$m" != *globalrot* ]]; then
        JIT_MODELS+=("$m")
    fi
done
if [[ ${#JIT_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo "--- Phase 3: JiT + MAN (local rot) ---"
    python3 tools/eval_m2m_checkpoints.py \
        --models ${JIT_MODELS[*]} \
        --max-samples 50 \
        --num-steps 50 \
        --test-replacement \
        --output-dir work_dirs/dit_eval_report/jit_man \
        $EXTRA_ARGS
fi

# Phase 4: JiT + GlobalRot variants
JIT_GR_MODELS=()
for m in "${MODELS[@]}"; do
    if [[ "$m" == dit_jit_man_globalrot_* ]]; then
        JIT_GR_MODELS+=("$m")
    fi
done
if [[ ${#JIT_GR_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo "--- Phase 4: JiT + MAN + GlobalRot ---"
    python3 tools/eval_m2m_checkpoints.py \
        --models ${JIT_GR_MODELS[*]} \
        --max-samples 50 \
        --num-steps 50 \
        --test-replacement \
        --output-dir work_dirs/dit_eval_report/jit_man_globalrot \
        $EXTRA_ARGS
fi

echo ""
echo "=== DiT Evaluation complete ==="
echo "Results: work_dirs/dit_eval_report/"
