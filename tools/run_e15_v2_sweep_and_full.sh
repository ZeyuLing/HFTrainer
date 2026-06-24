#!/bin/bash
# E15 v2 (2026-04-27) sweep + full evaluation runner.
#
# Layout (8 V100/A100 GPUs assumed):
#   GPU 0  uncond_local  default          200 samples  (full)
#   GPU 1  uncond_local  sweep_fast        30 samples  (ablation)
#   GPU 2  uncond_local  sweep_slow        30 samples  (ablation)
#   GPU 3  uncond_local  sweep_ncond5      30 samples  (ablation)
#   GPU 4  uncond_local  sweep_ncond60     30 samples  (ablation)
#   GPU 5  uncond_global default          200 samples  (full)
#   GPU 6  KIMODO        default          200 samples  (full)
#   GPU 7  free
#
# All runs use --save-npz so the eval-dashboard's 3D viewer works.
# Sweep N (--max-samples 30) is held fixed across the four sweep
# settings so the comparison is apples-to-apples; the winner gets
# locked in `_DATALIST_FILES`-aligned `default` settings.
#
# Outputs land under work_dirs/e15_v2_eval_${timestamp}/ and are
# imported into eval_dashboard.db at the end.
#
# Usage:
#   bash tools/run_e15_v2_sweep_and_full.sh [--skip-sweep] [--skip-full]
#                                            [--skip-kimodo] [--full-n 200]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"
cd "$PROJ_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="work_dirs/e15_v2_eval_${TIMESTAMP}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

SKIP_SWEEP=0
SKIP_FULL=0
SKIP_KIMODO=0
FULL_N=200
SWEEP_N=30
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-sweep)  SKIP_SWEEP=1; shift ;;
        --skip-full)   SKIP_FULL=1;  shift ;;
        --skip-kimodo) SKIP_KIMODO=1; shift ;;
        --full-n)      FULL_N="$2"; shift 2 ;;
        --sweep-n)     SWEEP_N="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

echo "=========================================================="
echo " E15 v2 sweep + full evaluation"
echo " Output:    ${OUT_ROOT}"
echo " Full N:    ${FULL_N}    Sweep N: ${SWEEP_N}"
echo " skip:      sweep=${SKIP_SWEEP} full=${SKIP_FULL} kimodo=${SKIP_KIMODO}"
echo " Started:   $(date)"
echo "=========================================================="

# ── helper: launch one M2M v2 run on a specific GPU ────────────────
launch_m2m() {
    local gpu="$1"
    local model="$2"
    local setting="$3"
    local n="$4"
    local tag="${model}__${setting}"
    local out_dir="${OUT_ROOT}/${tag}"
    local log="${LOG_DIR}/${tag}.log"
    mkdir -p "${out_dir}"
    echo "  [GPU ${gpu}] launching M2M ${model}/${setting} (n=${n}) -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        nohup python3 tools/eval_m2m_v2_all_tasks.py \
            --models "${model}" \
            --tasks E15 \
            --settings "${setting}" \
            --max-samples "${n}" \
            --num-steps 50 \
            --replacement-guidance skip_last \
            --text-guidance-scale 5.0 \
            --save-npz \
            --output-dir "${out_dir}" \
            --device cuda \
            > "${log}" 2>&1 &
    echo "    PID=$!  CUDA_VISIBLE_DEVICES=${gpu}"
}

launch_kimodo() {
    local gpu="$1"
    local n="$2"
    local out_dir="${OUT_ROOT}/kimodo__default"
    local log="${LOG_DIR}/kimodo__default.log"
    mkdir -p "${out_dir}"
    echo "  [GPU ${gpu}] launching KIMODO E15/default (n=${n}) -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        nohup python3 tools/run_kimodo_all_tasks.py \
            --tasks E15 \
            --settings default \
            --max-samples "${n}" \
            --output-dir "${out_dir}" \
            > "${log}" 2>&1 &
    echo "    PID=$!  CUDA_VISIBLE_DEVICES=${gpu}"
}

# ── Full runs (default setting, 200 samples each) ──────────────────
if [[ "${SKIP_FULL}" -eq 0 ]]; then
    launch_m2m 0 uncond_local  default "${FULL_N}"
    launch_m2m 5 uncond_global default "${FULL_N}"
fi
if [[ "${SKIP_KIMODO}" -eq 0 ]]; then
    launch_kimodo 6 "${FULL_N}"
fi

# ── Sweep runs (uncond_local only, 30 samples each) ────────────────
if [[ "${SKIP_SWEEP}" -eq 0 ]]; then
    launch_m2m 1 uncond_local sweep_fast    "${SWEEP_N}"
    launch_m2m 2 uncond_local sweep_slow    "${SWEEP_N}"
    launch_m2m 3 uncond_local sweep_ncond5  "${SWEEP_N}"
    launch_m2m 4 uncond_local sweep_ncond60 "${SWEEP_N}"
fi

echo ""
echo "All jobs dispatched. Waiting for completion..."
echo "Tail any log via:  tail -f ${LOG_DIR}/<tag>.log"
echo ""

wait

echo ""
echo "=========================================================="
echo " All E15 v2 runs finished at $(date)"
echo "=========================================================="
echo ""
echo "Per-tag exit and finish times:"
for log in "${LOG_DIR}"/*.log; do
    tag=$(basename "$log" .log)
    last=$(tail -3 "$log" 2>/dev/null | tr -d '\r' | head -3 | tr '\n' ' | ')
    echo "  [${tag}]  ${last}"
done

echo ""
echo "=========================================================="
echo " Splitting nested -> flat JSONs..."
echo "=========================================================="
for sub in "${OUT_ROOT}"/*/; do
    if ls "${sub}eval_v2_"*.json >/dev/null 2>&1; then
        python3 tools/split_eval_v2_to_flat.py \
            --in-dir "${sub}" \
            --out-dir "${sub}/import_jsons" \
            --timestamp "$(date '+%Y-%m-%d %H:%M:%S')" \
            >> "${LOG_DIR}/_import.log" 2>&1 || true
    fi
done

echo ""
echo "=========================================================="
echo " Importing flat JSONs into eval_dashboard.db..."
echo "=========================================================="
DB="motion_annot_web/eval_dashboard/eval_dashboard.db"
cp "${DB}" "${DB}.bak_before_e15_v2_import_${TIMESTAMP}" || true
for j in "${OUT_ROOT}"/*/import_jsons/*.json; do
    [[ -e "$j" ]] || continue
    python3 motion_annot_web/eval_dashboard/data_importer.py import "$j" \
        --notes "E15-v2 sweep+full ${TIMESTAMP}" \
        >> "${LOG_DIR}/_import.log" 2>&1 || true
done

# KIMODO results land in <out>/kimodo__default/<task_setting>/result.json
# (flat already; data_importer accepts that shape via the same CLI).
for j in "${OUT_ROOT}"/kimodo__*/*/result.json; do
    [[ -e "$j" ]] || continue
    python3 motion_annot_web/eval_dashboard/data_importer.py import "$j" \
        --notes "E15-v2 KIMODO ${TIMESTAMP}" \
        >> "${LOG_DIR}/_import.log" 2>&1 || true
done

echo ""
echo "Done. Dashboard import log: ${LOG_DIR}/_import.log"
echo "Summary: bash tools/run_e15_v2_sweep_and_full.sh > ${OUT_ROOT}/_outer.log 2>&1"
