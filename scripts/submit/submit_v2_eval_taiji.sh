#!/bin/bash
# Submit M2M v2 evaluation jobs to Taiji (latest checkpoints, all tasks)
# Each model runs as a separate 1-GPU job.
#
# Usage:
#   bash tools/submit_v2_eval_taiji.sh           # All 4 v2 models + KIMODO
#   bash tools/submit_v2_eval_taiji.sh --dry-run  # Show commands only
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN="${1:-}"
TIMESTAMP=$(date +%Y%m%d)

echo "Project root: $PROJ_ROOT"
echo "Timestamp: $TIMESTAMP"
echo ""

submit_job() {
    local JOB_NAME="$1"
    local CMD="$2"

    echo "=== ${JOB_NAME} ==="
    echo "  CMD: ${CMD:0:120}..."

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        echo "  [DRY RUN] Skipped"
        return
    fi

    # Create a temp script for Taiji
    local SCRIPT="/tmp/taiji_eval_${JOB_NAME}.sh"
    cat > "$SCRIPT" << INNEREOF
#!/bin/bash
set -euo pipefail
cd ${PROJ_ROOT}
export PYTHONPATH=${PROJ_ROOT}:\${PYTHONPATH:-}
${CMD}
INNEREOF
    chmod +x "$SCRIPT"

    # Submit via taiji_client
    taiji_client start \
        -task_flag "${JOB_NAME}" \
        -business_flag "AILab_DHC_DD" \
        -host_num 1 \
        -gpu_num 1 \
        -GPUName V100 \
        -start_cmd "bash ${SCRIPT}" \
        2>&1 | head -5

    echo "  ✅ Submitted: ${JOB_NAME}"
}

# V2 Models: 4 main models, all tasks
for MODEL in caption_local caption_global uncond_local uncond_global; do
    submit_job "m2m_v2_eval_${MODEL}_${TIMESTAMP}" \
        "python3 tools/eval_m2m_v2_all_tasks.py --models ${MODEL} --all-tasks --max-samples 80 --num-steps 50 --replacement-guidance skip_last --text-guidance-scale 1.0 --save-npz --output-dir work_dirs/m2m_v2_eval_latest/${MODEL}"
done

# KIMODO: all comparable tasks
submit_job "m2m_v2_eval_kimodo_${TIMESTAMP}" \
    "python3 tools/run_kimodo_all_tasks.py --all-tasks --max-samples 80 --output-dir work_dirs/m2m_v2_eval_latest/kimodo"

echo ""
echo "=== All jobs submitted ==="
echo "Monitor: taiji_client trl"
echo "Results: work_dirs/m2m_v2_eval_latest/"
