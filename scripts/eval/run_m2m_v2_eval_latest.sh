#!/bin/bash
# Run M2M v2 all-tasks evaluation with latest checkpoints.
# This script is designed to run on a Taiji GPU node (1x V100/A100).
#
# Usage:
#   bash tools/run_m2m_v2_eval_latest.sh               # 4 main v2 models, all tasks
#   bash tools/run_m2m_v2_eval_latest.sh --models caption_local caption_global  # specific models
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

cd "$PROJ_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="work_dirs/m2m_v2_eval_latest_${TIMESTAMP}"

echo "============================================================"
echo " M2M v2 All-Tasks Evaluation (Latest Checkpoints)"
echo " Output: ${OUTPUT_DIR}"
echo " Start:  $(date)"
echo "============================================================"

# Parse optional --models argument, default to all 4 v2 models
MODELS="${@:---models caption_local caption_global uncond_local uncond_global}"

python3 tools/eval_m2m_v2_all_tasks.py \
    ${MODELS} \
    --all-tasks \
    --max-samples 80 \
    --num-steps 50 \
    --replacement-guidance skip_last \
    --text-guidance-scale 1.0 \
    --save-npz \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda

echo ""
echo "============================================================"
echo " Evaluation complete at $(date)"
echo " Results: ${OUTPUT_DIR}"
echo "============================================================"

# Auto-import results to eval dashboard
echo ""
echo "=== Importing results to eval dashboard ==="
python3 -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'motion_annot_web/eval_dashboard')
from db_manager import EvalDashboardDB
from data_importer import import_directory
db = EvalDashboardDB('motion_annot_web/eval_dashboard/eval_dashboard.db')
results = import_directory(db, '${OUTPUT_DIR}')
ok = sum(1 for r in results if r.get('status') == 'ok')
print(f'Imported {ok}/{len(results)} result files')
"
