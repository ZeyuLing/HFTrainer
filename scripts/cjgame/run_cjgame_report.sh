#!/bin/bash
# Generate the final evaluation report after all repair jobs complete.
# Run on any machine (no GPU needed, but needs torch for quality checker).
# Usage: On debug machine, after both eval jobs complete:
#   cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
#   bash scripts/run_cjgame_report.sh

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

echo "======================================"
echo "CJGame Eval - Report Generation"
echo "======================================"

# Count completed repairs per model
echo "Repair results summary:"
for d in output/cjgame_repair_eval/*/repaired; do
    model=$(basename $(dirname "$d"))
    count=$(ls "$d"/*.npz 2>/dev/null | wc -l)
    echo "  $model: $count repaired files"
done

# Run report generation only (skips repair, loads masks and discovers repaired files from disk)
python3 scripts/eval_cjgame_repair.py \
    --max-samples 0 \
    --device cpu \
    --output-dir output/cjgame_repair_eval \
    --skip-mogendit \
    --skip-m2m \
    --seed 42 \
    2>&1 | tee output/cjgame_repair_eval/log_report.txt

echo "======================================"
echo "Report generated at: output/cjgame_repair_eval/eval_report.json"
echo ""
echo "To start the web viewer:"
echo "  cd motion_annot_web/cjgame_repair_eval && python3 app.py --port 8083"
