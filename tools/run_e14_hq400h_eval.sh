#!/bin/bash
# E14 Transition Stitching eval on the HQ400h test set.
#
# - Test data: data/eval/m2m_v2/eval_e14_hq400h_{static,move}100.json (L/M)
# - Models: uncond_local + uncond_global
#   (E14 is NOT caption_aware, caption models auto-skipped by the runner.)
# - N per setting: 100
# - Requires CUDA. By default uses GPU 0 and 1 for the two models.
#
# Outputs:
#   $OUT_ROOT/ul_e14/uncond_local/E14_{L,M}/npz/*.npz    + eval_v2_*.json
#   $OUT_ROOT/ug_e14/uncond_global/E14_{L,M}/npz/*.npz   + eval_v2_*.json
#
# Usage:
#   bash tools/run_e14_hq400h_eval.sh              # parallel (GPU 0 + 1)
#   SERIAL=1 bash tools/run_e14_hq400h_eval.sh     # serial on GPU 0

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="work_dirs/e14_hq400h_${TS}"
mkdir -p "$OUT_ROOT"
echo "Output dir: $OUT_ROOT"

MAX_SAMPLES="${MAX_SAMPLES:-100}"  # default 100 to cover the whole HQ400h test set

run_model () {
    local GPU=$1
    local MODEL=$2
    local TAG=$3
    CUDA_VISIBLE_DEVICES=$GPU python3 tools/eval_m2m_v2_all_tasks.py \
        --models "$MODEL" --tasks E14 \
        --max-samples "$MAX_SAMPLES" \
        --save-npz \
        --output-dir "$OUT_ROOT/$TAG" \
        2>&1 | tee "$OUT_ROOT/${TAG}.log"
}

if [[ "${SERIAL:-0}" == "1" ]]; then
    echo "[serial] uncond_local on GPU 0 ..."
    run_model 0 uncond_local ul_e14
    echo "[serial] uncond_global on GPU 0 ..."
    run_model 0 uncond_global ug_e14
else
    echo "[parallel] uncond_local on GPU 0 + uncond_global on GPU 1 ..."
    run_model 0 uncond_local ul_e14 &
    PID_UL=$!
    run_model 1 uncond_global ug_e14 &
    PID_UG=$!
    wait $PID_UL $PID_UG
fi

echo ""
echo "===================================================================="
echo "INFERENCE DONE. Outputs in: $OUT_ROOT"
echo "===================================================================="
echo ""
echo "Next steps (run manually after confirming logs look OK):"
echo ""
echo "# 1. Split nested eval JSON -> flat per-(model,task,setting) JSON"
echo "python3 tools/split_eval_v2_to_flat.py \\"
echo "    --in-dir \"$OUT_ROOT\" \\"
echo "    --out-dir \"$OUT_ROOT/import_jsons\" \\"
echo "    --timestamp \"$(date '+%Y-%m-%d %H:%M:%S')\""
echo ""
echo "# 2. Backup DB"
echo "cp motion_annot_web/eval_dashboard/eval_dashboard.db \\"
echo "   motion_annot_web/eval_dashboard/eval_dashboard.db.bak_before_e14_hq400h_import_\$(date +%Y%m%d_%H%M%S)"
echo ""
echo "# 3. Import each flat JSON"
echo "for j in $OUT_ROOT/import_jsons/*.json; do"
echo "  python3 motion_annot_web/eval_dashboard/data_importer.py import \"\$j\" \\"
echo "      --notes 'E14 hq400h 100-pair rerun ${TS}'"
echo "done"
echo ""
echo "# 4. Verify"
echo "sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db \\"
echo "    \"SELECT m.name, r.setting, r.num_samples, r.timestamp FROM eval_runs r JOIN models m ON r.model_id=m.id WHERE r.task_id='E14' ORDER BY r.timestamp DESC;\""
