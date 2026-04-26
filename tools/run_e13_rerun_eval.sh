#!/bin/bash
# E13 Multi-Prompt Autoregressive eval rerun.
#
# - Context: 2026-04-24 fix (2 parts). Symptom: "每段动作的尾帧几乎不动、静止".
#   Root cause was in HyMotionM2MPipeline._inference: `keep_mask = src_mask<0.5`
#   treated padding frames (idx >= tgt_length) as a known-condition region and
#   every ODE step pinned x[pad] ← x_clean[pad]. For E13 x_clean[pad] is the
#   normalize(synthetic zero) ≈ training mean pose. Even though attention was
#   key-masked on pad, the mean-pose anchor leaked into the tail of the valid
#   region through shared LayerNorms / residual paths.
#   Fixes:
#   (a) hftrainer/pipelines/motion/hymotion_m2m_pipeline.py:
#       keep_mask = (src_mask<0.5) & tgt_padding_mask  → pad excluded from
#       y0 init & replacement, matching training (where pad is never seen).
#   (b) tools/eval_m2m_v2_all_tasks.py::_run_one_segment:
#       pad motion_norm/src_mask back to 0 (training-dist) — the previous
#       workaround (pad src_mask=1) is no longer needed.
#
# - Task: E13 (settings A/B/C = 4 prompts, overlap_frames=1/10/30).
# - Models: caption_local + caption_global (only caption-aware models are
#   eligible for E13; previous run on 2026-04-22 used exactly this pair).
# - Datalist: data/eval/m2m_v2/eval_e13_multi_prompt.json (unchanged).
# - N per setting: 80 by default (matches prior E13 runs for direct compare).
#   Override via MAX_SAMPLES=100 ./tools/run_e13_rerun_eval.sh.
# - Requires CUDA. Parallel by default (caption_local on GPU 0, caption_global
#   on GPU 1). SERIAL=1 runs both on GPU 0 sequentially.
#
# Outputs:
#   $OUT_ROOT/cl_e13/caption_local/E13_{A,B,C}/npz/*.npz   + eval_v2_*.json
#   $OUT_ROOT/cg_e13/caption_global/E13_{A,B,C}/npz/*.npz  + eval_v2_*.json
#
# Usage:
#   bash tools/run_e13_rerun_eval.sh                         # parallel
#   SERIAL=1 bash tools/run_e13_rerun_eval.sh                # serial on GPU 0
#   MAX_SAMPLES=100 bash tools/run_e13_rerun_eval.sh         # larger sample
#   MODELS="caption_local" bash tools/run_e13_rerun_eval.sh  # one model only

set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

TS=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="work_dirs/e13_rerun_${TS}"
mkdir -p "$OUT_ROOT"
echo "Output dir: $OUT_ROOT"

MAX_SAMPLES="${MAX_SAMPLES:-80}"
MODELS_STR="${MODELS:-caption_local caption_global}"
read -ra MODELS_ARR <<< "$MODELS_STR"

run_model () {
    local GPU=$1
    local MODEL=$2
    local TAG=$3
    echo "[run_model] GPU=$GPU MODEL=$MODEL TAG=$TAG max_samples=$MAX_SAMPLES"
    CUDA_VISIBLE_DEVICES=$GPU python3 tools/eval_m2m_v2_all_tasks.py \
        --models "$MODEL" --tasks E13 \
        --max-samples "$MAX_SAMPLES" \
        --save-npz \
        --output-dir "$OUT_ROOT/$TAG" \
        2>&1 | tee "$OUT_ROOT/${TAG}.log"
}

# Build a (tag, model) list following MODELS order.
TAGS=()
for m in "${MODELS_ARR[@]}"; do
    case "$m" in
        caption_local)  TAGS+=("cl_e13") ;;
        caption_global) TAGS+=("cg_e13") ;;
        *)              TAGS+=("${m}_e13") ;;
    esac
done

if [[ "${SERIAL:-0}" == "1" || "${#MODELS_ARR[@]}" -eq 1 ]]; then
    for i in "${!MODELS_ARR[@]}"; do
        echo "[serial] ${MODELS_ARR[$i]} on GPU 0 ..."
        run_model 0 "${MODELS_ARR[$i]}" "${TAGS[$i]}"
    done
else
    # parallel: first model on GPU 0, second on GPU 1, etc.
    PIDS=()
    for i in "${!MODELS_ARR[@]}"; do
        GPU=$i
        echo "[parallel] ${MODELS_ARR[$i]} on GPU $GPU ..."
        run_model $GPU "${MODELS_ARR[$i]}" "${TAGS[$i]}" &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
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
echo "    --timestamp \"\$(date '+%Y-%m-%d %H:%M:%S')\""
echo ""
echo "# 2. Backup DB and delete the stale 2026-04-22 E13 runs first"
echo "cp motion_annot_web/eval_dashboard/eval_dashboard.db \\"
echo "   motion_annot_web/eval_dashboard/eval_dashboard.db.bak_before_e13_rerun_\$(date +%Y%m%d_%H%M%S)"
echo ""
echo "# (Optional: clean old E13 runs to avoid two versions in the UI)"
echo "sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db \\"
echo "  \"SELECT id FROM eval_runs WHERE task_id='E13';\" | while read rid; do"
echo "    sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db \\"
echo "      \"DELETE FROM sample_results WHERE eval_run_id=\$rid;"
echo "       DELETE FROM agg_metrics  WHERE eval_run_id=\$rid;"
echo "       DELETE FROM eval_runs    WHERE id=\$rid;\""
echo "done"
echo ""
echo "# 3. Import each flat JSON"
echo "for j in $OUT_ROOT/import_jsons/*.json; do"
echo "  python3 motion_annot_web/eval_dashboard/data_importer.py import \"\$j\" \\"
echo "      --notes 'E13 rerun ${TS} (post pad-mask fix)'"
echo "done"
echo ""
echo "# 4. Verify"
echo "sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db \\"
echo "    \"SELECT m.name, r.setting, r.num_samples, r.timestamp FROM eval_runs r JOIN models m ON r.model_id=m.id WHERE r.task_id='E13' ORDER BY r.timestamp DESC;\""
