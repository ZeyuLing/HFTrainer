#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "$ROOT" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

BASE=${BASE:-output/evaluation/table3_recon_baselines_0607}
GT_DIR=${GT_DIR:-output/evaluation/table3_recon_baselines_0606/hml263_gt_1p_max12_min40/new_joint_vecs}
ID_LIST=${ID_LIST:-output/evaluation/table3_recon_baselines_0606/hml263_gt_1p_max12_min40/test.txt}
FILTERED_ID_LIST=${FILTERED_ID_LIST:-$BASE/hml263_gt_1p_max12_min40/test_quality_filtered.txt}
QUALITY_REPORT=${QUALITY_REPORT:-$BASE/hml263_gt_1p_max12_min40/quality_filter_report.json}
mkdir -p "$BASE/logs"

python3 scripts/eval/build_hml263_quality_filtered_ids.py \
  --gt-dir "$GT_DIR" \
  --ids "$ID_LIST" \
  --out-ids "$FILTERED_ID_LIST" \
  --out-json "$QUALITY_REPORT" \
  > "$BASE/logs/hml263_quality_filter.log" 2>&1

EVAL_ID_LIST=${EVAL_ID_LIST:-$FILTERED_ID_LIST}

run_one() {
  local method=$1
  local pred_dir="output/evaluation/table3_recon_baselines_0606/hml_tokenizer_recon_1p_min40/${method}/merged/new_joint_vecs"
  local out_json="$BASE/hml_tokenizer_recon_1p_min40/${method}/merged/native_hml263_recon_metrics.json"
  mkdir -p "$(dirname "$out_json")"
  python3 scripts/eval/eval_hml263_recon_metrics.py \
    --gt-dir "$GT_DIR" \
    --pred-dir "$pred_dir" \
    --ids "$EVAL_ID_LIST" \
    --out-json "$out_json" \
    --source-fps 20 \
    --target-fps 20 \
    --skip-bad-gt \
    --quality-report-json "$BASE/hml_tokenizer_recon_1p_min40/${method}/merged/quality_filter_report.json" \
    > "$BASE/logs/${method}_native_hml263.log" 2>&1
}

echo "[hml263-native-recon] start $(date -Is)"
run_one t2mgpt &
pid_a=$!
run_one momask &
pid_b=$!
wait "$pid_a"
wait "$pid_b"
echo "[hml263-native-recon] done $(date -Is)"
