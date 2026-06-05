#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONUNBUFFERED=1

OUT_ROOT="${OUT_ROOT:-work_dirs/e9_lowq_expand_v3_full}"
DL_PATH="${DL_PATH:-data/eval/m2m_v2/eval_e9_repair_v3_expand.json}"
TARGET_PER_TYPE="${TARGET_PER_TYPE:-25}"
CANDIDATE_CAP="${CANDIDATE_CAP:-120}"
DASHBOARD_SETTING="${DASHBOARD_SETTING:-lowq_expand_v3}"
RUN_TAG="${RUN_TAG:-E9 low-quality expansion v3 full rerun on lzy_debug_machine_2, 2026-04-29}"

mkdir -p "$OUT_ROOT"

echo "[stage1] build datalist $(date '+%F %T')"
python3 tools/build_e9_repair_v2.py \
  --target-per-type "$TARGET_PER_TYPE" \
  --candidate-cap "$CANDIDATE_CAP" \
  --device cuda \
  --exclude-datalist data/eval/m2m_v2/eval_e9_repair_v2.json \
  --out "$DL_PATH" \
  > "$OUT_ROOT/build_e9_v3.log" 2>&1

echo "[stage2] compute MoGenDIT adaptive masks $(date '+%F %T')"
CUDA_VISIBLE_DEVICES="${MASK_GPU:-0}" \
python3 scripts/compute_adaptive_masks_for_eval.py \
  --eval-datalist "$DL_PATH" \
  --device cuda:0 \
  > "$OUT_ROOT/compute_adaptive_masks.log" 2>&1

echo "[stage3] run all models $(date '+%F %T')"
OUT="$OUT_ROOT/infer" \
DL="$DL_PATH" \
DASHBOARD_SETTING="$DASHBOARD_SETTING" \
RUN_TAG="$RUN_TAG" \
bash scripts/run_e9_all_models_variant.sh \
  > "$OUT_ROOT/run_all_models.log" 2>&1

echo "[done] $(date '+%F %T')"
