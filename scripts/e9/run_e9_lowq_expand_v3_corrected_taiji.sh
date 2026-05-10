#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONUNBUFFERED=1

ROOT_OUT="${ROOT_OUT:-work_dirs/e9_lowq_expand_v3_corrected}"
DL="${DL:-data/eval/m2m_v2/eval_e9_repair_v3_expand.json}"
DASHBOARD_SETTING="${DASHBOARD_SETTING:-lowq_expand_v3}"
RUN_TAG="${RUN_TAG:-E9 low-quality expansion v3 corrected adaptive-mask rerun on lzy_debug_machine_2, 2026-04-29}"

mkdir -p "$ROOT_OUT"

echo "[corrected] start $(date '+%F %T')"
echo "[corrected] datalist=$DL"

echo "[corrected] compute MoGenDIT adaptive masks $(date '+%F %T')"
CUDA_VISIBLE_DEVICES="${MASK_GPU:-0}" \
python3 scripts/compute_adaptive_masks_for_eval.py \
  --eval-datalist "$DL" \
  --device cuda:0 \
  > "$ROOT_OUT/compute_adaptive_masks.log" 2>&1

echo "[corrected] run full model set $(date '+%F %T')"
OUT="$ROOT_OUT/infer" \
DL="$DL" \
DASHBOARD_SETTING="$DASHBOARD_SETTING" \
RUN_TAG="$RUN_TAG" \
RUN_STABLE_CHAIN=0 \
bash scripts/run_e9_all_models_variant.sh \
  > "$ROOT_OUT/run_all_models.log" 2>&1

echo "[corrected] done $(date '+%F %T')"
