#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONUNBUFFERED=1
export MOGENDIT_ROOT="${MOGENDIT_ROOT:-$PWD/ref_repo/MoGenDiT}"
export MOGENDIT_CKPT_ROOT="${MOGENDIT_CKPT_ROOT:-/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/save/ckpt}"

OUT="${OUT:-work_dirs/e9_lowq_expand_v3_full}"
DL="${DL:-data/eval/m2m_v2/eval_e9_repair_v3_expand.json}"
DB="${DB:-motion_annot_web/eval_dashboard/eval_dashboard.db}"
DASHBOARD_SETTING="${DASHBOARD_SETTING:-lowq_expand_v3}"
RUN_TAG="${RUN_TAG:-E9 low-quality expansion v3 full rerun}"
RUN_STABLE_CHAIN="${RUN_STABLE_CHAIN:-0}"
LOG_DIR="$OUT/logs"
DL_BASENAME="$(basename "$DL")"

export DB DASHBOARD_SETTING

if [[ "${CLEAN_OUT:-1}" == "1" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$LOG_DIR" "$OUT/renamed_import_jsons"

echo "[runner] start $(date '+%F %T')"
echo "[runner] host=$(hostname) pwd=$PWD out=$OUT"
echo "[runner] datalist=$DL dashboard_setting=$DASHBOARD_SETTING"
nvidia-smi || true

run_stablemotion() {
  echo "[stablemotion] start $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_stablemotion_e9.py \
    --eval-datalist "$DL" \
    --output-dir "$OUT/stablemotion" \
    --max-samples 99999 \
    --device cuda \
    --ensemble \
    --enable-sits \
    --skip-timesteps 40 \
    > "$LOG_DIR/stablemotion_infer.log" 2>&1

  CUDA_VISIBLE_DEVICES=0 python3 scripts/stablemotion_to_dashboard.py \
    --src "$OUT/stablemotion" \
    --out-dir "$OUT/stablemotion" \
    --eval-datalist "$DL" \
    --model-name StableMotion \
    --setting default \
    --device cuda \
    > "$LOG_DIR/stablemotion_import_json.log" 2>&1

  if [[ "$RUN_STABLE_CHAIN" == "1" ]]; then
    CUDA_VISIBLE_DEVICES=5 python3 scripts/postprocess_hymotion_with_mogendit.py \
      --src "$OUT/stablemotion/stablemotion/E9_StableMotion" \
      --out-dir "$OUT/stablemotion_mogendit_chain" \
      --eval-datalist "$DL" \
      --model-name stablemotion \
      --setting StableMotion_F1234_mogendit_chained \
      --mode ada_denoise \
      --step 10 \
      --device cuda \
      > "$LOG_DIR/stablemotion_mogendit_chain.log" 2>&1
  else
    echo "[stablemotion] skip StableMotion+MoGenDIT_chain (RUN_STABLE_CHAIN=0)"
  fi
  echo "[stablemotion] done $(date '+%F %T')"
}

run_mogendit_only() {
  echo "[mogendit_only] start $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=1 python3 scripts/postprocess_hymotion_with_mogendit.py \
    --input-source lq \
    --out-dir "$OUT/mogendit_only" \
    --eval-datalist "$DL" \
    --model-name MoGenDIT \
    --setting D_default \
    --mode ada_denoise \
    --step 10 \
    --device cuda \
    > "$LOG_DIR/mogendit_only.log" 2>&1
  echo "[mogendit_only] done $(date '+%F %T')"
}

run_hymotion_family() {
  echo "[hymotion] start $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=2 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local \
    --tasks E9 \
    --settings D_strict_mask_d2_b3_bsmooth_combo \
    --data-file-override "$DL_BASENAME" \
    --max-samples 99999 \
    --num-steps 50 \
    --save-npz \
    --output-dir "$OUT/m2m_combo_v4" \
    --device cuda \
    > "$LOG_DIR/hymotion_v4.log" 2>&1

  CUDA_VISIBLE_DEVICES=3 python3 scripts/postprocess_hymotion_with_mogendit.py \
    --src "$OUT/m2m_combo_v4/uncond_local/E9_D_strict_mask_d2_b3_bsmooth_combo" \
    --out-dir "$OUT/m2m_combo_v5_mogendit" \
    --eval-datalist "$DL" \
    --model-name uncond_local \
    --setting D_strict_mask_d2_b3_bsmooth_combo_mogendit \
    --mode trans_regen \
    --step 10 \
    --device cuda \
    > "$LOG_DIR/hymotion_v5_trans_regen.log" 2>&1

  CUDA_VISIBLE_DEVICES=3 python3 scripts/postprocess_hymotion_with_mogendit.py \
    --src "$OUT/m2m_combo_v5_mogendit/uncond_local/E9_D_strict_mask_d2_b3_bsmooth_combo_mogendit" \
    --out-dir "$OUT/m2m_combo_v6_mogendit_chained" \
    --eval-datalist "$DL" \
    --model-name uncond_local \
    --setting D_strict_mask_d2_b3_bsmooth_combo_chained \
    --mode ada_denoise \
    --step 10 \
    --device cuda \
    > "$LOG_DIR/hymotion_v6_ada_denoise.log" 2>&1

  CUDA_VISIBLE_DEVICES=4 python3 scripts/lq_overlay_clean_frames.py \
    --hq-dir "$OUT/m2m_combo_v6_mogendit_chained/uncond_local/E9_D_strict_mask_d2_b3_bsmooth_combo_chained" \
    --eval-datalist "$DL" \
    --out-dir "$OUT/m2m_combo_v9_trans_qc_clean" \
    --model-name "HyMotion-M2M+MoGenDIT_chain+LQoverlay" \
    --setting D_v6chain \
    --mode trans_qc_clean \
    --device cuda \
    > "$LOG_DIR/hymotion_v9_trans_qc_clean.log" 2>&1

  python3 scripts/select_e9_best_qc_candidate.py \
    --out-dir "$OUT/m2m_qc_select_v1" \
    --eval-datalist "$DL" \
    --model-name "HyMotion-M2M+MoGenDIT_QCSelect" \
    --setting D_qc_select \
    --v4-json "$(ls -t "$OUT"/m2m_combo_v4/eval_v2_*.json | head -n 1)" \
    --v5-logs "$OUT/m2m_combo_v5_mogendit/logs" \
    --v6-logs "$OUT/m2m_combo_v6_mogendit_chained/logs" \
    --overlay-logs "$OUT/m2m_combo_v9_trans_qc_clean/logs" \
    --max-samples 99999 \
    > "$LOG_DIR/hymotion_qc_select.log" 2>&1
  echo "[hymotion] done $(date '+%F %T')"
}

run_stablemotion &
pid_stable=$!
run_mogendit_only &
pid_mog=$!
run_hymotion_family &
pid_hym=$!

failed=0
for pair in "StableMotion:$pid_stable" "MoGenDIT:$pid_mog" "HyMotion:$pid_hym"; do
  name="${pair%%:*}"
  pid="${pair##*:}"
  if wait "$pid"; then
    echo "[runner] $name pipeline succeeded"
  else
    status=$?
    echo "[runner] $name pipeline FAILED with status $status"
    failed=1
  fi
done

if [[ "$failed" != 0 ]]; then
  echo "[runner] abort import because at least one pipeline failed"
  exit 1
fi

echo "[runner] relabel import JSONs $(date '+%F %T')"
python3 tools/split_eval_v2_to_flat.py \
  --in-dir "$OUT/m2m_combo_v4" \
  --out-dir "$OUT/m2m_combo_v4/import_jsons"

python3 scripts/relabel_dashboard_json.py \
  "$OUT/m2m_combo_v4/import_jsons/uncond_local__E9_D_strict_mask_d2_b3_bsmooth_combo.json" \
  "$OUT/renamed_import_jsons/HyMotion-M2M__E9_default.json" \
  --model-name "HyMotion-M2M" \
  --setting default

python3 scripts/relabel_dashboard_json.py \
  "$OUT/m2m_combo_v5_mogendit/import_jsons/uncond_local__E9_D_strict_mask_d2_b3_bsmooth_combo_mogendit.json" \
  "$OUT/renamed_import_jsons/HyMotion-M2M+MoGenDIT__E9_default.json" \
  --model-name "HyMotion-M2M+MoGenDIT" \
  --setting default

python3 scripts/relabel_dashboard_json.py \
  "$OUT/m2m_combo_v6_mogendit_chained/import_jsons/uncond_local__E9_D_strict_mask_d2_b3_bsmooth_combo_chained.json" \
  "$OUT/renamed_import_jsons/HyMotion-M2M+MoGenDIT_chain__E9_default.json" \
  --model-name "HyMotion-M2M+MoGenDIT_chain" \
  --setting default

if [[ -f "$OUT/stablemotion_mogendit_chain/import_jsons/stablemotion__E9_StableMotion_F1234_mogendit_chained.json" ]]; then
  python3 scripts/relabel_dashboard_json.py \
    "$OUT/stablemotion_mogendit_chain/import_jsons/stablemotion__E9_StableMotion_F1234_mogendit_chained.json" \
    "$OUT/renamed_import_jsons/StableMotion+MoGenDIT_chain__E9_default.json" \
    --model-name "StableMotion+MoGenDIT_chain" \
    --setting default
fi

echo "[runner] import dashboard JSONs $(date '+%F %T')"
cp "$DB" "$DB.bak_before_${DASHBOARD_SETTING}_$(date +%Y%m%d_%H%M%S)"

import_one() {
  local json_path="$1"
  local note="$2"
  python3 motion_annot_web/eval_dashboard/data_importer.py import \
    "$json_path" \
    --db "$DB" \
    --task E9 \
    --setting "$DASHBOARD_SETTING" \
    --notes "$note"
}

import_one "$OUT/stablemotion/import_jsons/StableMotion__E9_default.json" \
  "$RUN_TAG | StableMotion ensemble+enable_sits+skip40"
import_one "$OUT/mogendit_only/import_jsons/MoGenDIT__E9_D_default.json" \
  "$RUN_TAG | MoGenDIT ada_denoise direct-LQ"
import_one "$OUT/renamed_import_jsons/HyMotion-M2M__E9_default.json" \
  "$RUN_TAG | HyMotion-M2M strict_mask_d2_b3_bsmooth_combo"
import_one "$OUT/renamed_import_jsons/HyMotion-M2M+MoGenDIT__E9_default.json" \
  "$RUN_TAG | HyMotion-M2M + MoGenDIT trans_regen"
import_one "$OUT/renamed_import_jsons/HyMotion-M2M+MoGenDIT_chain__E9_default.json" \
  "$RUN_TAG | HyMotion-M2M + MoGenDIT chained"
import_one "$OUT/m2m_qc_select_v1/import_jsons/HyMotion-M2M+MoGenDIT_QCSelect__E9_D_qc_select.json" \
  "$RUN_TAG | HyMotion QC selector over v4/v5/v6/overlay"
if [[ -f "$OUT/renamed_import_jsons/StableMotion+MoGenDIT_chain__E9_default.json" ]]; then
  import_one "$OUT/renamed_import_jsons/StableMotion+MoGenDIT_chain__E9_default.json" \
    "$RUN_TAG | StableMotion + MoGenDIT chained"
fi

python3 - <<'PY'
import os
import sqlite3
db = os.environ['DB']
setting = os.environ['DASHBOARD_SETTING']
con = sqlite3.connect(db)
rows = con.execute(
    """
    SELECT r.id, m.name, r.setting, r.num_samples, r.created_at
    FROM eval_runs r
    JOIN models m ON m.id = r.model_id
    WHERE r.task_id = 'E9' AND r.setting = ?
    ORDER BY r.created_at DESC, r.id DESC
    """
    , (setting,)
).fetchall()
for row in rows:
    print('[db]', row)
PY

echo "[runner] complete $(date '+%F %T')"
