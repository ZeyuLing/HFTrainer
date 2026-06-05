#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONUNBUFFERED=1
export MOGENDIT_ROOT="${MOGENDIT_ROOT:-$PWD/ref_repo/MoGenDiT}"
export MOGENDIT_CKPT_ROOT="${MOGENDIT_CKPT_ROOT:-/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/save/ckpt}"

OUT="${OUT:-work_dirs/e9_full_gpu_20260428}"
DL="${DL:-data/eval/m2m_v2/eval_e9_repair_v2.json}"
DB="${DB:-motion_annot_web/eval_dashboard/eval_dashboard.db}"
LOG_DIR="$OUT/logs"
if [[ "${CLEAN_OUT:-1}" == "1" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$LOG_DIR"

echo "[runner] start $(date '+%F %T')"
echo "[runner] host=$(hostname) pwd=$PWD out=$OUT"
nvidia-smi || true

run_stablemotion() {
  echo "[stablemotion] start $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_stablemotion_e9.py \
    --eval-datalist "$DL" \
    --output-dir "$OUT/stablemotion" \
    --max-samples 9999 \
    --device cuda \
    > "$LOG_DIR/stablemotion_infer.log" 2>&1

  CUDA_VISIBLE_DEVICES=0 python3 scripts/stablemotion_to_dashboard.py \
    --src "$OUT/stablemotion" \
    --out-dir "$OUT/stablemotion" \
    --eval-datalist "$DL" \
    --model-name StableMotion \
    --setting default \
    --device cuda \
    > "$LOG_DIR/stablemotion_import_json.log" 2>&1
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

run_hymotion_chain() {
  echo "[hymotion_chain] start $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=2 python3 tools/eval_m2m_v2_all_tasks.py \
    --models uncond_local \
    --tasks E9 \
    --settings D_strict_mask_d2_b3_bsmooth_combo \
    --max-samples 9999 \
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
  echo "[hymotion_chain] done $(date '+%F %T')"
}

run_stablemotion &
pid_stable=$!
run_mogendit_only &
pid_mog=$!
run_hymotion_chain &
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

echo "[runner] import dashboard JSONs $(date '+%F %T')"
cp "$DB" "$DB.bak_before_e9_full_gpu_$(date +%Y%m%d_%H%M%S)"

import_one() {
  local json_path="$1"
  local note="$2"
  if [[ ! -f "$json_path" ]]; then
    echo "[import] missing JSON: $json_path"
    return 1
  fi
  python3 motion_annot_web/eval_dashboard/data_importer.py import "$json_path" \
    --db "$DB" \
    --task E9 \
    --setting E9_default \
    --notes "$note"
}

import_one "$OUT/stablemotion/import_jsons/StableMotion__E9_default.json" \
  "Full GPU rerun on lzy_debug_machine, StableMotion reference-faithful pipeline, 2026-04-28"
import_one "$OUT/mogendit_only/import_jsons/MoGenDIT__E9_D_default.json" \
  "Full GPU rerun on lzy_debug_machine, MoGenDIT ref_repo pipeline, 2026-04-28"
import_one "$OUT/m2m_combo_v9_trans_qc_clean/import_jsons/HyMotion-M2M+MoGenDIT_chain+LQoverlay__E9_D_v6chain_trans_qc_clean.json" \
  "Full GPU rerun on lzy_debug_machine, HyMotion chain plus LQ translation QC clean overlay, 2026-04-28"

python3 - <<'PY'
import sqlite3
db = 'motion_annot_web/eval_dashboard/eval_dashboard.db'
con = sqlite3.connect(db)
rows = con.execute(
    """
    SELECT r.id, m.name, r.setting, r.num_samples, r.created_at
    FROM eval_runs r
    JOIN models m ON m.id = r.model_id
    WHERE r.task_id = 'E9' AND r.setting = 'E9_default'
    ORDER BY r.created_at DESC, r.id DESC
    LIMIT 8
    """
).fetchall()
for row in rows:
    print('[db]', row)
PY

echo "[runner] complete $(date '+%F %T')"
