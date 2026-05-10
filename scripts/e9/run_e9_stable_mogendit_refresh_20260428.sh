#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONUNBUFFERED=1
export MOGENDIT_ROOT="${MOGENDIT_ROOT:-$PWD/ref_repo/MoGenDiT}"
export MOGENDIT_CKPT_ROOT="${MOGENDIT_CKPT_ROOT:-/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT/save/ckpt}"

OUT="${OUT:-work_dirs/e9_refresh_20260428}"
DL="${DL:-data/eval/m2m_v2/eval_e9_repair_v2.json}"
DB="${DB:-motion_annot_web/eval_dashboard/eval_dashboard.db}"
LOG_DIR="$OUT/logs"
if [[ "${CLEAN_OUT:-1}" == "1" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$LOG_DIR"

echo "[refresh] start $(date '+%F %T')"
echo "[refresh] host=$(hostname) pwd=$PWD out=$OUT"
nvidia-smi || true

run_stablemotion() {
  echo "[stablemotion] start $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_stablemotion_e9.py \
    --eval-datalist "$DL" \
    --output-dir "$OUT/stablemotion" \
    --max-samples 9999 \
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

run_stablemotion &
pid_stable=$!
run_mogendit_only &
pid_mog=$!

failed=0
for pair in "StableMotion:$pid_stable" "MoGenDIT:$pid_mog"; do
  name="${pair%%:*}"
  pid="${pair##*:}"
  if wait "$pid"; then
    echo "[refresh] $name pipeline succeeded"
  else
    status=$?
    echo "[refresh] $name pipeline FAILED with status $status"
    failed=1
  fi
done

if [[ "$failed" != 0 ]]; then
  echo "[refresh] abort import because at least one pipeline failed"
  exit 1
fi

echo "[refresh] import dashboard JSONs $(date '+%F %T')"
cp "$DB" "$DB.bak_before_e9_refresh_$(date +%Y%m%d_%H%M%S)"

python3 motion_annot_web/eval_dashboard/data_importer.py import \
  "$OUT/stablemotion/import_jsons/StableMotion__E9_default.json" \
  --db "$DB" \
  --task E9 \
  --setting E9_default \
  --notes "Full GPU rerun on lzy_debug_machine_2, StableMotion ensemble+enable_sits+skip40, 2026-04-28"

python3 motion_annot_web/eval_dashboard/data_importer.py import \
  "$OUT/mogendit_only/import_jsons/MoGenDIT__E9_D_default.json" \
  --db "$DB" \
  --task E9 \
  --setting E9_default \
  --notes "Full GPU rerun on lzy_debug_machine_2, MoGenDIT ada_denoise with direct LQ NPZ input path, 2026-04-28"

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
      AND m.name IN ('StableMotion', 'MoGenDIT')
    ORDER BY r.created_at DESC, r.id DESC
    """
).fetchall()
for row in rows:
    print('[db]', row)
PY

echo "[refresh] complete $(date '+%F %T')"
