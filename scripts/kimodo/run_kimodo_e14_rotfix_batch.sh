#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export TEXT_ENCODER_MODE=local
export TEXT_ENCODER=dummy

LOG_DIR="work_dirs/e14_full_legaware_20260427/logs"
SHARD_ROOT="work_dirs/e14_full_legaware_20260427/kimodo_rotfix_shards"
FINAL_ROOT="work_dirs/e14_full_legaware_20260427/kimodo"
mkdir -p "$LOG_DIR"
rm -rf "$SHARD_ROOT"
mkdir -p "$SHARD_ROOT"

run_shard() {
  local setting="$1"
  local gpu="$2"
  local start_idx="$3"
  local end_idx="$4"
  local shard_id="$5"
  local out_dir="${SHARD_ROOT}/E14_${setting}_shard${shard_id}"
  local log_file="$LOG_DIR/kimodo_E14_${setting}_rotfix_shard${shard_id}_${start_idx}_${end_idx}.log"

  echo "[run] E14_${setting} idx ${start_idx}:${end_idx} on GPU ${gpu} -> ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u tools/run_kimodo_all_tasks.py \
    --tasks E14 \
    --settings "${setting}" \
    --max-samples 100 \
    --start-idx "${start_idx}" \
    --end-idx "${end_idx}" \
    --output-dir "${out_dir}" \
    --use-caption no \
    >"${log_file}" 2>&1
}

run_shard M 1 0 25 0 &
run_shard M 2 25 50 1 &
run_shard M 3 50 75 2 &
run_shard M 4 75 100 3 &
run_shard L 5 0 34 0 &
run_shard L 6 34 67 1 &
run_shard L 7 67 100 2 &

wait

python3 tools/merge_kimodo_e14_shards.py \
  --shard-root "$SHARD_ROOT" \
  --final-root "$FINAL_ROOT" \
  --settings M L \
  >"$LOG_DIR/merge_E14_rotfix.log" 2>&1

python3 tools/append_kimodo_context_soma77.py \
  --run-dir work_dirs/e14_full_legaware_20260427/kimodo/E14_M/E14_M \
  --data-file data/eval/m2m_v2/eval_e14_hq400h_move100.json \
  --motion-data-dir / \
  --placement velocity \
  --bone-offsets data/hymotion_m2m_data/bone_offsets_22.pt \
  >"$LOG_DIR/append_E14_M_rotfix.log" 2>&1

python3 tools/append_kimodo_context_soma77.py \
  --run-dir work_dirs/e14_full_legaware_20260427/kimodo/E14_L/E14_L \
  --data-file data/eval/m2m_v2/eval_e14_hq400h_static100.json \
  --motion-data-dir / \
  --placement overlap \
  --bone-offsets data/hymotion_m2m_data/bone_offsets_22.pt \
  >"$LOG_DIR/append_E14_L_rotfix.log" 2>&1

python3 tools/diagnose_kimodo_e14_boundary_jumps.py \
  work_dirs/e14_full_legaware_20260427/kimodo/E14_M/E14_M \
  work_dirs/e14_full_legaware_20260427/kimodo/E14_L/E14_L \
  --top-k 20 \
  >"$LOG_DIR/kimodo_E14_rotfix_boundary_diag.log" 2>&1

echo "[done] E14 rotfix batch complete"
echo "[diag] $LOG_DIR/kimodo_E14_rotfix_boundary_diag.log"
