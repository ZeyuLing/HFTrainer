#!/bin/bash
set -u

REPO=${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

RUN_ROOT=${RUN_ROOT:-work_dirs/m2m_v2_eval_four_new_missing_all_20260514_1306_machine1}
COMMON=(--max-samples 80 --num-steps 50 --replacement-guidance skip_last --text-guidance-scale 5.0 --save-npz --use-rewritten)
EVAL=scripts/eval/eval_m2m_v2_all_tasks.py

mkdir -p "$RUN_ROOT/logs"
echo "[rescue2-start] $(date)" | tee -a "$RUN_ROOT/logs/rescue_caption_e10_v2.log"

# Stop the broken first rescue loop if it is still waiting.
pkill -TERM -f /tmp/rescue_caption_e10_m1.sh || true

run_e10_group() {
  local model="$1"; shift
  local gpu="$1"; shift
  local group="$1"; shift
  local out="$RUN_ROOT/$model/e10_rescue/$group/E10"
  local log="$RUN_ROOT/logs/${model}__e10_rescue_${group}.log"
  mkdir -p "$out"
  echo "[start] $(date) model=$model group=$group gpu=$gpu settings=$*" > "$log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$EVAL" \
    --models "$model" \
    --tasks E10 \
    --settings "$@" \
    "${COMMON[@]}" \
    --output-dir "$out" >> "$log" 2>&1
  local rc=$?
  echo "[done] $(date) model=$model group=$group rc=$rc" >> "$log"
  return "$rc"
}

pids=()
run_e10_group kimodo_caption_E4 0 g1 A_upper B_lower C_spine_only D_arms_only &
pids+=("$!")
run_e10_group kimodo_caption_E4 1 g2 E_legs_only F_left_arm G_right_arm H_left_leg &
pids+=("$!")
run_e10_group kimodo_caption_E4 3 g3 I_right_leg J_feet_only K_no_feet &
pids+=("$!")
run_e10_group smpl_caption_E2 4 g1 A_upper B_lower C_spine_only D_arms_only &
pids+=("$!")
run_e10_group smpl_caption_E2 5 g2 E_legs_only F_left_arm G_right_arm H_left_leg &
pids+=("$!")
run_e10_group smpl_caption_E2 7 g3 I_right_leg J_feet_only K_no_feet &
pids+=("$!")

fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then
    fail=$((fail + 1))
  fi
done
echo "[rescue2-e10-done] $(date) fail=$fail" | tee -a "$RUN_ROOT/logs/rescue_caption_e10_v2.log"

while pgrep -f "$RUN_ROOT/.*/common_c/E13" >/dev/null; do
  echo "[rescue2-wait-common-c] $(date)" >> "$RUN_ROOT/logs/rescue_caption_e10_v2.log"
  sleep 60
done

while pgrep -f "four_new_missing_all_m1|$RUN_ROOT.*run_four_new_m2m_eval_missing_all" >/dev/null; do
  echo "[rescue2-wait-parent] $(date)" >> "$RUN_ROOT/logs/rescue_caption_e10_v2.log"
  sleep 30
done

if [ "$fail" -eq 0 ]; then
  for model in kimodo_caption_E4 smpl_caption_E2; do
    import_log="$RUN_ROOT/logs/${model}__rescue_import.log"
    flock "$REPO/motion_annot_web/eval_dashboard/eval_dashboard.db.import.lock" \
      python3 scripts/eval/split_and_import_eval_v2.py "$RUN_ROOT/$model" \
        --notes "four_new_models_missing_all_20260514_1306_missing_all:${model}:rescue_e10" \
        > "$import_log" 2>&1
    rc=$?
    echo "[rescue2-import] $(date) model=$model rc=$rc" | tee -a "$RUN_ROOT/logs/rescue_caption_e10_v2.log"
    if [ "$rc" -eq 0 ]; then
      rm -f "$RUN_ROOT/${model}.failed"
      echo "DONE $model $(date) RUN_ROOT=$RUN_ROOT rescue_e10" > "$RUN_ROOT/${model}.done"
    else
      echo "IMPORT_FAILED $model $(date) RUN_ROOT=$RUN_ROOT rescue_e10" > "$RUN_ROOT/${model}.failed"
      fail=$((fail + 1))
    fi
  done
fi

echo "[rescue2-all-done] $(date) fail=$fail" | tee -a "$RUN_ROOT/logs/rescue_caption_e10_v2.log"
exit "$fail"
