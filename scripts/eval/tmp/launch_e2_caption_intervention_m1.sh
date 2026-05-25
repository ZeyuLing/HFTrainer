set -euo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
RUN_ROOT="work_dirs/e2_caption_intervention_20260525_m1"
mkdir -p "$RUN_ROOT"
cat > "$RUN_ROOT/driver.sh" <<'EOS'
set -euo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
RUN_ROOT="work_dirs/e2_caption_intervention_20260525_m1"
export PYTHONPATH=.
run_one() {
  local model="$1" mode="$2" gpu="$3"
  local out="$RUN_ROOT/${model}_${mode}"
  mkdir -p "$out"
  echo "[$(date)] start model=$model mode=$mode gpu=$gpu" | tee "$out/launcher.log"
  env CUDA_VISIBLE_DEVICES="$gpu" \
    python3 -u scripts/eval/eval_m2m_v2_all_tasks.py \
      --models "$model" --tasks E2 \
      --settings start_1f both_1f pre20 mid60 \
      --max-samples 20 --num-steps 30 --replacement-guidance skip_last \
      --output-dir "$out" --text-guidance-scale 1.0 \
      --use-rewritten --save-npz --seed-base 0xE4A20000 \
      --caption-override-mode "$mode" \
      > "$out/run.log" 2>&1
  echo "[$(date)] done model=$model mode=$mode" | tee -a "$out/launcher.log"
}
(
  run_one smpl_caption_resume_E2 none 0
  run_one smpl_caption_resume_E2 blank 0
  run_one smpl_caption_resume_E2 shuffle 0
) > "$RUN_ROOT/gpu0_smpl.log" 2>&1 &
PID0=$!
(
  run_one M2M_v2_KIMODO_root_caption_permo_resume_E4 none 1
  run_one M2M_v2_KIMODO_root_caption_permo_resume_E4 blank 1
  run_one M2M_v2_KIMODO_root_caption_permo_resume_E4 shuffle 1
) > "$RUN_ROOT/gpu1_kimodo.log" 2>&1 &
PID1=$!
echo "$PID0 $PID1" > "$RUN_ROOT/pids.txt"
set +e
wait $PID0; RC0=$?
wait $PID1; RC1=$?
set -e
echo "finished_at=$(date) rc0=$RC0 rc1=$RC1" > "$RUN_ROOT/done.txt"
exit $((RC0 || RC1))
EOS
chmod +x "$RUN_ROOT/driver.sh"
nohup bash "$RUN_ROOT/driver.sh" > "$RUN_ROOT/nohup.log" 2>&1 &
echo $! > "$RUN_ROOT/driver.pid"
echo "launched driver_pid=$(cat "$RUN_ROOT/driver.pid") run_root=$RUN_ROOT"
