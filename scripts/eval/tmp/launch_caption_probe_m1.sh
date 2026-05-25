set -euo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
RUN_ROOT="work_dirs/caption_e1_cfg_probe_20260525_m1"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/ckpt_alias"
make_alias() {
  local alias="$1" src="$2"
  local d="$RUN_ROOT/ckpt_alias/$alias"
  rm -rf "$d"
  mkdir -p "$d"
  ln -s "$(pwd)/$src" "$d/$(basename "$src")"
}
make_alias smpl_e730 work_dirs/hymotion_m2m_v2_smpl_caption_resume_E2/checkpoint-epoch_730
make_alias smpl_e760 work_dirs/hymotion_m2m_v2_smpl_caption_resume_E2/checkpoint-epoch_760
make_alias smpl_e790 work_dirs/hymotion_m2m_v2_smpl_caption_resume_E2/checkpoint-epoch_790
make_alias kimodo_e660 work_dirs/hymotion_m2m_v2_kimodo_caption_permo_resume_E4/checkpoint-epoch_660
make_alias kimodo_e780 work_dirs/hymotion_m2m_v2_kimodo_caption_permo_resume_E4/checkpoint-epoch_780
make_alias kimodo_e800 work_dirs/hymotion_m2m_v2_kimodo_caption_permo_resume_E4/checkpoint-epoch_800
cat > "$RUN_ROOT/driver.sh" <<'EOS'
set -euo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
RUN_ROOT="work_dirs/caption_e1_cfg_probe_20260525_m1"
export PYTHONPATH=.
run_one() {
  local model="$1" alias="$2" scale="$3" gpu="$4"
  local envkey="_EVAL_WORK_DIR__${model}"
  local out="$RUN_ROOT/${model}_${alias}_gs${scale}"
  mkdir -p "$out"
  echo "[$(date)] start model=$model alias=$alias scale=$scale gpu=$gpu" | tee "$out/launcher.log"
  env CUDA_VISIBLE_DEVICES="$gpu" "${envkey}"="$RUN_ROOT/ckpt_alias/$alias" \
    python3 -u scripts/eval/eval_m2m_v2_all_tasks.py \
      --models "$model" --tasks E1 --settings default \
      --max-samples 12 --num-steps 30 --replacement-guidance skip_last \
      --output-dir "$out" --text-guidance-scale "$scale" \
      --use-rewritten --save-npz --seed-base 0xE4A10000 \
      > "$out/run.log" 2>&1
  echo "[$(date)] done model=$model alias=$alias scale=$scale" | tee -a "$out/launcher.log"
}
(
  run_one smpl_caption_resume_E2 smpl_e730 1.0 0
  run_one smpl_caption_resume_E2 smpl_e730 3.0 0
  run_one smpl_caption_resume_E2 smpl_e760 1.0 0
  run_one smpl_caption_resume_E2 smpl_e760 3.0 0
  run_one smpl_caption_resume_E2 smpl_e790 1.0 0
  run_one smpl_caption_resume_E2 smpl_e790 3.0 0
) > "$RUN_ROOT/gpu0_smpl.log" 2>&1 &
PID0=$!
(
  run_one M2M_v2_KIMODO_root_caption_permo_resume_E4 kimodo_e660 1.0 1
  run_one M2M_v2_KIMODO_root_caption_permo_resume_E4 kimodo_e660 3.0 1
  run_one M2M_v2_KIMODO_root_caption_permo_resume_E4 kimodo_e780 1.0 1
  run_one M2M_v2_KIMODO_root_caption_permo_resume_E4 kimodo_e780 3.0 1
  run_one M2M_v2_KIMODO_root_caption_permo_resume_E4 kimodo_e800 1.0 1
  run_one M2M_v2_KIMODO_root_caption_permo_resume_E4 kimodo_e800 3.0 1
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
