#!/usr/bin/env bash
# Launch tail M2M eval jobs on an already-running lzy_debug_machine_* Taiji node.
set -euo pipefail

MODE="${1:?mode required: m1 or m2}"
REPO="${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

case "$MODE" in
  m1)
    RUN_ROOT="${RUN_ROOT:-work_dirs/eval_4models_E1E15_latest_20260525_131715_lzy_debug_m1}"
    ;;
  m2)
    RUN_ROOT="${RUN_ROOT:-work_dirs/eval_4models_E1E15_latest_20260525_131715_lzy_debug_m2}"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT"
export MODE REPO RUN_ROOT

driver() {
  set -u
  {
    echo "started_at=$(date)"
    echo "host=$(hostname)"
    echo "mode=$MODE"
    echo "run_root=$RUN_ROOT"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
  } > "$RUN_ROOT/taiji_manifest.txt"

  set +e
  if [ "$MODE" = "m1" ]; then
    python3 -u scripts/eval/run_m2m_v2_latest_queue.py \
      --repo-root "$REPO" \
      --models M2M_v2_KIMODO_root_caption_permo_resume_E4 \
      --tasks E14 E15 \
      --gpus 0 1 \
      --output-root "$RUN_ROOT/caption_tail" \
      --max-samples 1000000 \
      --num-steps 50 \
      --replacement-guidance skip_last \
      --text-guidance-scale 1.0 \
      --use-rewritten \
      --save-npz \
      --run-caption-nonaware \
      --allow-uncond-caption-required \
      --include-routine-skipped \
      2>&1 | tee "$RUN_ROOT/queue_caption_tail.log"
    rc1=${PIPESTATUS[0]}

    python3 -u scripts/eval/run_m2m_v2_latest_queue.py \
      --repo-root "$REPO" \
      --models M2M_v2_KIMODO_root_uncond_E3 \
      --tasks E1 E2 E3 E4 E5 E7 E8 E9 E10 E13 E14 E15 \
      --gpus 2 3 4 5 6 7 \
      --output-root "$RUN_ROOT/kimodo_uncond" \
      --max-samples 1000000 \
      --num-steps 50 \
      --replacement-guidance skip_last \
      --text-guidance-scale 1.0 \
      --use-rewritten \
      --save-npz \
      --run-caption-nonaware \
      --allow-uncond-caption-required \
      --include-routine-skipped \
      2>&1 | tee "$RUN_ROOT/queue_kimodo_uncond.log"
    rc2=${PIPESTATUS[0]}
    queue_rc=$((rc1 || rc2))
  else
    python3 -u scripts/eval/run_m2m_v2_latest_queue.py \
      --repo-root "$REPO" \
      --models smpl_uncond_E1 \
      --tasks E1 E2 E3 E4 E5 E7 E8 E9 E10 E13 E14 E15 \
      --gpus 0 1 2 3 4 5 6 7 \
      --output-root "$RUN_ROOT/smpl_uncond" \
      --max-samples 1000000 \
      --num-steps 50 \
      --replacement-guidance skip_last \
      --text-guidance-scale 1.0 \
      --use-rewritten \
      --save-npz \
      --run-caption-nonaware \
      --allow-uncond-caption-required \
      --include-routine-skipped \
      2>&1 | tee "$RUN_ROOT/queue_smpl_uncond.log"
    queue_rc=${PIPESTATUS[0]}
  fi

  python3 scripts/eval/split_and_import_eval_v2.py \
    "$RUN_ROOT" \
    --include-skipped \
    --notes "latest checkpoint tail rerun 20260525 lzy_debug_machine_${MODE}" \
    2>&1 | tee "$RUN_ROOT/import.log"
  import_rc=${PIPESTATUS[0]}

  echo "finished_at=$(date) queue_rc=$queue_rc import_rc=$import_rc" \
    | tee -a "$RUN_ROOT/taiji_manifest.txt"
  exit $((queue_rc || import_rc))
}

nohup bash -c "$(declare -f driver); driver" > "$RUN_ROOT/driver.log" 2>&1 &
echo "$!" > "$RUN_ROOT/pid.txt"
echo "launched_${MODE}_pid=$(cat "$RUN_ROOT/pid.txt")"
echo "run_root=$RUN_ROOT"
