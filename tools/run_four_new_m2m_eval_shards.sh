#!/bin/bash
# Run one or more new M2M v2 models on their corresponding eval-dashboard cases.
#
# Intended for lzy_debug_machine_1/2. The four Phase-0 models map to the
# matching eval tasks:
#   smpl_uncond_E1    -> E1
#   smpl_caption_E2   -> E2
#   kimodo_uncond_E3  -> E3
#   kimodo_caption_E4 -> E4

set -u

REPO=${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

TS=${OUT_TS:-$(date +%Y%m%d_%H%M)}
RUN_ROOT=${RUN_ROOT:-work_dirs/m2m_v2_eval_four_new_${TS}}
MAX_SAMPLES=${MAX_SAMPLES:-80}
NUM_STEPS=${NUM_STEPS:-50}
GPU_BASE=${GPU_BASE:-0}
MODEL_CSV=${MODEL_CSV:?MODEL_CSV is required, e.g. kimodo_caption_E4,smpl_caption_E2}

mkdir -p "$RUN_ROOT/logs"
echo "RUN_ROOT=$RUN_ROOT" | tee -a "$RUN_ROOT/logs/run_meta.txt"
echo "MODEL_CSV=$MODEL_CSV GPU_BASE=$GPU_BASE MAX_SAMPLES=$MAX_SAMPLES NUM_STEPS=$NUM_STEPS" | tee -a "$RUN_ROOT/logs/run_meta.txt"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "$RUN_ROOT/logs/run_meta.txt"

COMMON="--max-samples $MAX_SAMPLES --num-steps $NUM_STEPS --replacement-guidance skip_last --text-guidance-scale 5.0 --save-npz"
EVAL="scripts/eval/eval_m2m_v2_all_tasks.py"

TASKS_SMPL_UNCOND_E1=(
  "E1:default"
)

TASKS_SMPL_CAPTION_E2_A=(
  "E2:start_1f,end_1f,both_1f"
)

TASKS_SMPL_CAPTION_E2_B=(
  "E2:pre20,post20,mid60"
)

TASKS_KIMODO_UNCOND_E3_A=(
  "E3:every_5f,every_10f,every_15f"
)

TASKS_KIMODO_UNCOND_E3_B=(
  "E3:every_30f,every_60f,adaptive"
)

TASKS_KIMODO_CAPTION_E4_A=(
  "E4:A_rhand_sparse,B_ankles_sparse,C_rhand_lfoot"
)

TASKS_KIMODO_CAPTION_E4_B=(
  "E4:D_both_hands,E_all4_sparse,F_rhand_dense"
)

run_task_list() {
  local model="$1"; shift
  local shard="$1"; shift
  local gpu="$1"; shift
  local task_list_name="$1"; shift
  local -n task_list_ref="$task_list_name"
  local shard_dir="$RUN_ROOT/$model/$shard"
  local log="$RUN_ROOT/logs/${model}__${shard}.log"
  mkdir -p "$shard_dir"
  {
    echo "[start] $(date) model=$model shard=$shard gpu=$gpu"
    for item in "${task_list_ref[@]}"; do
      local task="${item%%:*}"
      local settings_csv="${item#*:}"
      IFS=',' read -r -a settings <<< "$settings_csv"
      local out="$shard_dir/${task}"
      mkdir -p "$out"
      local extra=""
      case "$model" in
        *caption*) extra="--use-rewritten" ;;
      esac
      case "$model" in
        smpl_uncond_E1) extra="$extra --allow-uncond-caption-required" ;;
      esac
      echo "[run] $(date) model=$model task=$task settings=${settings[*]}"
      CUDA_VISIBLE_DEVICES="$gpu" python3 "$EVAL" \
        --models "$model" \
        --tasks "$task" \
        --settings "${settings[@]}" \
        $COMMON $extra \
        --output-dir "$out"
      local rc=$?
      echo "[done-task] $(date) model=$model task=$task rc=$rc"
      if [ "$rc" -ne 0 ]; then
        echo "[fail] model=$model shard=$shard task=$task rc=$rc"
        return "$rc"
      fi
    done
    echo "[done] $(date) model=$model shard=$shard"
  } > "$log" 2>&1
}

run_model() {
  local model="$1"; local base_gpu="$2"
  local model_dir="$RUN_ROOT/$model"
  mkdir -p "$model_dir"
  echo "[model-start] $(date) $model base_gpu=$base_gpu" | tee -a "$RUN_ROOT/logs/run_meta.txt"
  declare -a shard_pids=()
  case "$model" in
    smpl_uncond_E1)
      run_task_list "$model" e1 "$((base_gpu + 0))" TASKS_SMPL_UNCOND_E1 &
      shard_pids+=("$!")
      ;;
    smpl_caption_E2)
      run_task_list "$model" e2a "$((base_gpu + 0))" TASKS_SMPL_CAPTION_E2_A &
      shard_pids+=("$!")
      run_task_list "$model" e2b "$((base_gpu + 1))" TASKS_SMPL_CAPTION_E2_B &
      shard_pids+=("$!")
      ;;
    kimodo_uncond_E3)
      run_task_list "$model" e3a "$((base_gpu + 0))" TASKS_KIMODO_UNCOND_E3_A &
      shard_pids+=("$!")
      run_task_list "$model" e3b "$((base_gpu + 1))" TASKS_KIMODO_UNCOND_E3_B &
      shard_pids+=("$!")
      ;;
    kimodo_caption_E4)
      run_task_list "$model" e4a "$((base_gpu + 0))" TASKS_KIMODO_CAPTION_E4_A &
      shard_pids+=("$!")
      run_task_list "$model" e4b "$((base_gpu + 1))" TASKS_KIMODO_CAPTION_E4_B &
      shard_pids+=("$!")
      ;;
    *)
      echo "Unknown model mapping for $model" | tee -a "$RUN_ROOT/logs/run_meta.txt"
      echo "FAILED $model $(date) RUN_ROOT=$RUN_ROOT" > "$RUN_ROOT/${model}.failed"
      return 1
      ;;
  esac
  local fail=0
  for p in "${shard_pids[@]}"; do
    if ! wait "$p"; then fail=$((fail + 1)); fi
  done
  echo "[model-shards-done] $(date) $model fail=$fail" | tee -a "$RUN_ROOT/logs/run_meta.txt"
  if [ "$fail" -eq 0 ]; then
    local lock="$REPO/motion_annot_web/eval_dashboard/eval_dashboard.db.import.lock"
    local import_log="$RUN_ROOT/logs/${model}__import.log"
    flock "$lock" python3 scripts/eval/split_and_import_eval_v2.py "$model_dir" \
      --notes "four_new_models_${TS}:${model}" > "$import_log" 2>&1
    local import_rc=$?
    echo "[model-import-done] $(date) $model rc=$import_rc" | tee -a "$RUN_ROOT/logs/run_meta.txt"
    if [ "$import_rc" -eq 0 ]; then
      echo "DONE $model $(date) RUN_ROOT=$RUN_ROOT" > "$RUN_ROOT/${model}.done"
    else
      echo "IMPORT_FAILED $model $(date) RUN_ROOT=$RUN_ROOT" > "$RUN_ROOT/${model}.failed"
      fail=$((fail + 1))
    fi
  else
    echo "FAILED $model $(date) RUN_ROOT=$RUN_ROOT" > "$RUN_ROOT/${model}.failed"
  fi
  return "$fail"
}

IFS=',' read -r -a MODELS <<< "$MODEL_CSV"
declare -a PIDS=()
for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  base_gpu=$((GPU_BASE + idx * 4))
  run_model "$model" "$base_gpu" &
  PIDS+=("$!")
done

fail=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then fail=$((fail + 1)); fi
done

echo "[all-done] $(date) fail=$fail" | tee -a "$RUN_ROOT/logs/run_meta.txt"
exit "$fail"
