#!/usr/bin/env bash
# Run the latest caption checkpoints for E1/E2 in setting-level shards.

set -u

REPO=${REPO:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

RUN_ROOT=${RUN_ROOT:-work_dirs/eval_e1e2_caption_latest_$(date +%Y%m%d_%H%M)}
MAX_SAMPLES=${MAX_SAMPLES:-1000000}
NUM_STEPS=${NUM_STEPS:-50}
GPU_LIST=${GPU_LIST:-0 1 2 3 4 5 6 7}

read -r -a MODELS <<< "${MODEL_LIST:-smpl_caption_resume_E2 M2M_v2_KIMODO_root_caption_permo_resume_E4}"
SETTINGS_E1=(default)
SETTINGS_E2=(
  start_1f end_1f both_1f
  pre20 post20 mid60
  pre20_uncond post20_uncond mid60_uncond
)

mkdir -p "$RUN_ROOT/logs"
{
  echo "started_at=$(date)"
  echo "run_root=$RUN_ROOT"
  echo "models=${MODELS[*]}"
  echo "gpu_list=$GPU_LIST"
  echo "max_samples=$MAX_SAMPLES num_steps=$NUM_STEPS"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
} > "$RUN_ROOT/manifest.txt"

read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "No GPUs provided" >&2
  exit 2
fi

idx=0
pids=()
for model in "${MODELS[@]}"; do
  for task in E1 E2; do
    if [ "$task" = E1 ]; then
      settings=("${SETTINGS_E1[@]}")
    else
      settings=("${SETTINGS_E2[@]}")
    fi
    for setting in "${settings[@]}"; do
      gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      out="$RUN_ROOT/$model/${task}_${setting}"
      log="$RUN_ROOT/logs/${model}__${task}_${setting}.log"
      mkdir -p "$out"
      echo "[launch] $(date) gpu=$gpu model=$model task=$task setting=$setting" \
        | tee -a "$RUN_ROOT/manifest.txt"
      CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_m2m_v2_all_tasks.py \
        --models "$model" \
        --tasks "$task" \
        --settings "$setting" \
        --max-samples "$MAX_SAMPLES" \
        --num-steps "$NUM_STEPS" \
        --replacement-guidance skip_last \
        --text-guidance-scale 1.0 \
        --use-rewritten \
        --save-npz \
        --output-dir "$out" > "$log" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      while [ "$(jobs -pr | wc -l)" -ge "${#GPUS[@]}" ]; do
        sleep 20
      done
    done
  done
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
echo "finished_at=$(date) fail=$fail" >> "$RUN_ROOT/manifest.txt"
exit "$fail"
