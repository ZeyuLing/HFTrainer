#!/usr/bin/env bash
# Sharded \\ours{} editing generation for Table 9 (MotionFix) / Table 10 (PerMo).
# Spawns NGPU processes of eval_m2m_v2_all_tasks.py, each a --shard, all saving
# NPZ (motion_135) under <OUT>/<model>/<task>_<setting>/npz/<global_idx>.npz.
#
# Env:
#   MODEL     eval registry key (default smpl_caption_editfix_latest)
#   TASK      task id (default E16)
#   SETTING   setting name (default style_edit)
#   DATAFILE  datalist override (required for MotionFix; empty=native task data)
#   OUT       output dir (required)
#   NGPU,GPUS,MAX_SAMPLES,NUM_STEPS,CFG  sampling params
#   USE_REWRITTEN  1 -> pass --use-rewritten (PerMo native), else raw caption
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

MODEL=${MODEL:-smpl_caption_editfix_latest}
TASK=${TASK:-E16}
SETTING=${SETTING:-style_edit}
DATAFILE=${DATAFILE:-}
OUT=${OUT:?set OUT}
NGPU=${NGPU:-4}
GPUS=${GPUS:-0,1,2,3}
MAX_SAMPLES=${MAX_SAMPLES:-0}
NUM_STEPS=${NUM_STEPS:-50}
CFG=${CFG:-2.5}
USE_REWRITTEN=${USE_REWRITTEN:-0}
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

NUM_NODES=${NUM_NODES:-1}; NODE_RANK=${NODE_RANK:-${INDEX:-0}}
TOTAL_SHARDS=$((NGPU*NUM_NODES))
LOG="$OUT/_logs"; mkdir -p "$LOG"

dfarg=""; [ -n "$DATAFILE" ] && dfarg="--data-file-override $DATAFILE"
# MAX_SAMPLES=0 means ALL samples; the python default is only 50, so pass a
# large cap to disable the limit.
if [ "$MAX_SAMPLES" = "0" ]; then limarg="--max-samples 1000000"; else limarg="--max-samples $MAX_SAMPLES"; fi
rwarg=""; [ "$USE_REWRITTEN" = "1" ] && rwarg="--use-rewritten"

echo "[start-edit-gen] $(date) MODEL=$MODEL TASK=$TASK SETTING=$SETTING DATAFILE=$DATAFILE OUT=$OUT shards=$TOTAL_SHARDS"
pids=()
for s in $(seq 0 $((NGPU-1))); do
  g=${GPU_ARR[$s]}; gshard=$((NODE_RANK*NGPU + s))
  CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/eval_m2m_v2_all_tasks.py \
    --models "$MODEL" --tasks "$TASK" --settings "$SETTING" \
    $dfarg $limarg $rwarg --save-npz \
    --num-steps "$NUM_STEPS" --replacement-guidance skip_last \
    --text-guidance-scale "$CFG" \
    --num-shards "$TOTAL_SHARDS" --shard-index "$gshard" \
    --output-dir "$OUT" \
    > "$LOG/gen_shard${gshard}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[gen-done] $(date) npz=$(find "$OUT" -path '*/npz/*.npz' 2>/dev/null | wc -l)"
touch "$OUT/_GEN_DONE_node${NODE_RANK}"
