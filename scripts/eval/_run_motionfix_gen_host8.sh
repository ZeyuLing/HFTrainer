#!/usr/bin/env bash
# 8-GPU sharded MotionFix instruction-edit generation on one host.
# Spawns NGPU shards of eval_m2m_v2_all_tasks.py (E16 style_edit) over the
# MotionFix datalist, each pinned to one GPU, all writing NPZ under OUT.
#
# Env (optional except OUT):
#   MODEL (default smpl_caption_editfix_latest)  OUT (required)
#   NGPU (default 8)  GPUS (default 0..NGPU-1)
#   MAX_SAMPLES (0=all 1013)  NUM_STEPS (50)  CFG (2.5)
#   DATAFILE (default eval_motionfix_instruction.json, basename under EVAL_DATA_DIR)
#   EDITING_MODE (default 1; set 0 for the no-reactive control)
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
PY="$ROOT/.venv_t2m_a100/bin/python"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-smpl_caption_editfix_latest}
OUT=${OUT:?set OUT}
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_SAMPLES=${MAX_SAMPLES:-0}
NUM_STEPS=${NUM_STEPS:-50}
CFG=${CFG:-2.5}
DATAFILE=${DATAFILE:-eval_motionfix_instruction.json}
EDITING_MODE=${EDITING_MODE:-1}
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

if [ "$MAX_SAMPLES" = "0" ]; then limarg="--max-samples 1000000"; else limarg="--max-samples $MAX_SAMPLES"; fi
# No-reactive control: gate via env var (see eval_m2m_v2_all_tasks.py).
if [ "$EDITING_MODE" = "0" ]; then export MFIX_FORCE_NO_EDITING=1; else unset MFIX_FORCE_NO_EDITING; fi
mkdir -p "$OUT/_logs"
echo "[mfix-host8] $(date) MODEL=$MODEL OUT=$OUT NGPU=$NGPU MAX=$MAX_SAMPLES STEPS=$NUM_STEPS CFG=$CFG EDITING_MODE=$EDITING_MODE"
pids=()
for s in $(seq 0 $((NGPU-1))); do
  g=${GPU_ARR[$s]}
  CUDA_VISIBLE_DEVICES="$g" "$PY" scripts/eval/eval_m2m_v2_all_tasks.py \
    --models "$MODEL" --tasks E16 --settings style_edit \
    --data-file-override "$DATAFILE" $limarg --save-npz \
    --num-steps "$NUM_STEPS" --replacement-guidance skip_last \
    --text-guidance-scale "$CFG" \
    --num-shards "$NGPU" --shard-index "$s" \
    --output-dir "$OUT" > "$OUT/_logs/gen_shard${s}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[mfix-host8-done] $(date) npz=$(find "$OUT" -path '*/npz/*.npz' 2>/dev/null | wc -l)"
touch "$OUT/_GEN_DONE"
