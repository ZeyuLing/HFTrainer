#!/usr/bin/env bash
# MotionFix instruction-edit generation (Table 9) for \ours editing model.
# Generates motion_135 NPZ via eval_m2m_v2_all_tasks.py (E16 style_edit) using the
# MotionFix datalist (source motion -> reactive channel + edit text -> regen).
#
# Env (all optional except OUT):
#   MODEL        eval registry key (default smpl_caption_editfix_latest)
#   OUT          output dir (required)
#   GPU          CUDA device index (default 0)
#   MAX_SAMPLES  0=all(1013); else cap (default 40 for pilot)
#   NUM_STEPS    ODE steps (default 50)
#   CFG          text guidance scale (default 2.5)
#   DATAFILE     datalist (default MotionFix instruction datalist)
#   NUM_SHARDS / SHARD_INDEX  sharding (default 1 / 0)
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
PY="$ROOT/.venv_t2m_a100/bin/python"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-smpl_caption_editfix_latest}
OUT=${OUT:?set OUT}
GPU=${GPU:-0}
MAX_SAMPLES=${MAX_SAMPLES:-40}
NUM_STEPS=${NUM_STEPS:-50}
CFG=${CFG:-2.5}
# NOTE: --data-file-override is resolved UNDER EVAL_DATA_DIR (data/eval/m2m_v2),
# so pass the basename only, not a full path.
DATAFILE=${DATAFILE:-eval_motionfix_instruction.json}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}

if [ "$MAX_SAMPLES" = "0" ]; then limarg="--max-samples 1000000"; else limarg="--max-samples $MAX_SAMPLES"; fi
mkdir -p "$OUT/_logs"
echo "[mfix-gen] $(date) MODEL=$MODEL OUT=$OUT GPU=$GPU MAX=$MAX_SAMPLES STEPS=$NUM_STEPS CFG=$CFG shard=$SHARD_INDEX/$NUM_SHARDS"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval/eval_m2m_v2_all_tasks.py \
  --models "$MODEL" --tasks E16 --settings style_edit \
  --data-file-override "$DATAFILE" $limarg --save-npz \
  --num-steps "$NUM_STEPS" --replacement-guidance skip_last \
  --text-guidance-scale "$CFG" \
  --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" \
  --output-dir "$OUT" 2>&1 | tee "$OUT/_logs/gen_shard${SHARD_INDEX}.log"
echo "[mfix-gen-done] $(date) npz=$(find "$OUT" -path '*/npz/*.npz' 2>/dev/null | wc -l)"
