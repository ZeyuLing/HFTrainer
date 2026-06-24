#!/usr/bin/env bash
# Generate OLD PRISM (no KT-spectral, checkpoint-iter_15000) T2M predictions on
# the HumanML3D test set and evaluate with the MotionStreamer Evaluator_272, so
# the qualitative compare site (and tables, if wanted) can show the pre-KT model
# next to the current epoch15 model under the SAME inference recipe.
#
# Same recipe as run_prism_epoch15_ms272_h3d.sh PHASE=t2m, only the config +
# checkpoint differ (sequential joint_pos_mode, iter-15000 weights).
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

NGPU=${NGPU:-8}
CONFIG=configs/prism/prism_1b_tp2m_multiframe.py
CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000}
ANNO=data/annotation/test_hml3d.json
REWRITTEN=data/annotation/test_hml3d_rewritten.json
STEPS=50
GUIDANCE=5.0
KAFS=${KAFS:-depth_driven}

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/prism_old15k_ms272_h3d}
PREP="$OUT_ROOT/prep"; LOG="$OUT_ROOT/logs"; RES="$OUT_ROOT/results"
mkdir -p "$PREP" "$LOG" "$RES"

MS_REL="ref_repo/MotionStreamer/MotionStreamer"
GT272_DIR="$MS_REL/humanml3d_272/motion_data"

echo "[start] $(date) OLD PRISM ckpt=$CKPT kafs=$KAFS" | tee -a "$LOG/run.log"

bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true
if [ ! -f /dev/shm/eval272_epoch99.ckpt ]; then
  cp "$MS_REL/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt" /dev/shm/eval272_epoch99.ckpt 2>/dev/null || true
fi

repack_one() {  # name src_dir
  local name="$1" src="$2" dst="$PREP/$1"
  if [ -f "$dst/_DONE" ]; then echo "$dst"; return 0; fi
  mkdir -p "$dst"
  python3 scripts/eval/repack_pred_to_272ids.py --npz-dir "$src" \
    --anno-file "$ANNO" --out-dir "$dst" --workers 16 \
    > "$LOG/repack_$name.log" 2>&1 && touch "$dst/_DONE"
  echo "$dst"
}
eval_one() {  # name pred_dir gpu
  local name="$1" pred="$2" gpu="$3" oj="$RES/$1.json"
  [ -s "$oj" ] && return 0
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$pred" --tag "$name" --also-refk --out-json "$oj" \
    > "$LOG/eval_$name.log" 2>&1
  if [ ! -s "$oj" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
      --pred-dir "$pred" --tag "$name" --out-json "$oj" >> "$LOG/eval_$name.log" 2>&1 || true
  fi
}

gen_t2m() {
  local out="outputs/evaluation/prism_old15k_rw/h3d"
  mkdir -p "$out/_logs"
  echo "[gen-t2m] $(date) -> $out/$KAFS" | tee -a "$LOG/run.log"
  pids=()
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/eval_prism_kafs_ablation.py \
      --config "$CONFIG" --checkpoint "$CKPT" --kafs-mode "$KAFS" \
      --anno-file "$ANNO" --rewritten-caption-file "$REWRITTEN" \
      --data-dir data/motionhub --output-dir "$out" \
      --num-inference-steps $STEPS --guidance-scale $GUIDANCE \
      --num-shards $NGPU --shard-idx $g --skip-existing \
      > "$out/_logs/${KAFS}_shard${g}of${NGPU}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "$out/$KAFS"
}

gd="$(gen_t2m)"
n=$(ls "$gd"/*.npz 2>/dev/null | wc -l); echo "[gen-t2m done] n=$n" | tee -a "$LOG/run.log"
p="$(repack_one ours_old15k "$gd")"
eval_one ours_old15k "$p" 0

echo "[results]" | tee -a "$LOG/run.log"
for j in "$RES"/*.json; do [ -s "$j" ] && echo "  $j" | tee -a "$LOG/run.log"; done
touch "$OUT_ROOT/_DONE_t2m"
echo "[done] $(date)" | tee -a "$LOG/run.log"
