#!/usr/bin/env bash
# Generate PRISM (epoch_15, latest ckpt) T2M predictions on the HumanML3D test
# set with KAFS DISABLED (--kafs-mode none), then repack -> canon272 row135 ->
# MS-272 eval. Purpose:
#   * item5: add a "no-KAFS" column to the t2m_compare viewer (checkpoint stays
#            latest e15; only the inference-time KAFS scaling is turned off).
#   * item3: native-272 metrics for KAFS none vs depth_driven on the FULL set
#            (the earlier 64-sample diag was too small for stable FID/R).
# Mirrors run_prism_epoch15_ms272_h3d.sh PHASE=t2m; only kafs-mode + out dirs differ.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

NGPU=${NGPU:-8}
CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py
CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_15}
ANNO=data/annotation/test_hml3d.json
REWRITTEN=data/annotation/test_hml3d_rewritten.json
STEPS=50
GUIDANCE=5.0

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/prism_epoch15_ms272_h3d}
PREP="$OUT_ROOT/prep"; LOG="$OUT_ROOT/logs"; RES="$OUT_ROOT/results"
mkdir -p "$PREP" "$LOG" "$RES"
MS_REL="ref_repo/MotionStreamer/MotionStreamer"

echo "[start] $(date) KAFS=none ckpt=$CKPT" | tee -a "$LOG/run_none.log"

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

# --- generation: T2M kafs=none (8 shards over 8 GPUs) -------------------------
out="outputs/evaluation/prism_kt_spectral_epoch15_rw/h3d"
mkdir -p "$out/_logs"
echo "[gen-t2m-none] $(date) -> $out/none" | tee -a "$LOG/run_none.log"
pids=()
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/eval_prism_kafs_ablation.py \
    --config "$CONFIG" --checkpoint "$CKPT" --kafs-mode none \
    --anno-file "$ANNO" --rewritten-caption-file "$REWRITTEN" \
    --data-dir data/motionhub --output-dir "$out" \
    --num-inference-steps $STEPS --guidance-scale $GUIDANCE \
    --num-shards $NGPU --shard-idx $g --skip-existing \
    > "$out/_logs/none_shard${g}of${NGPU}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
gd="$out/none"
n=$(ls "$gd"/*.npz 2>/dev/null | wc -l); echo "[gen done] none n=$n" | tee -a "$LOG/run_none.log"

p="$(repack_one ours_e15_none "$gd")"
eval_one ours_e15_none "$p" 0

echo "[results]" | tee -a "$LOG/run_none.log"
ls -la "$RES"/ours_e15_none.json 2>/dev/null | tee -a "$LOG/run_none.log"
touch "$OUT_ROOT/_DONE_none"
echo "[done] $(date)" | tee -a "$LOG/run_none.log"
