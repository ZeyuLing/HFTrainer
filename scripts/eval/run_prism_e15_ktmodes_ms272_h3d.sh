#!/usr/bin/env bash
# TABLE VIII (KT-RoPE ablation), HumanML3D / MS-272 columns.
# KT-RoPE is parameter-free, so all three position modes are produced from the
# SAME latest checkpoint (kt_spectral_unified epoch_15) by toggling joint_pos_mode
# at inference; KAFS is held at depth_driven (same as "ours") so the only varied
# factor is the joint position encoding.
#   seqInfer -> "Sequential RoPE (no-KT)" row  (+ t2m_compare viewer column)
#   dfsInfer -> "DFS Reindexing" row
#   (Projected Spectral / ours = prism_epoch15_ms272_h3d ours_e15, already done)
# Each mode: generate on 8 GPUs -> repack canon272 row135 -> eval_motionstreamer_272.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

NGPU=${NGPU:-8}
CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_15}
ANNO=data/annotation/test_hml3d.json
REWRITTEN=data/annotation/test_hml3d_rewritten.json
STEPS=50
GUIDANCE=5.0

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/prism_e15_ktmodes_h3d}
PREP="$OUT_ROOT/prep"; LOG="$OUT_ROOT/logs"; RES="$OUT_ROOT/results"
mkdir -p "$PREP" "$LOG" "$RES"
MS_REL="ref_repo/MotionStreamer/MotionStreamer"

echo "[start] $(date) KT-modes ablation ckpt=$CKPT" | tee -a "$LOG/run.log"
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
    > "$LOG/eval_$name.log" 2>&1 || \
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$pred" --tag "$name" --out-json "$oj" >> "$LOG/eval_$name.log" 2>&1 || true
}

gen_mode() {  # tag config out_subdir
  local tag="$1" cfg="$2" out="$3"
  mkdir -p "$out/_logs"
  echo "[gen $tag] $(date) cfg=$cfg -> $out/depth_driven" | tee -a "$LOG/run.log"
  pids=()
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/eval_prism_kafs_ablation.py \
      --config "$cfg" --checkpoint "$CKPT" --kafs-mode depth_driven \
      --anno-file "$ANNO" --rewritten-caption-file "$REWRITTEN" \
      --data-dir data/motionhub --output-dir "$out" \
      --num-inference-steps $STEPS --guidance-scale $GUIDANCE \
      --num-shards $NGPU --shard-idx $g --skip-existing \
      > "$out/_logs/${tag}_shard${g}of${NGPU}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "$out/depth_driven"
}

# ---- Sequential RoPE (no-KT) ----
gd_seq="$(gen_mode seq configs/prism/prism_1b_tp2m_multiframe_kt_seqInfer.py outputs/evaluation/prism_e15_ktmodes_h3d/gen_seq)"
n=$(ls "$gd_seq"/*.npz 2>/dev/null | wc -l); echo "[gen seq done] n=$n" | tee -a "$LOG/run.log"
p_seq="$(repack_one ours_e15_seq "$gd_seq")"
eval_one ours_e15_seq "$p_seq" 0

# ---- DFS Reindexing ----
gd_dfs="$(gen_mode dfs configs/prism/prism_1b_tp2m_multiframe_kt_dfsInfer.py outputs/evaluation/prism_e15_ktmodes_h3d/gen_dfs)"
n=$(ls "$gd_dfs"/*.npz 2>/dev/null | wc -l); echo "[gen dfs done] n=$n" | tee -a "$LOG/run.log"
p_dfs="$(repack_one ours_e15_dfs "$gd_dfs")"
eval_one ours_e15_dfs "$p_dfs" 0

echo "[results]" | tee -a "$LOG/run.log"
for j in "$RES"/*.json; do [ -s "$j" ] && echo "  $j" | tee -a "$LOG/run.log"; done
touch "$OUT_ROOT/_DONE"
echo "[done] $(date)" | tee -a "$LOG/run.log"
