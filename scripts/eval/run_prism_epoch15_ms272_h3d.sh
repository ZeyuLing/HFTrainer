#!/usr/bin/env bash
# Generate PRISM (latest checkpoint = epoch_31) predictions on the HumanML3D
# test set and evaluate them with the MotionStreamer Evaluator_272, so the
# paper's "ours" rows use the up-to-date checkpoint under the public benchmark.
#
# PHASE=t2m   -> T2M depth_driven  (Table 1 ours)
# PHASE=tp2m  -> TP2M cond1/5/9    (Table 2 ours)
#
# Both phases: generate on 8 GPUs (8 shards), then repack -> canon272 row135 ->
# eval_motionstreamer_272.py (native + refk FID). Reuses the exact recipes from
# run_gen_node.sh and run_prism_tp2m_single_0606.sh; only the checkpoint/out dirs
# change. --skip-existing makes the job resumable after preemption.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

PHASE=${PHASE:-t2m}
NGPU=${NGPU:-8}
CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py
CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_31}
ANNO=${ANNO:-data/annotation/test_hml3d.json}
REWRITTEN=${REWRITTEN:-data/annotation/test_hml3d_rewritten.json}
STEPS=50
GUIDANCE=5.0
SMOOTH_OUTPUT=${SMOOTH_OUTPUT:-0}
RUN_PHYS=${RUN_PHYS:-0}
SKIP_CACHE=${SKIP_CACHE:-0}

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/prism_epoch31_ms272_h3d}
PREP="$OUT_ROOT/prep"; LOG="$OUT_ROOT/logs"; RES="$OUT_ROOT/results"
mkdir -p "$PREP" "$LOG" "$RES"

MS_REL="ref_repo/MotionStreamer/MotionStreamer"
GT272_DIR="$MS_REL/humanml3d_272/motion_data"

echo "[start] $(date) phase=$PHASE ckpt=$CKPT smooth=$SMOOTH_OUTPUT" | tee -a "$LOG/run.log"

# --- optionally cache evaluator ckpt + GT/text to /dev/shm ---------------------
if [ "$SKIP_CACHE" = "1" ]; then
  echo "[cache] skipped" > "$LOG/cache.log"
else
  bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true
  if [ ! -f /dev/shm/eval272_epoch99.ckpt ]; then
    cp "$MS_REL/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt" /dev/shm/eval272_epoch99.ckpt 2>/dev/null || true
  fi
fi

# --- repack one method to canonical-id row135 npz (npz kind = SMPLX, anno ids) -
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
phys_one() {  # name pred_dir
  local name="$1" pred="$2" oj="$RES/phys_$1.json"
  [ "$RUN_PHYS" = "1" ] || return 0
  [ -s "$oj" ] && return 0
  python3 scripts/eval/compute_phys_h3d.py --m135-dir "$pred" --tag "$name" \
    --workers 16 --out-json "$oj" > "$LOG/phys_$name.log" 2>&1 || true
}
poseq_one() {  # name pred_dir gpu
  local name="$1" pred="$2" gpu="$3" oj="$RES/poseq_$1.json"
  [ "$RUN_PHYS" = "1" ] || return 0
  [ -s "$oj" ] && return 0
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/compute_pose_quality_h3d.py \
    --m135-dir "$pred" --tag "$name" --out-json "$oj" \
    > "$LOG/poseq_$name.log" 2>&1 || true
}

# --- generation: T2M depth_driven (8 shards over 8 GPUs) -----------------------
gen_t2m() {
  if [ -n "${T2M_GEN_DIR:-}" ]; then
    echo "[gen-t2m skip] using existing T2M_GEN_DIR=$T2M_GEN_DIR" | tee -a "$LOG/run.log" >&2
    echo "$T2M_GEN_DIR"
    return 0
  fi
  local out="${GEN_OUT:-outputs/evaluation/prism_kt_spectral_epoch31_rw/h3d}"
  local smooth_args=()
  if [ "$SMOOTH_OUTPUT" = "1" ]; then smooth_args=(--smooth-output); fi
  mkdir -p "$out/_logs"
  echo "[gen-t2m] $(date) -> $out/depth_driven" | tee -a "$LOG/run.log" >&2
  pids=()
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/eval_prism_kafs_ablation.py \
      --config "$CONFIG" --checkpoint "$CKPT" --kafs-mode depth_driven \
      --anno-file "$ANNO" --rewritten-caption-file "$REWRITTEN" \
      --data-dir data/motionhub --output-dir "$out" \
      --num-inference-steps $STEPS --guidance-scale $GUIDANCE \
      --num-shards $NGPU --shard-idx $g --skip-existing "${smooth_args[@]}" \
      > "$out/_logs/depth_driven_shard${g}of${NGPU}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "$out/depth_driven"
}

# --- generation: TP2M one condition (8 shards over 8 GPUs) ---------------------
gen_tp2m_cond() {  # cond
  local cond="$1"
  local rundir="outputs/evaluation/prism_tp2m_epoch31_0617/h3d"
  local gendir="$rundir/cond${cond}_depth_driven"
  mkdir -p "$gendir" "$rundir/_logs"
  echo "[gen-tp2m c$cond] $(date) -> $gendir" | tee -a "$LOG/run.log" >&2
  pids=()
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/eval_prism_tp2m_prefix.py \
      --config "$CONFIG" --checkpoint "$CKPT" \
      --anno-file "$ANNO" --data-dir data/motionhub --output-dir "$rundir" \
      --condition-num-frames "$cond" --kafs-mode depth_driven \
      --num-inference-steps $STEPS --guidance-scale $GUIDANCE \
      --min-frames "$((cond + 1))" --max-frames 360 \
      --num-shards $NGPU --shard-idx $g --skip-existing \
      > "$rundir/_logs/cond${cond}_s${g}of${NGPU}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "$gendir"
}

if [ "$PHASE" = "t2m" ]; then
  gd="$(gen_t2m)"
  n=$(ls "$gd"/*.npz 2>/dev/null | wc -l); echo "[gen-t2m done] n=$n" | tee -a "$LOG/run.log"
  expected=$(python3 - "$ANNO" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
if isinstance(data, dict) and isinstance(data.get("data_list"), (list, dict)):
    print(len(data["data_list"]))
elif isinstance(data, list):
    print(len(data))
else:
    print(len(data))
PY
)
  if [ "$n" -ne "$expected" ]; then
    echo "[gen-t2m error] generated n=$n but annotation expected=$expected anno=$ANNO" | tee -a "$LOG/run.log" >&2
    exit 3
  fi
  tag="${T2M_TAG:-ours_e31}"
  p="$(repack_one "$tag" "$gd")"
  eval_one "$tag" "$p" 0
  phys_one "$tag" "$p"
  poseq_one "$tag" "$p" 0
elif [ "$PHASE" = "tp2m" ]; then
  for c in 1 5 9; do
    gd="$(gen_tp2m_cond "$c")"
    n=$(ls "$gd"/*.npz 2>/dev/null | wc -l); echo "[gen-tp2m c$c done] n=$n" | tee -a "$LOG/run.log"
    p="$(repack_one "ours_e31_c$c" "$gd")"
    eval_one "ours_e31_c$c" "$p" 0
  done
else
  echo "Unknown PHASE=$PHASE" >&2; exit 2
fi

echo "[results]" | tee -a "$LOG/run.log"
for j in "$RES"/*.json; do [ -s "$j" ] && echo "  $j" | tee -a "$LOG/run.log"; done
touch "$OUT_ROOT/_DONE_${PHASE}"
echo "[done] $(date) phase=$PHASE" | tee -a "$LOG/run.log"
