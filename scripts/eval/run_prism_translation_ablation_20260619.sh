#!/usr/bin/env bash
# Paired PRISM translation-decoding ablation for Table-1 HumanML3D official ids.
#
# rollout:  use current PRISM behavior, first absolute translation plus cumsum
#           of subsequent relative deltas.
# absolute: use the decoded absolute translation channels directly.
#
# Stages:
#   STAGE=gen_abs   generate only the absolute-translation SMPL NPZs.
#   STAGE=post      convert/repack both rollout and absolute, then run metrics.
#   STAGE=all       run gen_abs followed by post on this host.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HFTRAINER_SKIP_AUTOREGISTER="${HFTRAINER_SKIP_AUTOREGISTER:-1}"

STAGE="${STAGE:-post}"
RUN_ROOT="${RUN_ROOT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_translation_ablation_20260619}"
ANNO="${ANNO:-data/annotation/test_hml3d_official272_gtlen.json}"
DATA_DIR="${DATA_DIR:-.}"
CONFIG="${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}"
CKPT="${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_31}"
KAFS_MODE="${KAFS_MODE:-depth_driven}"
SEED="${SEED:-42}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-5.0}"
SMOOTH_OUTPUT="${SMOOTH_OUTPUT:-1}"
SKIP_MOTION_EXISTENCE_CHECK="${SKIP_MOTION_EXISTENCE_CHECK:-1}"
MIN_FRAMES="${MIN_FRAMES:-24}"
MAX_FRAMES="${MAX_FRAMES:-360}"
TOTAL_SHARDS="${TOTAL_SHARDS:-64}"
SHARD_BASE="${SHARD_BASE:-0}"
NGPU="${NGPU:-${TJ_GPU_NUM:-8}}"
WORKERS="${WORKERS:-32}"
MS_DEVICE="${MS_DEVICE:-cuda}"
MC_GPU="${MC_GPU:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-32}"
N_REPEATS="${N_REPEATS:-20}"

# Existing current-rollout full run. Keep this fixed so the ablation isolates
# translation decode only for the same PRISM checkpoint/config/protocol family.
ROLLOUT_NPZ_DIR="${ROLLOUT_NPZ_DIR:-outputs/evaluation/t2m/humanml3d_official_test/ms272/prism_epoch31_smooth_exactlen_0617_vermo/h3d/depth_driven}"
ABS_OUT="${ABS_OUT:-$RUN_ROOT/absolute/h3d}"
ABS_NPZ_DIR="$ABS_OUT/$KAFS_MODE"

PREP_DIR="$RUN_ROOT/prep"
MC_DIR="$RUN_ROOT/motionclip135"
RES_DIR="$RUN_ROOT/results"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$PREP_DIR" "$MC_DIR" "$RES_DIR/ms_eval" "$RES_DIR/motionclip" "$LOG_DIR"

count_ext() {
  local dir="$1" ext="$2"
  [[ -d "$dir" ]] || { echo 0; return; }
  find "$dir" -maxdepth 1 -name "*.$ext" | wc -l
}

require_npz_count() {
  local name="$1" dir="$2" min_count="$3"
  local n
  n="$(count_ext "$dir" npz)"
  echo "[coverage] $name npz=$n dir=$dir" | tee -a "$LOG_DIR/run.log"
  if (( n < min_count )); then
    echo "[error] $name has $n npz files, expected at least $min_count" | tee -a "$LOG_DIR/run.log"
    return 1
  fi
}

gen_abs() {
  echo "[gen_abs] $(date -Is) out=$ABS_OUT shards=$TOTAL_SHARDS base=$SHARD_BASE ngpu=$NGPU" | tee -a "$LOG_DIR/run.log"
  CONFIG="$CONFIG" \
  CKPT="$CKPT" \
  KAFS_MODE="$KAFS_MODE" \
  OUT="$ABS_OUT" \
  OUT_SUBDIR="$KAFS_MODE" \
  ANNO="$ANNO" \
  REWRITTEN="" \
  DATA_DIR="$DATA_DIR" \
  STEPS="$STEPS" \
  GUIDANCE="$GUIDANCE" \
  SEED="$SEED" \
  SMOOTH_OUTPUT="$SMOOTH_OUTPUT" \
  SKIP_MOTION_EXISTENCE_CHECK="$SKIP_MOTION_EXISTENCE_CHECK" \
  MIN_FRAMES="$MIN_FRAMES" \
  MAX_FRAMES="$MAX_FRAMES" \
  TOTAL_SHARDS="$TOTAL_SHARDS" \
  SHARD_BASE="$SHARD_BASE" \
  NGPU="$NGPU" \
  TRANSLATION_DECODE_MODE=absolute \
  bash scripts/eval/run_prism_genonly_param.sh 2>&1 | tee "$LOG_DIR/gen_abs_${SHARD_BASE}_of_${TOTAL_SHARDS}.log"
}

repack_for_ms_eval() {
  local name="$1"
  local npz_dir="$2"
  local out="$PREP_DIR/$name"
  mkdir -p "$out"
  echo "[repack] $name $npz_dir -> $out" | tee -a "$LOG_DIR/run.log"
  python3 scripts/eval/repack_pred_to_272ids.py \
    --npz-dir "$npz_dir" \
    --anno-file "$ANNO" \
    --id-passthrough \
    --out-dir "$out" \
    --workers "$WORKERS" \
    > "$LOG_DIR/repack_${name}.log" 2>&1
  require_npz_count "prep/$name" "$out" 3972
}

convert_for_motionclip() {
  local name="$1"
  local npz_dir="$2"
  local out="$MC_DIR/$name"
  mkdir -p "$out"
  echo "[motionclip-convert] $name $npz_dir -> $out" | tee -a "$LOG_DIR/run.log"
  python3 scripts/eval/convert_smplx_npz_dir_to_135d.py \
    --input-dir "$npz_dir" \
    --output-dir "$out" \
    --skip-existing \
    --progress-every 500 \
    > "$LOG_DIR/motionclip_convert_${name}.log" 2>&1
  local n
  n="$(count_ext "$out" npy)"
  echo "[coverage] motionclip/$name npy=$n dir=$out" | tee -a "$LOG_DIR/run.log"
  if (( n < 3972 )); then
    echo "[error] motionclip/$name has $n npy files, expected at least 3972" | tee -a "$LOG_DIR/run.log"
    return 1
  fi
}

run_ms_eval_one() {
  local name="$1"
  local prep="$PREP_DIR/$name"
  local out="$RES_DIR/ms_eval/${name}.json"
  echo "[ms-eval] $name prep=$prep" | tee -a "$LOG_DIR/run.log"
  python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$prep" \
    --tag "prism_${name}_translation" \
    --also-refk \
    --min-motion-len 60 \
    --max-motion-length 300 \
    --device "$MS_DEVICE" \
    --out-json "$out" \
    > "$LOG_DIR/ms_eval_${name}.log" 2>&1
}

run_motionclip_eval() {
  local manifest="$RUN_ROOT/pred_manifest.tsv"
  : > "$manifest"
  printf "rollout\t%s\n" "$MC_DIR/rollout" >> "$manifest"
  printf "absolute\t%s\n" "$MC_DIR/absolute" >> "$manifest"
  echo "[motionclip-eval] manifest=$manifest" | tee -a "$LOG_DIR/run.log"
  CUDA_VISIBLE_DEVICES="$MC_GPU" python3 scripts/eval/eval_motionclip_table1_dirs.py \
    --evaluator-ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno-file "$ANNO" \
    --data-dir "$DATA_DIR" \
    --real-dir outputs/evaluation/t2m/humanml3d_official_test/motionclip_table1_20260619/motionclip135/real \
    --pred-manifest "$manifest" \
    --out-dir "$RES_DIR/motionclip" \
    --min-frames 60 \
    --max-frames 300 \
    --chunk-size "$CHUNK_SIZE" \
    --forward-batch-size "$FORWARD_BATCH_SIZE" \
    --n-repeats "$N_REPEATS" \
    --seed 0 \
    > "$LOG_DIR/motionclip_eval.log" 2>&1
}

summarize() {
  python3 - <<'PY' "$RES_DIR"
import json
import sys
from pathlib import Path

res = Path(sys.argv[1])
rows = ["metric_set\tmode\tsamples\tR1\tR3\tFID\tMM\tDiv\tpath"]
for mode in ("rollout", "absolute"):
    p = res / "ms_eval" / f"{mode}.json"
    if p.exists():
        d = json.load(open(p))
        pred = d.get("pred", {})
        rp = pred.get("r_precision", [None, None, None])
        rows.append(
            "MotionStreamer\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
                mode,
                pred.get("nb", d.get("ids_with_required_files")),
                rp[0], rp[2],
                pred.get("fid_vs_gt_native", float("nan")),
                pred.get("matching_score", float("nan")),
                pred.get("diversity", float("nan")),
                p,
            )
        )
mc = res / "motionclip" / "summary.json"
if mc.exists():
    d = json.load(open(mc))
    for mode, row in d.items():
        rp = row["r_precision_pred"]
        rows.append(
            "MotionCLIP\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
                mode,
                row["samples"],
                rp[0], rp[2],
                row["fid_mean"],
                row["mm_dist_pred_mean"],
                row["diversity_pred_mean"],
                mc,
            )
        )
out = res / "summary.tsv"
out.write_text("\n".join(rows) + "\n")
print(out.read_text())
PY
}

post() {
  echo "[post] $(date -Is) run_root=$RUN_ROOT" | tee -a "$LOG_DIR/run.log"
  require_npz_count rollout "$ROLLOUT_NPZ_DIR" 4042
  require_npz_count absolute "$ABS_NPZ_DIR" 4042

  repack_for_ms_eval rollout "$ROLLOUT_NPZ_DIR"
  repack_for_ms_eval absolute "$ABS_NPZ_DIR"

  convert_for_motionclip rollout "$ROLLOUT_NPZ_DIR"
  convert_for_motionclip absolute "$ABS_NPZ_DIR"

  run_ms_eval_one rollout
  run_ms_eval_one absolute
  run_motionclip_eval
  summarize | tee "$LOG_DIR/summary.log"
  touch "$RUN_ROOT/_DONE"
}

echo "[start] stage=$STAGE run_root=$RUN_ROOT $(date -Is)" | tee -a "$LOG_DIR/run.log"
case "$STAGE" in
  gen_abs)
    gen_abs
    ;;
  post)
    post
    ;;
  all)
    gen_abs
    post
    ;;
  *)
    echo "unknown STAGE=$STAGE (expected gen_abs|post|all)" >&2
    exit 2
    ;;
esac
echo "[done] stage=$STAGE $(date -Is)" | tee -a "$LOG_DIR/run.log"
