#!/usr/bin/env bash
# Postprocess/evaluate the strict official-272 PRISM KAFS/KT comparison.
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

RUN_ROOT="${RUN_ROOT:-outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_kafs_kt_compare_20260621}"
ANNO="${ANNO:-data/annotation/test_hml3d_official272_gtlen.json}"
DATA_DIR="${DATA_DIR:-.}"
WORKERS="${WORKERS:-32}"
MS_DEVICE="${MS_DEVICE:-cuda}"
MC_GPU="${MC_GPU:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-32}"
N_REPEATS="${N_REPEATS:-20}"

CURRENT_ABS="outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_translation_ablation_epoch39_20260621/absolute/h3d/depth_driven"
E39_NONE="$RUN_ROOT/epoch39_no_kafs/h3d/none"
ITER15_NONE="$RUN_ROOT/iter15k_no_kt_no_kafs/h3d/none"

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

require_count() {
  local name="$1" dir="$2" ext="$3" min_count="$4"
  local n
  n="$(count_ext "$dir" "$ext")"
  echo "[coverage] $name $ext=$n dir=$dir" | tee -a "$LOG_DIR/post.log"
  if (( n < min_count )); then
    echo "[error] $name has $n .$ext files, expected at least $min_count" | tee -a "$LOG_DIR/post.log"
    return 1
  fi
}

repack_for_ms_eval() {
  local name="$1"
  local npz_dir="$2"
  local out="$PREP_DIR/$name"
  mkdir -p "$out"
  echo "[repack] $name $npz_dir -> $out" | tee -a "$LOG_DIR/post.log"
  python3 scripts/eval/repack_pred_to_272ids.py \
    --npz-dir "$npz_dir" \
    --anno-file "$ANNO" \
    --id-passthrough \
    --out-dir "$out" \
    --workers "$WORKERS" \
    > "$LOG_DIR/repack_${name}.log" 2>&1
  require_count "prep/$name" "$out" npz 3972
}

convert_for_motionclip() {
  local name="$1"
  local npz_dir="$2"
  local out="$MC_DIR/$name"
  mkdir -p "$out"
  echo "[motionclip-convert] $name $npz_dir -> $out" | tee -a "$LOG_DIR/post.log"
  python3 scripts/eval/convert_smplx_npz_dir_to_135d.py \
    --input-dir "$npz_dir" \
    --output-dir "$out" \
    --skip-existing \
    --progress-every 500 \
    > "$LOG_DIR/motionclip_convert_${name}.log" 2>&1
  require_count "motionclip/$name" "$out" npy 3972
}

run_ms_eval_one() {
  local name="$1"
  local prep="$PREP_DIR/$name"
  local out="$RES_DIR/ms_eval/${name}.json"
  echo "[ms-eval] $name prep=$prep" | tee -a "$LOG_DIR/post.log"
  python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$prep" \
    --tag "prism_${name}" \
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
  printf "epoch39_depth_kafs_abs\t%s\n" "$MC_DIR/epoch39_depth_kafs_abs" >> "$manifest"
  printf "epoch39_no_kafs_abs\t%s\n" "$MC_DIR/epoch39_no_kafs_abs" >> "$manifest"
  printf "iter15k_no_kt_no_kafs_abs\t%s\n" "$MC_DIR/iter15k_no_kt_no_kafs_abs" >> "$manifest"
  echo "[motionclip-eval] manifest=$manifest" | tee -a "$LOG_DIR/post.log"
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

run_distribution_diag() {
  echo "[distribution] full official common set" | tee -a "$LOG_DIR/post.log"
  python3 scripts/eval/diagnose_prism_ms272_distribution_20260621.py \
    --max-ids 0 \
    --out-json "$RES_DIR/distribution_diag_full.json" \
    --method "epoch39_depth_kafs_abs=$PREP_DIR/epoch39_depth_kafs_abs" \
    --method "epoch39_no_kafs_abs=$PREP_DIR/epoch39_no_kafs_abs" \
    --method "iter15k_no_kt_no_kafs_abs=$PREP_DIR/iter15k_no_kt_no_kafs_abs" \
    > "$LOG_DIR/distribution_diag_full.log" 2>&1
}

summarize() {
  python3 - <<'PY' "$RES_DIR"
import json
import sys
from pathlib import Path

res = Path(sys.argv[1])
methods = [
    "epoch39_depth_kafs_abs",
    "epoch39_no_kafs_abs",
    "iter15k_no_kt_no_kafs_abs",
]
rows = ["metric_set\tmethod\tsamples\tR1\tR3\tFID\tMM\tDiv\tpath"]
for name in methods:
    p = res / "ms_eval" / f"{name}.json"
    if p.exists():
        d = json.load(open(p))
        pred = d.get("pred", {})
        rp = pred.get("r_precision", [None, None, None])
        rows.append(
            "MotionStreamer\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
                name,
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
    for name in methods:
        if name not in d:
            continue
        row = d[name]
        rp = row["r_precision_pred"]
        rows.append(
            "MotionCLIP\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
                name,
                row["samples"],
                rp[0], rp[2],
                row["fid_mean"],
                row["mm_dist_pred_mean"],
                row["diversity_pred_mean"],
                mc,
            )
        )
diag = res / "distribution_diag_full.json"
if diag.exists():
    d = json.load(open(diag))
    for name in methods:
        diff = d.get("diff_vs_gt", {}).get(name)
        if not diff:
            continue
        rows.append(
            "Distribution\t{}\t{}\t\t\t\tlocal_rot_mean_abs={:.4f};joint_pos_mean_abs={:.4f};y_mean_m={:.4f}\t\t{}".format(
                name,
                d.get("ids"),
                diff["local_rot6d"]["norm_mean_abs"],
                diff["joint_pos"]["norm_mean_abs"],
                diff["joint_y_offset_m_mean"],
                diag,
            )
        )
out = res / "summary.tsv"
out.write_text("\n".join(rows) + "\n")
print(out.read_text())
PY
}

echo "[post-start] $(date -Is) run_root=$RUN_ROOT" | tee -a "$LOG_DIR/post.log"
require_count "epoch39_depth_kafs_abs/raw" "$CURRENT_ABS" npz 4042
require_count "epoch39_no_kafs_abs/raw" "$E39_NONE" npz 4042
require_count "iter15k_no_kt_no_kafs_abs/raw" "$ITER15_NONE" npz 4042

repack_for_ms_eval epoch39_depth_kafs_abs "$CURRENT_ABS"
repack_for_ms_eval epoch39_no_kafs_abs "$E39_NONE"
repack_for_ms_eval iter15k_no_kt_no_kafs_abs "$ITER15_NONE"

convert_for_motionclip epoch39_depth_kafs_abs "$CURRENT_ABS"
convert_for_motionclip epoch39_no_kafs_abs "$E39_NONE"
convert_for_motionclip iter15k_no_kt_no_kafs_abs "$ITER15_NONE"

run_ms_eval_one epoch39_depth_kafs_abs
run_ms_eval_one epoch39_no_kafs_abs
run_ms_eval_one iter15k_no_kt_no_kafs_abs
run_motionclip_eval
run_distribution_diag
summarize | tee "$LOG_DIR/summary.log"
touch "$RUN_ROOT/_POST_DONE"
echo "[post-done] $(date -Is)" | tee -a "$LOG_DIR/post.log"
