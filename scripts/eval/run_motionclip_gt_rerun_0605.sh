#!/usr/bin/env bash
# Re-run Real(SMPL) controls under both original-caption and rewritten-caption
# protocols. This isolates whether table GT values are being compared against
# a different text protocol from generated methods.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:${PYTHONPATH:-}

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/motionclip_gt_rerun_0605}
CKPT=${CKPT:-checkpoints/motion_clip/motionclip_base_1p_aug_hq}
DATA_DIR=${DATA_DIR:-data/motionhub}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-64}
mkdir -p "$OUT_ROOT/logs"

run_one() {
  local gpu="$1"
  local tag="$2"
  local anno="$3"
  local rewrite="$4"
  local out_dir="$OUT_ROOT/$tag"
  mkdir -p "$out_dir"
  local args=(
    --evaluator_ckpt "$CKPT"
    --anno_file "$anno"
    --data_dir "$DATA_DIR"
    --gt_only
    --out_json "$out_dir/gt_c${CHUNK_SIZE}.json"
    --forward_batch_size "$FORWARD_BATCH_SIZE"
    --chunk_size "$CHUNK_SIZE"
    --n_repeats "$N_REPEATS"
  )
  if [[ "$rewrite" != "-" ]]; then
    args+=(--rewritten_caption_file "$rewrite")
  fi
  echo "[start] $tag gpu=$gpu anno=$anno rewrite=$rewrite $(date -Is)" \
    | tee "$OUT_ROOT/logs/${tag}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    "${args[@]}" >> "$OUT_ROOT/logs/${tag}.log" 2>&1
  echo "[done] $tag $(date -Is)" | tee -a "$OUT_ROOT/logs/${tag}.log"
}

pids=()
run_one 0 h3d_original data/annotation/test_hml3d.json - &
pids+=("$!")
run_one 1 h3d_rewritten data/annotation/test_hml3d.json data/annotation/test_hml3d_rewritten.json &
pids+=("$!")
run_one 2 mh_original data/annotation/test_motionhub_t2m.json - &
pids+=("$!")
run_one 3 mh_rewritten data/annotation/test_motionhub_t2m.json data/annotation/test_motionhub_t2m_rewritten.json &
pids+=("$!")

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done

python3 - <<'PY' | tee "$OUT_ROOT/summary.txt"
import json
from pathlib import Path

root = Path("outputs/evaluation/motionclip_gt_rerun_0605")
for tag in ["h3d_original", "h3d_rewritten", "mh_original", "mh_rewritten"]:
    path = root / tag / "gt_c64.json"
    if not path.exists():
        print(tag, "missing", path)
        continue
    d = json.load(open(path))
    print(
        tag,
        "samples", d.get("samples"),
        "R1", f"{d.get('r_precision_pred_top1_mean'):.4f}",
        "R3", f"{d.get('r_precision_pred_top3_mean'):.4f}",
        "FID", f"{d.get('fid_mean'):.6f}",
        "MM", f"{d.get('mm_dist_pred_mean'):.4f}",
        "Div", f"{d.get('diversity_pred_mean'):.4f}",
    )
PY

exit "$rc"
