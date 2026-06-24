#!/bin/bash
# Full rerun for Table 2 PRISM prefix-pose-conditioned generation.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "${ROOT}" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "${ROOT}"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
PY=${PY:-python3}

OUT=${OUT:-outputs/evaluation/prism_tp2m_table2_0606}
NUM_GPUS=${NUM_GPUS:-8}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
CHECKPOINT=${CHECKPOINT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_10}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-5.0}

mkdir -p "$OUT/logs" "$OUT/metrics"
echo "[start] out=$OUT ckpt=$CHECKPOINT gpus=$NUM_GPUS"

run_dataset() {
  local tag=$1
  local anno=$2
  local max_frames=$3
  local cond=$4
  local out_dir="$OUT/$tag"
  mkdir -p "$out_dir" "$OUT/logs/$tag"

  for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i "$PY" scripts/eval/eval_prism_tp2m_prefix.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --anno-file "$anno" \
      --data-dir data/motionhub \
      --output-dir "$out_dir" \
      --condition-num-frames "$cond" \
      --kafs-mode depth_driven \
      --num-inference-steps "$STEPS" \
      --guidance-scale "$GUIDANCE" \
      --min-frames "$((cond + 1))" \
      --max-frames "$max_frames" \
      --num-shards "$NUM_GPUS" \
      --shard-idx "$i" \
      --skip-existing \
      > "$OUT/logs/$tag/cond${cond}_gen_$i.log" 2>&1 &
  done
  wait
  echo "[$tag cond=$cond gen done] npz=$(find "$out_dir/cond${cond}_depth_driven" -maxdepth 1 -name '*.npz' | wc -l)"

  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "$anno" \
    --data_dir data/motionhub \
    --pred_dir "$out_dir/cond${cond}_depth_driven" \
    --out_json "$OUT/metrics/${tag}_cond${cond}_prism_c64.json" \
    --forward_batch_size 64 \
    --chunk_size "$CHUNK_SIZE" \
    --n_repeats "$N_REPEATS" \
    > "$OUT/logs/$tag/cond${cond}_eval_motionclip.log" 2>&1
}

eval_real() {
  local tag=$1
  local anno=$2
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "$anno" \
    --data_dir data/motionhub \
    --gt_only \
    --out_json "$OUT/metrics/${tag}_real_c64.json" \
    --forward_batch_size 64 \
    --chunk_size "$CHUNK_SIZE" \
    --n_repeats "$N_REPEATS" \
    > "$OUT/logs/${tag}_real_eval.log" 2>&1
}

eval_real h3d data/annotation/test_hml3d.json
eval_real mh data/annotation/test_motionhub_t2m.json

for cond in 1 5 9; do
  run_dataset h3d data/annotation/test_hml3d.json 360 "$cond"
done
for cond in 1 5 9; do
  run_dataset mh data/annotation/test_motionhub_t2m.json 360 "$cond"
done

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/evaluation/prism_tp2m_table2_0606/metrics")
for path in sorted(root.glob("*.json")):
    d = json.loads(path.read_text())
    print(path.name, {
        "samples": d.get("samples"),
        "r3": d.get("r_precision_pred_top3_mean") if not d.get("gt_only") else d.get("r_precision_real_top3_mean"),
        "fid": d.get("fid_mean"),
        "mm": d.get("mm_dist_pred_mean") if not d.get("gt_only") else d.get("mm_dist_real_mean"),
        "div": d.get("diversity_pred_mean") if not d.get("gt_only") else d.get("diversity_real_mean"),
    })
PY

echo "[done]"
