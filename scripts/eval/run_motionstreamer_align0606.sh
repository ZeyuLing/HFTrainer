#!/bin/bash
# Rerun MotionStreamer T2M inference with GT-root/yaw alignment before SMPL
# MotionCLIP evaluation. Intended for Taiji nodes.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:${PYTHONPATH:-}

OUT=${OUT:-outputs/evaluation/motionstreamer_align0606}
NUM_GPUS=${NUM_GPUS:-8}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
mkdir -p "$OUT/logs" "$OUT/metrics"

echo "[start] out=$OUT num_gpus=$NUM_GPUS"

for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i python3 scripts/eval/gen_motionstreamer_smpl_npz.py \
    --dataset humanml3d \
    --out-dir "$OUT/h3d_all_npz" \
    --num-shards "$NUM_GPUS" \
    --shard-index "$i" \
    --anno-file data/annotation/test_hml3d.json \
    --data-dir data/motionhub \
    --humanml3d-protocol all \
    --caption-protocol original \
    --align-to-gt-root \
    --align-root-mode yaw \
    --skip-existing \
    > "$OUT/logs/h3d_gen_$i.log" 2>&1 &
done
wait
echo "[h3d gen done] npz=$(find "$OUT/h3d_all_npz" -maxdepth 1 -name '*.npz' | wc -l)"

for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i python3 scripts/eval/gen_motionstreamer_smpl_npz.py \
    --dataset motionhub \
    --out-dir "$OUT/mh_npz" \
    --num-shards "$NUM_GPUS" \
    --shard-index "$i" \
    --anno-file data/annotation/test_motionhub_t2m.json \
    --data-dir data/motionhub \
    --caption-protocol original \
    --align-to-gt-root \
    --align-root-mode yaw \
    --skip-existing \
    > "$OUT/logs/mh_gen_$i.log" 2>&1 &
done
wait
echo "[mh gen done] npz=$(find "$OUT/mh_npz" -maxdepth 1 -name '*.npz' | wc -l)"

CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file data/annotation/test_hml3d.json \
  --data_dir data/motionhub \
  --pred_dir "$OUT/h3d_all_npz" \
  --out_json "$OUT/metrics/h3d_all_align_orig_c64.json" \
  --forward_batch_size 64 \
  --chunk_size "$CHUNK_SIZE" \
  --n_repeats "$N_REPEATS" \
  > "$OUT/logs/eval_h3d_motionclip.log" 2>&1

CUDA_VISIBLE_DEVICES=1 python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file data/annotation/test_motionhub_t2m.json \
  --data_dir data/motionhub \
  --pred_dir "$OUT/mh_npz" \
  --out_json "$OUT/metrics/mh_align_orig_c64.json" \
  --forward_batch_size 64 \
  --chunk_size "$CHUNK_SIZE" \
  --n_repeats "$N_REPEATS" \
  > "$OUT/logs/eval_mh_motionclip.log" 2>&1

python3 - <<'PY'
import json
from pathlib import Path

root = Path("outputs/evaluation/motionstreamer_align0606/metrics")
for name in ["h3d_all_align_orig_c64.json", "mh_align_orig_c64.json"]:
    p = root / name
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    print(name, {
        "samples": d.get("samples"),
        "r1": d.get("r_precision_pred_top1_mean"),
        "r3": d.get("r_precision_pred_top3_mean"),
        "fid": d.get("fid_mean"),
        "mm": d.get("mm_dist_pred_mean"),
        "div": d.get("diversity_pred_mean"),
    })
PY

echo "[done]"
