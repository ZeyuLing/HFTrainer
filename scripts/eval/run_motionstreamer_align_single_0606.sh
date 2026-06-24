#!/bin/bash
# Single-dataset MotionStreamer T2M inference with GT-root/yaw alignment.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "${ROOT}" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "${ROOT}"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export HFTRAINER_SKIP_AUTOREGISTER=${HFTRAINER_SKIP_AUTOREGISTER:-1}

OUT=${OUT:-outputs/evaluation/motionstreamer_align0606_single}
DATASET=${DATASET:-mh}  # h3d | mh
NUM_GPUS=${NUM_GPUS:-8}
TOTAL_SHARDS=${TOTAL_SHARDS:-$NUM_GPUS}
SHARD_BASE=${SHARD_BASE:-0}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
RUN_MOTIONCLIP_EVAL=${RUN_MOTIONCLIP_EVAL:-1}
mkdir -p "$OUT/logs" "$OUT/metrics"

if [ "$DATASET" = "h3d" ]; then
  TAG=h3d
  MS_DATASET=humanml3d
  ANNO=data/annotation/test_hml3d.json
  GEN_DIR="$OUT/h3d_all_npz"
  EXTRA_ARGS=(--humanml3d-protocol all)
elif [ "$DATASET" = "mh" ]; then
  TAG=mh
  MS_DATASET=motionhub
  ANNO=data/annotation/test_motionhub_t2m.json
  GEN_DIR="$OUT/mh_npz"
  EXTRA_ARGS=()
else
  echo "Unknown DATASET=$DATASET" >&2
  exit 2
fi

echo "[start] dataset=$DATASET out=$OUT num_gpus=$NUM_GPUS total_shards=$TOTAL_SHARDS shard_base=$SHARD_BASE"
for i in $(seq 0 $((NUM_GPUS - 1))); do
  shard_index=$((SHARD_BASE + i))
  if [ "$shard_index" -ge "$TOTAL_SHARDS" ]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES=$i python3 scripts/eval/gen_motionstreamer_smpl_npz.py \
    --dataset "$MS_DATASET" \
    --out-dir "$GEN_DIR" \
    --num-shards "$TOTAL_SHARDS" \
    --shard-index "$shard_index" \
    --anno-file "$ANNO" \
    --data-dir data/motionhub \
    --caption-protocol original \
    --align-to-gt-root \
    --align-root-mode yaw \
    --skip-existing \
    "${EXTRA_ARGS[@]}" \
    > "$OUT/logs/${TAG}_gen_${shard_index}of${TOTAL_SHARDS}.log" 2>&1 &
done
wait

echo "[gen done] npz=$(find "$GEN_DIR" -maxdepth 1 -name '*.npz' | wc -l)"
if [ "$RUN_MOTIONCLIP_EVAL" != "1" ]; then
  echo "[done] generation-only RUN_MOTIONCLIP_EVAL=$RUN_MOTIONCLIP_EVAL"
  exit 0
fi
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file "$ANNO" \
  --data_dir data/motionhub \
  --pred_dir "$GEN_DIR" \
  --out_json "$OUT/metrics/${TAG}_align_orig_c64.json" \
  --forward_batch_size 64 \
  --chunk_size "$CHUNK_SIZE" \
  --n_repeats "$N_REPEATS" \
  > "$OUT/logs/eval_${TAG}_motionclip.log" 2>&1

echo "[done] $OUT/metrics/${TAG}_align_orig_c64.json"
