#!/bin/bash
# Single split/condition PRISM TP2M rerun for parallel Table 2 evaluation.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "${ROOT}" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "${ROOT}"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
PY=${PY:-python3}

OUT=${OUT:-outputs/evaluation/prism_tp2m_table2_0606}
DATASET=${DATASET:-h3d}  # h3d | mh
COND=${COND:-1}
NUM_GPUS=${NUM_GPUS:-8}
LOCAL_GPUS=${LOCAL_GPUS:-${NUM_GPUS}}
TOTAL_SHARDS=${TOTAL_SHARDS:-${NUM_GPUS}}
SHARD_OFFSET=${SHARD_OFFSET:-0}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
CHECKPOINT=${CHECKPOINT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_10}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-5.0}
SKIP_EVAL=${SKIP_EVAL:-0}

if [[ "${DATASET}" == "h3d" ]]; then
  TAG=h3d
  ANNO=data/annotation/test_hml3d.json
elif [[ "${DATASET}" == "mh" ]]; then
  TAG=mh
  ANNO=data/annotation/test_motionhub_t2m.json
else
  echo "Unknown DATASET=${DATASET}" >&2
  exit 2
fi

RUN_DIR="${OUT}/${TAG}"
GEN_DIR="${RUN_DIR}/cond${COND}_depth_driven"
mkdir -p "${GEN_DIR}" "${OUT}/logs/${TAG}" "${OUT}/metrics"

echo "[start] dataset=${DATASET} cond=${COND} out=${OUT} local_gpus=${LOCAL_GPUS} total_shards=${TOTAL_SHARDS} shard_offset=${SHARD_OFFSET}"
for i in $(seq 0 $((LOCAL_GPUS - 1))); do
  shard_idx=$((SHARD_OFFSET + i))
  CUDA_VISIBLE_DEVICES=$i "$PY" scripts/eval/eval_prism_tp2m_prefix.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --anno-file "${ANNO}" \
    --data-dir data/motionhub \
    --output-dir "${RUN_DIR}" \
    --condition-num-frames "${COND}" \
    --kafs-mode depth_driven \
    --num-inference-steps "${STEPS}" \
    --guidance-scale "${GUIDANCE}" \
    --min-frames "$((COND + 1))" \
    --max-frames 360 \
    --num-shards "${TOTAL_SHARDS}" \
    --shard-idx "${shard_idx}" \
    --skip-existing \
    > "${OUT}/logs/${TAG}/cond${COND}_gen_s${shard_idx}_of${TOTAL_SHARDS}.log" 2>&1 &
done
wait

echo "[gen done] npz=$(find "${GEN_DIR}" -maxdepth 1 -name '*.npz' | wc -l)"
if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "[skip eval] SKIP_EVAL=1"
  exit 0
fi
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file "${ANNO}" \
  --data_dir data/motionhub \
  --pred_dir "${GEN_DIR}" \
  --out_json "${OUT}/metrics/${TAG}_cond${COND}_prism_c64.json" \
  --forward_batch_size 64 \
  --chunk_size "${CHUNK_SIZE}" \
  --n_repeats "${N_REPEATS}" \
  > "${OUT}/logs/${TAG}/cond${COND}_eval_motionclip.log" 2>&1

echo "[done] ${OUT}/metrics/${TAG}_cond${COND}_prism_c64.json"
