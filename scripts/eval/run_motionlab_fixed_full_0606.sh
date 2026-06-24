#!/usr/bin/env bash
# Full MotionLab inference after fixing unified RFMOTION instruction conditioning.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DATASET=${DATASET:-h3d}  # h3d | mh
OUT_DIR=${OUT_DIR:-outputs/evaluation/motionlab_fixed0606/${DATASET}}
LOGDIR=${LOGDIR:-outputs/evaluation/motionlab_fixed0606/logs_${DATASET}}
GPU_LIST=${GPU_LIST:-0,1}
NUM_SHARDS=${NUM_SHARDS:-2}
STAGE=${STAGE:-eval}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_SAMPLES=${MAX_SAMPLES:-}
EXTRA_INFER_ARGS=${EXTRA_INFER_ARGS:---no-cfg-from-checkpoint --cfg configs/config_rfmotion.yaml}
RUN_NATIVE_EVAL=${RUN_NATIVE_EVAL:-1}
NUM_REPEATS=${NUM_REPEATS:-20}
EVAL_GPU=${EVAL_GPU:-0}

mkdir -p "${OUT_DIR}" "${LOGDIR}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -lt "${NUM_SHARDS}" ]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

common_args=(
  --out-dir "${OUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --stage "${STAGE}"
  --num-shards "${NUM_SHARDS}"
  --skip-existing
)
if [ -n "${MAX_SAMPLES}" ]; then
  common_args+=(--max-samples "${MAX_SAMPLES}")
fi

if [ "${DATASET}" = "mh" ]; then
  common_args+=(
    --anno-file data/annotation/test_motionhub_t2m.json
    --caption-file data/annotation/test_motionhub_t2m_rewritten.json
    --data-dir data/motionhub
  )
elif [ "${DATASET}" != "h3d" ]; then
  echo "Unsupported DATASET=${DATASET}" >&2
  exit 2
fi

echo "[start] dataset=${DATASET} out=${OUT_DIR} shards=${NUM_SHARDS} gpus=${GPU_LIST} stage=${STAGE} $(date)" | tee "${LOGDIR}/run.log"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$shard]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/motionlab_infer_hml3d263.py \
    "${common_args[@]}" \
    --shard-index "${shard}" \
    ${EXTRA_INFER_ARGS} \
    > "${LOGDIR}/infer_s${shard}_gpu${gpu}.log" 2>&1 &
done
wait

count=$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.npy' | wc -l)
echo "[infer done] count=${count} $(date)" | tee -a "${LOGDIR}/run.log"

if [ "${RUN_NATIVE_EVAL}" = "1" ] && [ "${DATASET}" = "h3d" ]; then
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" python3 scripts/eval/eval_momask_native_h3d263.py \
    --recon_root work_dirs/h3d263_eval/h3d263_test_recon_fk \
    --src_h3d272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
    --momask_root ref_repo/Momask/momask-codes \
    --mode pred \
    --pred_dir "${OUT_DIR}" \
    --num_repeats "${NUM_REPEATS}" \
    --drop_mirrored \
    --caption_selection first \
    --output "${LOGDIR}/motionlab_fixed_h3d_native_rep${NUM_REPEATS}.json" \
    > "${LOGDIR}/native_eval.log" 2>&1

  python3 - <<PY | tee "${LOGDIR}/summary.txt"
import json
d = json.load(open("${LOGDIR}/motionlab_fixed_h3d_native_rep${NUM_REPEATS}.json"))
print(
    "h3d_native",
    "samples", d.get("n_samples"),
    "R1", f"{d['r_precision']['mean'][0]:.4f}",
    "R3", f"{d['r_precision']['mean'][2]:.4f}",
    "FID", f"{d['fid']['mean']:.4f}",
    "MM", f"{d['matching_score']['mean']:.4f}",
    "Div", f"{d['diversity']['mean']:.4f}",
)
PY
else
  echo "[skip native eval] DATASET=${DATASET} RUN_NATIVE_EVAL=${RUN_NATIVE_EVAL}" | tee "${LOGDIR}/summary.txt"
fi

touch "${LOGDIR}/_DONE"
echo "[done] $(date)" | tee -a "${LOGDIR}/run.log"
