#!/usr/bin/env bash
# Retarget one HML3D-263 baseline on HumanML3D only and evaluate with MotionCLIP.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

METHOD=${METHOD:?Set METHOD}
H3D_SRC=${H3D_SRC:?Set H3D_SRC}
EVAL_ROOT=${EVAL_ROOT:-outputs/evaluation/${METHOD}_h3d_only_rw_c64_eval0606}
SMPL_ROOT=${SMPL_ROOT:-outputs/evaluation/${METHOD}_h3d_only_smpl135_0606}
MC135_ROOT=${MC135_ROOT:-outputs/evaluation/${METHOD}_h3d_only_motionclip135_0606}
GPU_LIST=${GPU_LIST:-0,1,3,4,5,6,7}
NUM_SHARDS=${NUM_SHARDS:-7}
LOGDIR="${EVAL_ROOT}/logs"

mkdir -p "${LOGDIR}" "${EVAL_ROOT}/h3d" "${SMPL_ROOT}/h3d" "${MC135_ROOT}/h3d"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -lt "${NUM_SHARDS}" ]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

echo "[start] method=${METHOD} src=${H3D_SRC} shards=${NUM_SHARDS} gpus=${GPU_LIST} $(date)" | tee "${LOGDIR}/run.log"
echo "[source-count] $(find "${H3D_SRC}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)" | tee -a "${LOGDIR}/run.log"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$shard]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "${H3D_SRC}" \
    --out-dir "${SMPL_ROOT}/h3d" \
    --model-dir ref_repo/MDM/body_models \
    --source-fps 20 \
    --target-fps 30 \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard}" \
    --device cuda \
    --batch-size 512 \
    --floor-align \
    --refine-iters 0 \
    --skip-existing \
    > "${LOGDIR}/ik_h3d_s${shard}_gpu${gpu}.log" 2>&1 &
done
wait

echo "[remap] $(date)" | tee -a "${LOGDIR}/run.log"
python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
  --anno-file data/annotation/test_hml3d.json \
  --data-dir data/motionhub \
  --src-dir "${SMPL_ROOT}/h3d" \
  --out-dir "${MC135_ROOT}/h3d" \
  --include-mirrors \
  --key-fallback \
  --align-to-gt-root \
  --overwrite \
  --workers 16 \
  > "${LOGDIR}/remap_h3d.log" 2>&1

echo "[eval] $(date)" | tee -a "${LOGDIR}/run.log"
CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --anno_file data/annotation/test_hml3d.json \
  --data_dir data/motionhub \
  --pred_dir "${MC135_ROOT}/h3d" \
  --rewritten_caption_file data/annotation/test_hml3d_rewritten.json \
  --chunk_size 64 \
  --out_json "${EVAL_ROOT}/h3d/${METHOD}_rw_c64.json" \
  --n_repeats 20 \
  --seed 42 \
  > "${LOGDIR}/eval_h3d.log" 2>&1

python3 - <<PY | tee "${EVAL_ROOT}/summary.txt"
import json
from pathlib import Path
p = Path("${EVAL_ROOT}/h3d/${METHOD}_rw_c64.json")
d = json.load(open(p))
print(
    "h3d",
    "samples", d.get("samples"),
    "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
    "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
    "FID", f"{d.get('fid_mean', float('nan')):.4f}",
    "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
    "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
)
PY

touch "${EVAL_ROOT}/_DONE"
echo "[done] $(date)" | tee -a "${LOGDIR}/run.log"
