#!/usr/bin/env bash
# Evaluate already-remapped HML3D-263 -> SMPL predictions against the original
# annotation captions. Generation may still use rewritten prompts; this script
# only changes the evaluator text protocol.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SRC_ROOT=${SRC_ROOT:-outputs/evaluation/aligned_hml263_baselines_0605_seq/motionclip135}
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/aligned_hml263_baselines_0605_original_eval}
LOGDIR="${OUT_ROOT}/logs"
mkdir -p "${LOGDIR}" "${OUT_ROOT}/h3d" "${OUT_ROOT}/mh"

METHODS=(${METHODS:-motiongpt3 mld momask mdm t2mgpt motiongpt})
EVAL_PARALLEL=${EVAL_PARALLEL:-4}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}

H3D_ANNO="data/annotation/test_hml3d.json"
MH_ANNO="data/annotation/test_motionhub_t2m.json"

eval_one() {
  local split="$1"
  local method="$2"
  local anno="$3"
  local gpu="$4"
  local pred="${SRC_ROOT}/${split}/${method}"
  if [[ ! -d "${pred}" ]]; then
    echo "[skip] ${split}/${method}: missing ${pred}" | tee -a "${LOGDIR}/run.log"
    return 0
  fi
  echo "[eval-original] ${split}/${method} gpu=${gpu} $(date)" | tee -a "${LOGDIR}/run.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${pred}" \
    --chunk_size "${CHUNK_SIZE}" \
    --out_json "${OUT_ROOT}/${split}/${method}_aligned_orig_c${CHUNK_SIZE}.json" \
    --n_repeats "${N_REPEATS}" \
    --seed 42 \
    > "${LOGDIR}/eval_${split}_${method}.log" 2>&1
}

echo "[start] $(date) src=${SRC_ROOT} methods=${METHODS[*]}" | tee "${LOGDIR}/run.log"

running=0
gpu=0
for method in "${METHODS[@]}"; do
  eval_one h3d "${method}" "${H3D_ANNO}" "${gpu}" &
  running=$((running + 1))
  gpu=$(((gpu + 1) % 8))
  if (( running >= EVAL_PARALLEL )); then
    wait
    running=0
  fi

  eval_one mh "${method}" "${MH_ANNO}" "${gpu}" &
  running=$((running + 1))
  gpu=$(((gpu + 1) % 8))
  if (( running >= EVAL_PARALLEL )); then
    wait
    running=0
  fi
done
wait

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
for split in ("h3d", "mh"):
    print(f"[{split}]")
    for p in sorted((root / split).glob("*_aligned_orig_c${CHUNK_SIZE}.json")):
        d = json.load(open(p))
        print(
            p.name,
            "samples", d.get("samples"),
            "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
            "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
            "FID", f"{d.get('fid_mean', float('nan')):.4f}",
            "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
            "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
        )
PY

touch "${OUT_ROOT}/_DONE"
echo "[done] $(date)" | tee -a "${LOGDIR}/run.log"
