#!/usr/bin/env bash
# Evaluate existing MotionGPT SMPL/MotionCLIP135 outputs with original captions.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PRED_ROOT=${PRED_ROOT:-outputs/evaluation/motiongpt_smpl135_fpsfix_0605_motionclip135}
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/motiongpt_orig_eval0606}
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
mkdir -p "${OUT_ROOT}/h3d" "${OUT_ROOT}/mh" "${OUT_ROOT}/logs"

run_eval() {
  local split="$1"
  local gpu="$2"
  local anno="$3"
  local pred="${PRED_ROOT}/${split}"
  local out="${OUT_ROOT}/${split}/motiongpt_orig_c${CHUNK_SIZE}.json"
  if [ ! -d "${pred}" ]; then
    echo "[missing] ${pred}" >&2
    return 2
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${pred}" \
    --chunk_size "${CHUNK_SIZE}" \
    --out_json "${out}" \
    --n_repeats "${N_REPEATS}" \
    --seed 42 \
    > "${OUT_ROOT}/logs/eval_${split}.log" 2>&1
}

echo "[start] $(date) pred=${PRED_ROOT}" | tee "${OUT_ROOT}/logs/run.log"
run_eval h3d "${H3D_GPU:-1}" data/annotation/test_hml3d.json &
run_eval mh "${MH_GPU:-3}" data/annotation/test_motionhub_t2m.json &
wait

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
for split in ("h3d", "mh"):
    p = root / split / "motiongpt_orig_c${CHUNK_SIZE}.json"
    if not p.exists():
        print(split, "missing", p)
        continue
    d = json.load(open(p))
    print(
        split,
        "samples", d.get("samples"),
        "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
        "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
        "FID", f"{d.get('fid_mean', float('nan')):.4f}",
        "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
        "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
    )
PY

touch "${OUT_ROOT}/_DONE"
echo "[done] $(date)" | tee -a "${OUT_ROOT}/logs/run.log"
