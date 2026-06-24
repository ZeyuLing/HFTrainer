#!/usr/bin/env bash
# Evaluate an HML3D-263 baseline through the validated 263 -> 272 ->
# MotionCLIP135 path, using original captions for retrieval metrics.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

METHOD=${METHOD:?Set METHOD, e.g. motiongpt}
H3D_SRC=${H3D_SRC:?Set H3D_SRC to the H3D 263D output directory}
MH_SRC=${MH_SRC:?Set MH_SRC to the MotionHub 263D output directory}

EVAL_ROOT=${EVAL_ROOT:-outputs/evaluation/${METHOD}_via272_original_c64_eval0605}
H272_ROOT=${H272_ROOT:-${EVAL_ROOT}/h272}
MC135_ROOT=${MC135_ROOT:-${EVAL_ROOT}/motionclip135}
LOGDIR="${EVAL_ROOT}/logs"
WORKERS=${WORKERS:-16}
POSITION_SOURCE=${POSITION_SOURCE:-ric}
CONVERT_ONLY=${CONVERT_ONLY:-0}
mkdir -p "${LOGDIR}" "${H272_ROOT}/h3d" "${H272_ROOT}/mh" "${MC135_ROOT}/h3d" "${MC135_ROOT}/mh"

convert_263_to_272() {
  local split="$1"
  local src="$2"
  local out="${H272_ROOT}/${split}"
  echo "[${split}-263to272] $(date)" | tee -a "${LOGDIR}/run.log"
  python3 scripts/data/convert_hml263_pose_to_h3d272.py \
    --pred_dir_263 "${src}" \
    --out_dir_272 "${out}" \
    --position_source "${POSITION_SOURCE}" \
    --skip_existing \
    > "${LOGDIR}/convert263_${split}.log" 2>&1
}

convert_272_to_motionclip() {
  local split="$1"
  local anno="$2"
  local out="${MC135_ROOT}/${split}"
  echo "[${split}-272to135] $(date)" | tee -a "${LOGDIR}/run.log"
  python3 scripts/eval/convert_ms272_dir_for_t2m_eval.py \
    --src-dir "${H272_ROOT}/${split}" \
    --anno-file "${anno}" \
    --data-dir data/motionhub \
    --motionclip-dir "${out}" \
    --align-to-gt-root \
    --workers "${WORKERS}" \
    --overwrite \
    > "${LOGDIR}/convert272_${split}.log" 2>&1
}

run_motionclip_eval() {
  local split="$1"
  local anno="$2"
  local gpu="$3"
  echo "[${split}-motionclip] $(date)" | tee -a "${LOGDIR}/run.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${MC135_ROOT}/${split}" \
    --chunk_size 64 \
    --out_json "${EVAL_ROOT}/${split}_${METHOD}_via272_orig_c64.json" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/eval_${split}.log" 2>&1
}

echo "[start] method=${METHOD} $(date)" | tee "${LOGDIR}/run.log"
echo "[sources] h3d=$(find "${H3D_SRC}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l) mh=$(find "${MH_SRC}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l) position_source=${POSITION_SOURCE}" | tee -a "${LOGDIR}/run.log"

convert_263_to_272 h3d "${H3D_SRC}" &
convert_263_to_272 mh "${MH_SRC}" &
wait

convert_272_to_motionclip h3d data/annotation/test_hml3d.json &
convert_272_to_motionclip mh data/annotation/test_motionhub_t2m.json &
wait

if [ "${CONVERT_ONLY}" = "1" ]; then
  echo "[convert-only] $(date)" | tee -a "${LOGDIR}/run.log"
  touch "${EVAL_ROOT}/_CONVERT_DONE"
  exit 0
fi

run_motionclip_eval h3d data/annotation/test_hml3d.json "${H3D_GPU:-0}" &
run_motionclip_eval mh data/annotation/test_motionhub_t2m.json "${MH_GPU:-0}" &
wait

python3 - <<PY | tee "${EVAL_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${EVAL_ROOT}")
method = "${METHOD}"
for split in ("h3d", "mh"):
    p = root / f"{split}_{method}_via272_orig_c64.json"
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

touch "${EVAL_ROOT}/_DONE"
echo "[done] $(date)" | tee -a "${LOGDIR}/run.log"
