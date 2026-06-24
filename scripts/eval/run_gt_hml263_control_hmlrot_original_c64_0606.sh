#!/usr/bin/env bash
# Clean GT control rerun:
#   Real(SMPL) -> MotionCLIP evaluator sanity
#   Real(SMPL) -> HML3D-263 -> SMPL with HML local-rotation initialization
# Evaluation uses original captions and chunk size 64.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/gt_hml263_control_hmlrot_original_c64_0606}
BASE_263=${BASE_263:-outputs/evaluation/humanml3d/gt_smpl135_to_hml263}
SMPL_ROOT=${SMPL_ROOT:-${OUT_ROOT}/smpl_npz}
MC135_ROOT=${MC135_ROOT:-${OUT_ROOT}/motionclip135}
NUM_SHARDS=${NUM_SHARDS:-8}
WORKERS=${WORKERS:-16}
LIMIT=${LIMIT:-0}
LOGDIR="${OUT_ROOT}/logs"

mkdir -p "${LOGDIR}" "${OUT_ROOT}/real_smpl" "${SMPL_ROOT}" "${MC135_ROOT}"
exec > >(tee -a "${LOGDIR}/run.log") 2>&1

echo "[start] $(date) out=${OUT_ROOT} limit=${LIMIT}"

limit_arg=()
if [ "${LIMIT}" != "0" ]; then
  limit_arg=(--limit "${LIMIT}")
fi

build_hml263() {
  local split="$1"
  local anno="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}"
  echo "[${split}-build-hml263] $(date)"
  python3 scripts/eval/build_gt_smpl135_to_hml263.py \
    --anno-file "${anno}" \
    --data-dir data/motionhub \
    --out-dir "${out_dir}" \
    --workers "${WORKERS}" \
    --skip-existing \
    "${limit_arg[@]}" \
    > "${LOGDIR}/build_${split}.log" 2>&1
}

run_gt_only() {
  local split="$1"
  local anno="$2"
  local gpu="$3"
  echo "[${split}-real-smpl] $(date)"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --gt_only \
    --rot6d_convention column \
    --chunk_size 64 \
    --out_json "${OUT_ROOT}/real_smpl/${split}_real_smpl_orig_c64.json" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/real_smpl_${split}.log" 2>&1
}

run_ik_split() {
  local split="$1"
  local src="$2"
  local out="${SMPL_ROOT}/${split}"
  mkdir -p "${out}"
  echo "[${split}-ik-hmlrot] $(date) src=$(find "${src}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu=$((shard % 8))
    CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "${src}" \
      --out-dir "${out}" \
      --model-dir ref_repo/MDM/body_models \
      --source-fps 20 \
      --target-fps 30 \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --device cuda \
      --batch-size 512 \
      --floor-align \
      --rotation-init hml263 \
      --rot6d-convention column \
      --refine-iters 0 \
      --skip-existing \
      "${limit_arg[@]}" \
      > "${LOGDIR}/ik_${split}_s${shard}_of_${NUM_SHARDS}.log" 2>&1 &
  done
  wait
  echo "[${split}-ik-done] npz=$(find "${out}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)"
}

remap_split() {
  local split="$1"
  local anno="$2"
  local src="${SMPL_ROOT}/${split}"
  local out="${MC135_ROOT}/${split}"
  mkdir -p "${out}"
  echo "[${split}-remap] $(date)"
  python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
    --anno-file "${anno}" \
    --data-dir data/motionhub \
    --src-dir "${src}" \
    --out-dir "${out}" \
    --include-mirrors \
    --key-fallback \
    --align-to-gt-root \
    --overwrite \
    --workers "${WORKERS}" \
    > "${LOGDIR}/remap_${split}.log" 2>&1
}

eval_split() {
  local split="$1"
  local anno="$2"
  local gpu="$3"
  echo "[${split}-eval-control] $(date)"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${MC135_ROOT}/${split}" \
    --rot6d_convention column \
    --chunk_size 64 \
    --out_json "${OUT_ROOT}/${split}_hml263_to_smpl_hmlrot_orig_c64.json" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/eval_${split}.log" 2>&1
}

H3D_263="${BASE_263}/humanml3d"
MH_263="${BASE_263}/motionhub"
build_hml263 h3d data/annotation/test_hml3d.json "${H3D_263}"
build_hml263 mh data/annotation/test_motionhub_t2m.json "${MH_263}"

run_gt_only h3d data/annotation/test_hml3d.json 0 &
run_gt_only mh data/annotation/test_motionhub_t2m.json 1 &
wait

run_ik_split h3d "${H3D_263}"
run_ik_split mh "${MH_263}"

remap_split h3d data/annotation/test_hml3d.json &
remap_split mh data/annotation/test_motionhub_t2m.json &
wait

eval_split h3d data/annotation/test_hml3d.json 0 &
eval_split mh data/annotation/test_motionhub_t2m.json 1 &
wait

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
items = [
    ("h3d_real_smpl", root / "real_smpl/h3d_real_smpl_orig_c64.json"),
    ("mh_real_smpl", root / "real_smpl/mh_real_smpl_orig_c64.json"),
    ("h3d_hml263_to_smpl", root / "h3d_hml263_to_smpl_hmlrot_orig_c64.json"),
    ("mh_hml263_to_smpl", root / "mh_hml263_to_smpl_hmlrot_orig_c64.json"),
]
for name, path in items:
    if not path.exists():
        print(name, "missing", path)
        continue
    d = json.load(open(path))
    print(
        name,
        "samples", d.get("samples"),
        "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
        "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
        "FID", f"{d.get('fid_mean', float('nan')):.4f}",
        "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
        "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
    )
PY

touch "${OUT_ROOT}/_DONE"
echo "[done] $(date)"
