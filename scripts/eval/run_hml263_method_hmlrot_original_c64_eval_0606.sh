#!/usr/bin/env bash
# Retarget one HumanML3D-263 baseline to SMPL using the HML263 local-rotation
# block for initialization, then evaluate with original captions.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

METHOD=${METHOD:?Set METHOD, e.g. flowmdm}
H3D_SRC=${H3D_SRC:?Set H3D_SRC to the H3D 263D output directory}
MH_SRC=${MH_SRC:?Set MH_SRC to the MotionHub 263D output directory}

EVAL_ROOT=${EVAL_ROOT:-outputs/evaluation/${METHOD}_hmlrot_original_c64_eval0606}
SMPL_ROOT=${SMPL_ROOT:-${EVAL_ROOT}/smpl_npz}
MC135_ROOT=${MC135_ROOT:-${EVAL_ROOT}/motionclip135}
NUM_SHARDS=${NUM_SHARDS:-8}
WORKERS=${WORKERS:-16}
LIMIT=${LIMIT:-0}
LOGDIR="${EVAL_ROOT}/logs"

mkdir -p "${LOGDIR}" "${SMPL_ROOT}" "${MC135_ROOT}"
exec > >(tee -a "${LOGDIR}/run.log") 2>&1

echo "[start] method=${METHOD} $(date) limit=${LIMIT}"
echo "[sources] h3d=$(find "${H3D_SRC}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l) mh=$(find "${MH_SRC}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)"

limit_arg=()
if [ "${LIMIT}" != "0" ]; then
  limit_arg=(--limit "${LIMIT}")
fi

run_ik_split() {
  local split="$1"
  local src="$2"
  local out="${SMPL_ROOT}/${split}"
  mkdir -p "${out}"
  echo "[${split}-ik-hmlrot] $(date)"
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
  echo "[${split}-eval-original] $(date)"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${MC135_ROOT}/${split}" \
    --rot6d_convention column \
    --chunk_size 64 \
    --out_json "${EVAL_ROOT}/${split}_${METHOD}_hmlrot_orig_c64.json" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/eval_${split}.log" 2>&1
}

run_ik_split h3d "${H3D_SRC}"
run_ik_split mh "${MH_SRC}"

remap_split h3d data/annotation/test_hml3d.json &
remap_split mh data/annotation/test_motionhub_t2m.json &
wait

eval_split h3d data/annotation/test_hml3d.json 0 &
eval_split mh data/annotation/test_motionhub_t2m.json 1 &
wait

python3 - <<PY | tee "${EVAL_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${EVAL_ROOT}")
method = "${METHOD}"
for split in ("h3d", "mh"):
    path = root / f"{split}_{method}_hmlrot_orig_c64.json"
    if not path.exists():
        print(split, "missing", path)
        continue
    d = json.load(open(path))
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
echo "[done] $(date)"
