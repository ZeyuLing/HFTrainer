#!/usr/bin/env bash
# Retarget one HML3D-263 baseline to SMPL/MotionCLIP135 and evaluate H3D/MotionHub.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

METHOD=${METHOD:?Set METHOD, e.g. t2mgpt}
H3D_SRC=${H3D_SRC:?Set H3D_SRC to the H3D 263D output directory}
MH_SRC=${MH_SRC:?Set MH_SRC to the MotionHub 263D output directory}

EVAL_ROOT=${EVAL_ROOT:-outputs/evaluation/${METHOD}_rw_c64_eval0605}
SMPL_ROOT=${SMPL_ROOT:-outputs/evaluation/${METHOD}_smpl135_fpsfix_0605}
MC135_ROOT=${MC135_ROOT:-outputs/evaluation/${METHOD}_smpl135_fpsfix_0605_motionclip135}
NUM_SHARDS=${NUM_SHARDS:-8}
LOGDIR="${EVAL_ROOT}/logs"

mkdir -p "${LOGDIR}" "${EVAL_ROOT}/h3d" "${EVAL_ROOT}/mh" "${SMPL_ROOT}" "${MC135_ROOT}"

run_ik_shard() {
  local split="$1"
  local src="$2"
  local gpu="$3"
  local shard="$4"
  local out="${SMPL_ROOT}/${split}"
  mkdir -p "${out}"
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
    --refine-iters 0 \
    --skip-existing \
    > "${LOGDIR}/ik_${split}_s${shard}_of_${NUM_SHARDS}_gpu${gpu}.log" 2>&1
}

run_retarget_split() {
  local split="$1"
  local src="$2"
  echo "[${split}-retarget] $(date)" | tee -a "${LOGDIR}/run.log"
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    run_ik_shard "${split}" "${src}" "${i}" "${i}" &
  done
  wait
}

run_remap_split() {
  local split="$1"
  local anno="$2"
  echo "[${split}-remap] $(date)" | tee -a "${LOGDIR}/run.log"
  python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
    --anno-file "${anno}" \
    --data-dir data/motionhub \
    --src-dir "${SMPL_ROOT}/${split}" \
    --out-dir "${MC135_ROOT}/${split}" \
    --include-mirrors \
    --key-fallback \
    --align-to-gt-root \
    --overwrite \
    --workers 16 \
    > "${LOGDIR}/remap_${split}.log" 2>&1
}

run_eval_split() {
  local split="$1"
  local anno="$2"
  local caption="$3"
  local gpu="$4"
  echo "[${split}-eval] $(date)" | tee -a "${LOGDIR}/run.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${MC135_ROOT}/${split}" \
    --rewritten_caption_file "${caption}" \
    --chunk_size 64 \
    --out_json "${EVAL_ROOT}/${split}/${METHOD}_rw_c64.json" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/eval_${split}.log" 2>&1
}

echo "[start] method=${METHOD} $(date)" | tee "${LOGDIR}/run.log"
echo "[sources] h3d=$(find "${H3D_SRC}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l) mh=$(find "${MH_SRC}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)" | tee -a "${LOGDIR}/run.log"

run_retarget_split h3d "${H3D_SRC}"
run_retarget_split mh "${MH_SRC}"

run_remap_split h3d data/annotation/test_hml3d.json &
run_remap_split mh data/annotation/test_motionhub_t2m.json &
wait

run_eval_split h3d data/annotation/test_hml3d.json data/annotation/test_hml3d_rewritten.json 0 &
run_eval_split mh data/annotation/test_motionhub_t2m.json data/annotation/test_motionhub_t2m_rewritten.json 1 &
wait

python3 - <<PY | tee "${EVAL_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${EVAL_ROOT}")
for split in ("h3d", "mh"):
    p = root / split / "${METHOD}_rw_c64.json"
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
