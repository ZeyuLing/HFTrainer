#!/usr/bin/env bash
# MotionGPT embed-repair outputs: rel-retarget MotionHub and evaluate original captions.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

H3D_MC135=${H3D_MC135:-outputs/evaluation/motiongpt_official_embedrepair_h3d_rel_motionclip135_0606/h3d}
MH_SRC=${MH_SRC:-outputs/evaluation/humanml3d/motiongpt_embedrepair_full0605_neo/motionhub}
MH_SMPL=${MH_SMPL:-outputs/evaluation/motiongpt_official_embedrepair_mh_rel_smpl135_0606/mh}
MH_MC135=${MH_MC135:-outputs/evaluation/motiongpt_official_embedrepair_mh_rel_motionclip135_0606/mh}
EVAL_ROOT=${EVAL_ROOT:-outputs/evaluation/motiongpt_official_embedrepair_rel_orig_eval0606}
NUM_SHARDS=${NUM_SHARDS:-3}
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-1,3,4}"
CHUNK_SIZE=${CHUNK_SIZE:-64}
N_REPEATS=${N_REPEATS:-20}
LOGDIR="${EVAL_ROOT}/logs"

mkdir -p "${LOGDIR}" "${EVAL_ROOT}/h3d" "${EVAL_ROOT}/mh" "${MH_SMPL}" "${MH_MC135}"
exec > >(tee -a "${LOGDIR}/run.log") 2>&1

echo "[start] $(date)"
echo "[sources] h3d_mc135=$(find "${H3D_MC135}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l) mh_src=$(find "${MH_SRC}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)"

echo "[mh-retarget] $(date)"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_LIST[$((shard % ${#GPU_LIST[@]}))]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "${MH_SRC}" \
    --out-dir "${MH_SMPL}" \
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
    > "${LOGDIR}/ik_mh_s${shard}_gpu${gpu}.log" 2>&1 &
done
wait
echo "[mh-retarget-done] npz=$(find "${MH_SMPL}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)"

echo "[mh-remap] $(date)"
python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
  --anno-file data/annotation/test_motionhub_t2m.json \
  --data-dir data/motionhub \
  --src-dir "${MH_SMPL}" \
  --out-dir "${MH_MC135}" \
  --include-mirrors \
  --key-fallback \
  --align-to-gt-root \
  --overwrite \
  --workers 16 \
  > "${LOGDIR}/remap_mh.log" 2>&1
echo "[mh-remap-done] npy=$(find "${MH_MC135}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)"

eval_split() {
  local split="$1"
  local gpu="$2"
  local anno="$3"
  local pred="$4"
  local out="${EVAL_ROOT}/${split}/motiongpt_embedrepair_rel_orig_c${CHUNK_SIZE}.json"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${pred}" \
    --chunk_size "${CHUNK_SIZE}" \
    --out_json "${out}" \
    --n_repeats "${N_REPEATS}" \
    --seed 42 \
    > "${LOGDIR}/eval_${split}.log" 2>&1
}

echo "[eval] $(date)"
eval_split h3d "${GPU_LIST[0]}" data/annotation/test_hml3d.json "${H3D_MC135}" &
eval_split mh "${GPU_LIST[$((1 % ${#GPU_LIST[@]}))]}" data/annotation/test_motionhub_t2m.json "${MH_MC135}" &
wait

python3 - <<PY | tee "${EVAL_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${EVAL_ROOT}")
for split in ("h3d", "mh"):
    p = root / split / "motiongpt_embedrepair_rel_orig_c${CHUNK_SIZE}.json"
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
echo "[done] $(date)"
