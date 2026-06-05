#!/usr/bin/env bash
# Small A/B for the Real HML3D-263 -> SMPL retarget control.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SRC=${SRC:-outputs/evaluation/humanml3d/gt_smpl135_to_hml263/humanml3d}
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/gt_hml263_retarget_ab_0605}
NUM_SHARDS=${NUM_SHARDS:-8}
MAX_CASES=${MAX_CASES:-256}
LOGDIR="${OUT_ROOT}/logs"
mkdir -p "${LOGDIR}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"
if [ "${NUM_SHARDS}" -gt "${#GPUS[@]}" ]; then
  echo "[warn] NUM_SHARDS=${NUM_SHARDS} exceeds GPU_LIST size=${#GPUS[@]}; GPUs will be reused" >&2
fi

IDS="${OUT_ROOT}/ids_${MAX_CASES}.txt"
python3 - <<PY > "${IDS}"
import json
from pathlib import Path
src = Path("${SRC}")
anno = json.load(open("data/annotation/test_hml3d.json"))["data_list"]
n = 0
for key, entry in anno.items():
    if (src / f"{key}.npy").exists():
        print(key)
        n += 1
        if n >= int("${MAX_CASES}"):
            break
PY
echo "[start] src=${SRC} ids=$(wc -l < "${IDS}") out=${OUT_ROOT} $(date)" | tee "${LOGDIR}/run.log"

run_variant() {
  local tag="$1"
  local orient="$2"
  local refine="$3"
  local parent_w="$4"
  local pose_l2="$5"
  local angle_w="$6"
  local smpl_dir="${OUT_ROOT}/${tag}/smpl_npz"
  local mc_dir="${OUT_ROOT}/${tag}/motionclip135"
  mkdir -p "${smpl_dir}" "${mc_dir}" "${OUT_ROOT}/${tag}"
  echo "[variant:${tag}] orient=${orient} refine=${refine} $(date)" | tee -a "${LOGDIR}/run.log"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "${SRC}" \
      --out-dir "${smpl_dir}" \
      --ids "${IDS}" \
      --model-dir ref_repo/MDM/body_models \
      --source-fps 20 \
      --target-fps 30 \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --device cuda \
      --batch-size 512 \
      --floor-align \
      --orientation-mode "${orient}" \
      --parent-ref-weight "${parent_w}" \
      --refine-iters "${refine}" \
      --pose-l2-weight "${pose_l2}" \
      --angle-prior-weight "${angle_w}" \
      --skip-existing \
      > "${LOGDIR}/ik_${tag}_s${shard}_gpu${gpu}.log" 2>&1 &
  done
  wait
  python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
    --anno-file data/annotation/test_hml3d.json \
    --src-dir "${smpl_dir}" \
    --out-dir "${mc_dir}" \
    --include-mirrors \
    --key-fallback \
    --overwrite \
    --workers 8 \
    > "${LOGDIR}/remap_${tag}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_hml3d.json \
    --data_dir data/motionhub \
    --pred_dir "${mc_dir}" \
    --rewritten_caption_file data/annotation/test_hml3d_rewritten.json \
    --chunk_size 64 \
    --out_json "${OUT_ROOT}/${tag}/motionclip_c64.json" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/eval_${tag}.log" 2>&1
}

run_variant bone_r0 bone 0 0.25 0.0 0.0
run_variant parent_r0 parent_frame 0 0.25 0.0 0.0
run_variant parent_r30 parent_frame 30 0.25 1e-5 0.0

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
for tag in ["bone_r0", "parent_r0", "parent_r30"]:
    print("\\n" + tag)
    p = root / tag / "motionclip_c64.json"
    if p.exists():
        d = json.load(open(p))
        print(
            "samples", d.get("samples"),
            "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
            "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
            "FID", f"{d.get('fid_mean', float('nan')):.4f}",
            "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
            "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
        )
    vals = []
    for s in (root / tag / "smpl_npz").glob("_retarget_summary_s*_of_*.json"):
        x = json.load(open(s))
        vals.append(x.get("mean_mpjpe_mm"))
    vals = [v for v in vals if v is not None]
    if vals:
        print("mean shard MPJPE(mm)", f"{sum(vals)/len(vals):.2f}")
PY

touch "${OUT_ROOT}/_DONE"
echo "[done] $(date)" | tee -a "${LOGDIR}/run.log"
