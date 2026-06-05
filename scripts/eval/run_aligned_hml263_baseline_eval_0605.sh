#!/usr/bin/env bash
# Re-evaluate existing HML3D-263 -> SMPL retarget outputs with the corrected
# root-frame alignment protocol used by the Real(HML3D->SMPL) control.
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/aligned_hml263_baselines_0605}
MC135_ROOT="${OUT_ROOT}/motionclip135"
LOGDIR="${OUT_ROOT}/logs"
mkdir -p "${LOGDIR}" "${OUT_ROOT}/h3d" "${OUT_ROOT}/mh" "${MC135_ROOT}/h3d" "${MC135_ROOT}/mh"

H3D_ANNO="data/annotation/test_hml3d.json"
H3D_CAPTION="data/annotation/test_hml3d_rewritten.json"
MH_ANNO="data/annotation/test_motionhub_t2m.json"
MH_CAPTION="data/annotation/test_motionhub_t2m_rewritten.json"

declare -A H3D_SRC=(
  [motiongpt3]=outputs/evaluation/humanml3d_smpl135_fpsfix_v5_fixed0604/motiongpt3_fixed
  [mld]=outputs/evaluation/humanml3d_smpl135_fpsfix_v5_fixed0604/mld_v1_rootfix
  [momask]=outputs/evaluation/momask_all2_smpl135_0605/h3d
  [mdm]=outputs/evaluation/humanml3d_smpl135_fpsfix_v5_fixed0604/mdm_fixed
  [t2mgpt]=outputs/evaluation/t2mgpt_smpl135_fpsfix_0605/h3d
  [motiongpt]=outputs/evaluation/motiongpt_smpl135_fpsfix_0605/h3d
)

declare -A MH_SRC=(
  [motiongpt3]=outputs/evaluation/motionhub_smpl135_fpsfix_0605/motiongpt3_fixed
  [mld]=outputs/evaluation/motionhub_smpl135_fpsfix_0605/mld_adapter
  [momask]=outputs/evaluation/momask_all2_smpl135_0605/mh
  [mdm]=outputs/evaluation/motionhub_smpl135_fpsfix_0605/mdm_fixed
  [t2mgpt]=outputs/evaluation/t2mgpt_smpl135_fpsfix_0605/mh
  [motiongpt]=outputs/evaluation/motiongpt_smpl135_fpsfix_0605/mh
)

METHODS=(${METHODS:-motiongpt3 mld momask mdm t2mgpt motiongpt})

remap_one() {
  local split="$1"
  local method="$2"
  local src="$3"
  local anno="$4"
  local out="${MC135_ROOT}/${split}/${method}"
  if [[ ! -d "${src}" ]]; then
    echo "[skip-remap] ${split}/${method}: missing ${src}" | tee -a "${LOGDIR}/run.log"
    return 0
  fi
  echo "[remap] ${split}/${method} src=${src} $(date)" | tee -a "${LOGDIR}/run.log"
  python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
    --anno-file "${anno}" \
    --data-dir data/motionhub \
    --src-dir "${src}" \
    --out-dir "${out}" \
    --include-mirrors \
    --key-fallback \
    --align-to-gt-root \
    --overwrite \
    --workers "${REMAP_WORKERS:-16}" \
    > "${LOGDIR}/remap_${split}_${method}.log" 2>&1
}

eval_one() {
  local split="$1"
  local method="$2"
  local anno="$3"
  local caption="$4"
  local gpu="$5"
  local pred="${MC135_ROOT}/${split}/${method}"
  if [[ ! -d "${pred}" ]]; then
    echo "[skip-eval] ${split}/${method}: missing ${pred}" | tee -a "${LOGDIR}/run.log"
    return 0
  fi
  echo "[eval] ${split}/${method} gpu=${gpu} $(date)" | tee -a "${LOGDIR}/run.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file "${anno}" \
    --data_dir data/motionhub \
    --pred_dir "${pred}" \
    --rewritten_caption_file "${caption}" \
    --chunk_size 64 \
    --out_json "${OUT_ROOT}/${split}/${method}_aligned_rw_c64.json" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/eval_${split}_${method}.log" 2>&1
}

echo "[start] $(date) methods=${METHODS[*]}" | tee "${LOGDIR}/run.log"

for method in "${METHODS[@]}"; do
  remap_one h3d "${method}" "${H3D_SRC[${method}]}" "${H3D_ANNO}"
  remap_one mh "${method}" "${MH_SRC[${method}]}" "${MH_ANNO}"
done

idx=0
for method in "${METHODS[@]}"; do
  eval_one h3d "${method}" "${H3D_ANNO}" "${H3D_CAPTION}" "$((idx % 8))" &
  idx=$((idx + 1))
  eval_one mh "${method}" "${MH_ANNO}" "${MH_CAPTION}" "$((idx % 8))" &
  idx=$((idx + 1))
done
wait

python3 - <<PY | tee "${OUT_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
for split in ("h3d", "mh"):
    print(f"[{split}]")
    for p in sorted((root / split).glob("*_aligned_rw_c64.json")):
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
