#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

EVAL_ROOT=${EVAL_ROOT:-outputs/evaluation/t2m_baseline_rw_c64_eval0605}
H3D_MC135_ROOT=${H3D_MC135_ROOT:-outputs/evaluation/humanml3d_smpl135_fpsfix_v5_fixed0604_motionclip135_v6}
MH_263_ROOT=${MH_263_ROOT:-outputs/evaluation/humanml3d}
MH_SMPL_ROOT=${MH_SMPL_ROOT:-outputs/evaluation/motionhub_smpl135_fpsfix_0605}
MH_MC135_ROOT=${MH_MC135_ROOT:-outputs/evaluation/motionhub_smpl135_fpsfix_0605_motionclip135}
LOGDIR="${EVAL_ROOT}/logs"
mkdir -p "${LOGDIR}" "${EVAL_ROOT}/h3d" "${EVAL_ROOT}/mh" "${MH_SMPL_ROOT}" "${MH_MC135_ROOT}"

run_h3d_eval() {
  local method="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_hml3d.json \
    --data_dir data/motionhub \
    --pred_dir "${H3D_MC135_ROOT}/${method}" \
    --rewritten_caption_file data/annotation/test_hml3d_rewritten.json \
    --chunk_size 64 \
    --out_json "${EVAL_ROOT}/h3d/${method}_rw_c64.json" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/h3d_${method}_rw_c64.log" 2>&1
}

run_ik_shard() {
  local method="$1"
  local src="$2"
  local gpu="$3"
  local nshards="$4"
  local shard="$5"
  local out="${MH_SMPL_ROOT}/${method}"
  mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "${src}" \
    --out-dir "${out}" \
    --model-dir ref_repo/MDM/body_models \
    --source-fps 20 \
    --target-fps 30 \
    --num-shards "${nshards}" \
    --shard-index "${shard}" \
    --device cuda \
    --batch-size 512 \
    --floor-align \
    --refine-iters 0 \
    --skip-existing \
    > "${LOGDIR}/mh_ik_${method}_s${shard}_of_${nshards}_gpu${gpu}.log" 2>&1
}

run_remap() {
  local method="$1"
  python3 scripts/eval/remap_hml3d_smpl_to_motionclip135.py \
    --anno-file data/annotation/test_motionhub_t2m.json \
    --data-dir data/motionhub \
    --src-dir "${MH_SMPL_ROOT}/${method}" \
    --out-dir "${MH_MC135_ROOT}/${method}" \
    --include-mirrors \
    --key-fallback \
    --align-to-gt-root \
    --overwrite \
    --workers 8 \
    > "${LOGDIR}/mh_remap_${method}.log" 2>&1
}

run_mh_eval() {
  local method="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_motionhub_t2m.json \
    --data_dir data/motionhub \
    --pred_dir "${MH_MC135_ROOT}/${method}" \
    --rewritten_caption_file data/annotation/test_motionhub_t2m_rewritten.json \
    --chunk_size 64 \
    --out_json "${EVAL_ROOT}/mh/${method}_rw_c64.json" \
    --n_repeats 20 \
    --seed 42 \
    > "${LOGDIR}/mh_${method}_rw_c64.log" 2>&1
}

echo "[h3d-eval] $(date)" | tee "${LOGDIR}/run.log"
run_h3d_eval momask 0 &
run_h3d_eval mdm_fixed 1 &
run_h3d_eval motiongpt3_fixed 2 &
run_h3d_eval mld_v1_rootfix 3 &
wait

echo "[mh-retarget] $(date)" | tee -a "${LOGDIR}/run.log"
run_ik_shard momask "${MH_263_ROOT}/momask_hml3d263_gumbel/motionhub" 0 2 0 &
run_ik_shard momask "${MH_263_ROOT}/momask_hml3d263_gumbel/motionhub" 1 2 1 &
run_ik_shard mdm_fixed "${MH_263_ROOT}/mdm_hml3d263/motionhub" 2 2 0 &
run_ik_shard mdm_fixed "${MH_263_ROOT}/mdm_hml3d263/motionhub" 3 2 1 &
run_ik_shard motiongpt3_fixed "${MH_263_ROOT}/motiongpt3_hml3d263/motionhub" 4 2 0 &
run_ik_shard motiongpt3_fixed "${MH_263_ROOT}/motiongpt3_hml3d263/motionhub" 5 2 1 &
run_ik_shard mld_adapter "${MH_263_ROOT}/mld_hml3d263_adapter/motionhub" 6 2 0 &
run_ik_shard mld_adapter "${MH_263_ROOT}/mld_hml3d263_adapter/motionhub" 7 2 1 &
wait

echo "[mh-remap] $(date)" | tee -a "${LOGDIR}/run.log"
run_remap momask &
run_remap mdm_fixed &
run_remap motiongpt3_fixed &
run_remap mld_adapter &
wait

echo "[mh-eval] $(date)" | tee -a "${LOGDIR}/run.log"
run_mh_eval momask 0 &
run_mh_eval mdm_fixed 1 &
run_mh_eval motiongpt3_fixed 2 &
run_mh_eval mld_adapter 3 &
wait

python3 - <<'PY' | tee "${EVAL_ROOT}/summary.txt"
import json
from pathlib import Path
root = Path("outputs/evaluation/t2m_baseline_rw_c64_eval0605")
for split in ("h3d", "mh"):
    print(f"[{split}]")
    for p in sorted((root / split).glob("*.json")):
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
touch "${EVAL_ROOT}/_DONE"
echo "[done] $(date)" | tee -a "${LOGDIR}/run.log"
