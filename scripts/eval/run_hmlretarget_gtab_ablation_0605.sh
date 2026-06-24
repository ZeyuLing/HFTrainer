#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT_ROOT=${1:-outputs/evaluation/hmlretarget_gtab_0605}
GPU=${2:-0}
LIMIT=${LIMIT:-128}
mkdir -p "${OUT_ROOT}/logs"
exec > >(tee "${OUT_ROOT}/logs/run.log") 2>&1

echo "[start] $(date) out=${OUT_ROOT} gpu=${GPU} limit=${LIMIT}"

for cfg in base gmm1 gmm_relaxed; do
  OUT="${OUT_ROOT}/${cfg}/smpl135"
  MC="${OUT_ROOT}/${cfg}/motionclip135"
  mkdir -p "${OUT}" "${MC}"
  if [ "${cfg}" = base ]; then
    EXTRA=(--refine-iters 0)
  elif [ "${cfg}" = gmm1 ]; then
    EXTRA=(
      --refine-iters 80
      --refine-lr 0.01
      --gmm-pose-prior-weight 1e-5
      --angle-prior-weight 1e-4
      --pose-l2-weight 1e-5
      --foot-height-align
    )
  else
    EXTRA=(
      --refine-iters 80
      --refine-lr 0.01
      --gmm-pose-prior-weight 3e-5
      --angle-prior-weight 1e-4
      --pose-l2-weight 1e-5
      --joint-fit-weight-preset relaxed_upper
      --foot-height-align
    )
  fi

  echo "[retarget] ${cfg} ${EXTRA[*]}"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir work_dirs/h3d263_eval/h3d263_test_recon_fk/new_joint_vecs \
    --out-dir "${OUT}" \
    --model-dir ref_repo/MDM/body_models \
    --source-fps 20 \
    --target-fps 30 \
    --floor-align \
    --limit "${LIMIT}" \
    "${EXTRA[@]}"

  python3 scripts/eval/convert_row135_npz_to_motionclip_col.py \
    --anno-file data/annotation/test_hml3d.json \
    --src-dir "${OUT}" \
    --out-dir "${MC}" \
    --overwrite

  CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/eval/eval_with_motionclip_evaluator.py \
    --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --anno_file data/annotation/test_hml3d.json \
    --data_dir data/motionhub \
    --pred_dir "${MC}" \
    --out_json "${OUT_ROOT}/${cfg}/motionclip_orig_c32_rep5.json" \
    --chunk_size 32 \
    --n_repeats 5 \
    --forward_batch_size 64
done

echo "[done] $(date)"
