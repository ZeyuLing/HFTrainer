#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export HFTRAINER_SKIP_AUTOREGISTER=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_ROOT="outputs/evaluation/recons/humanml3d_official_test"
METRIC_DIR="${OUT_ROOT}/_metrics/motionclip_vae_recon_comparison_20260624"
MS_MC135="${OUT_ROOT}/motion135/motionstreamer_tae_fkpos_canon272_for_motionclip"
GT_MC135="${OUT_ROOT}/motion135/gt_0beta_for_motionclip"
PRISM_MC135="${OUT_ROOT}/motion135/prism_vae_for_motionclip"
MANIFEST="${METRIC_DIR}/pred_manifest.tsv"
LOG_DIR="${METRIC_DIR}/logs"

mkdir -p "${METRIC_DIR}" "${LOG_DIR}"

echo "[step] repack GT/PRISM row-major motion_135 -> MotionCLIP column-major"
python3 scripts/eval/repack_motion135_6d_convention.py \
  --in-dir "${OUT_ROOT}/motion135/gt_0beta" \
  --out-dir "${GT_MC135}" \
  --src row \
  --dst column \
  --workers 32 \
  2>&1 | tee "${LOG_DIR}/repack_gt_row_to_column.log"

python3 scripts/eval/repack_motion135_6d_convention.py \
  --in-dir "${OUT_ROOT}/motion135/prism_vae" \
  --out-dir "${PRISM_MC135}" \
  --src row \
  --dst column \
  --workers 32 \
  2>&1 | tee "${LOG_DIR}/repack_prism_row_to_column.log"

echo "[step] convert MotionStreamer TAE 272 -> MotionCLIP135"
python3 scripts/eval/convert_ms272_dir_for_t2m_eval.py \
  --src-dir "${OUT_ROOT}/ms272/motionstreamer_tae_fkpos_canon272" \
  --anno-file data/annotation/test_hml3d_official272_gtlen.json \
  --data-dir . \
  --motionclip-dir "${MS_MC135}" \
  --rot6d-convention column \
  --workers 32 \
  --overwrite \
  2>&1 | tee "${LOG_DIR}/convert_motionstreamer_tae_mc135.log"

cat > "${MANIFEST}" <<'EOF'
GT	outputs/evaluation/recons/humanml3d_official_test/motion135/gt_0beta_for_motionclip
MotionLCM VAE	outputs/evaluation/recons/humanml3d_official_test/motion135/motionlcm_vae_bridge_ik80/pred_m135
MotionStreamer TAE	outputs/evaluation/recons/humanml3d_official_test/motion135/motionstreamer_tae_fkpos_canon272_for_motionclip
PRISM VAE (ours)	outputs/evaluation/recons/humanml3d_official_test/motion135/prism_vae_for_motionclip
EOF

echo "[step] MotionCLIP evaluator"
python3 scripts/eval/eval_motionclip_table1_dirs.py \
  --evaluator-ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --clip-pretrained checkpoints/clip-vit-base-patch32 \
  --stats-file data/statistic/smplx55_stats_hymotion_aug.json \
  --anno-file data/annotation/test_hml3d_official272_gtlen.json \
  --data-dir . \
  --caption-key hierarchical_caption \
  --real-dir "${GT_MC135}" \
  --pred-manifest "${MANIFEST}" \
  --out-dir "${METRIC_DIR}" \
  --min-frames 60 \
  --max-frames 300 \
  --forward-batch-size 64 \
  --chunk-size 32 \
  --n-repeats 20 \
  --seed 0 \
  2>&1 | tee "${LOG_DIR}/eval_motionclip_recons.log"

echo "[done] ${METRIC_DIR}/summary.tsv"
