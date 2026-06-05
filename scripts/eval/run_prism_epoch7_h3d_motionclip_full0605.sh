#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

RAW_DIR="${RAW_DIR:-outputs/evaluation/prism_kt_spectral_epoch7_rw_merged0604/h3d/depth_driven}"
NPY_DIR="${NPY_DIR:-outputs/evaluation/prism_kt_spectral_epoch7_rw_merged0604/h3d/depth_driven_135d_full0605}"
OUT_JSON="${OUT_JSON:-outputs/evaluation/prism_kt_spectral_epoch7_rw_merged0604/h3d/metrics_depth_driven_full0605.json}"
LOG_PREFIX="${LOG_PREFIX:-[prism_epoch7_h3d]}"

echo "${LOG_PREFIX} raw_npz=$(find "${RAW_DIR}" -maxdepth 1 -name '*.npz' | wc -l)"

python3 scripts/eval/convert_smplx_npz_dir_to_135d.py \
  --input-dir "${RAW_DIR}" \
  --output-dir "${NPY_DIR}" \
  --skip-existing \
  --progress-every 200

echo "${LOG_PREFIX} converted_npy=$(find "${NPY_DIR}" -maxdepth 1 -name '*.npy' | wc -l)"

python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --anno_file data/annotation/test_hml3d.json \
  --data_dir data/motionhub \
  --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
  --clip_pretrained checkpoints/clip-vit-base-patch32 \
  --stats_file data/statistic/smplx55_stats_hymotion_aug.json \
  --pred_dir "${NPY_DIR}" \
  --out_json "${OUT_JSON}" \
  --n_repeats 20 \
  --seed 42 \
  --rewritten_caption_file data/annotation/test_hml3d_rewritten.json \
  --chunk_size 64 \
  --forward_batch_size "${FORWARD_BATCH_SIZE:-32}"

echo "${LOG_PREFIX} wrote ${OUT_JSON}"
