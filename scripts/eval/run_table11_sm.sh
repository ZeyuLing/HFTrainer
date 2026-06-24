#!/bin/bash
# Table 11 StableMotion enhanced pipeline (run on Taiji debug machine)
set -x
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/StableMotion || exit 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== STEP pip install ema_pytorch ==="
python3 -m pip install -q ema_pytorch 2>&1 | tail -2

echo "=== STEP_FIX_ENHANCED (ensemble+SITS) ==="
python3 -m sample.fix_globsmpl \
  --model_path save/stablemotion/ema001000000.pt --use_ema --batch_size 32 \
  --testdata_dir dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --ensemble --enable_sits --classifier_scale 100 \
  --output_dir ./output/brokenamass_star_sm_enhanced

echo "=== STEP_CLEAN_GT ==="
python3 -m sample.fix_globsmpl \
  --model_path save/stablemotion/ema001000000.pt --use_ema --batch_size 32 \
  --testdata_dir dataset/AMASS_20.0_fps_nh_globsmpl_base_cano \
  --output_dir ./output/brokenamass_star_clean_v2 --collect_dataset

echo "=== STEP_EVAL_ENHANCED ==="
python3 -m eval.eval_scripts \
  --data_path output/brokenamass_star_sm_enhanced/results.npy \
  --gt_data_path output/brokenamass_star_clean_v2/results_collected.npy \
  --motiontypes motion_fix motion --force_redo

echo "=== ALL_DONE ==="
