#!/bin/bash
# Table 11 StableMotion VANILLA pipeline (plain DDPM, no ensemble/SITS/CFG).
# Compares against the enhanced run to check whether enhanced over-edits.
# GT (results_collected.npy) is reused from the enhanced run.
set -x
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/StableMotion || exit 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== STEP pip install ema_pytorch ==="
python3 -m pip install -q ema_pytorch 2>&1 | tail -2

echo "=== STEP_FIX_VANILLA (plain DDPM) ==="
python3 -m sample.fix_globsmpl \
  --model_path save/stablemotion/ema001000000.pt --use_ema --batch_size 32 \
  --testdata_dir dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --output_dir ./output/brokenamass_star_sm_vanilla

echo "=== STEP_CLEAN_GT (reuse if present) ==="
if [ ! -f output/brokenamass_star_clean_v2/results_collected.npy ]; then
  python3 -m sample.fix_globsmpl \
    --model_path save/stablemotion/ema001000000.pt --use_ema --batch_size 32 \
    --testdata_dir dataset/AMASS_20.0_fps_nh_globsmpl_base_cano \
    --output_dir ./output/brokenamass_star_clean_v2 --collect_dataset
else
  echo "GT already collected, skip"
fi

echo "=== STEP_EVAL_VANILLA ==="
python3 -m eval.eval_scripts \
  --data_path output/brokenamass_star_sm_vanilla/results.npy \
  --gt_data_path output/brokenamass_star_clean_v2/results_collected.npy \
  --motiontypes motion_fix motion --force_redo

echo "=== ALL_DONE ==="
