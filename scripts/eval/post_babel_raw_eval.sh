#!/usr/bin/env bash
# After PRISM-raw / MS-raw generation finishes on Taiji, resample each method's
# concatenated motion to canonical lengths and run the FlowMDM-native BABEL
# evaluator. Produces /tmp/{prism,ms}_raw_eval.log for the raw-terse ablation.
set -uo pipefail
ROOT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$ROOT"
FM="$ROOT/ref_repo/FlowMDM"

PRISM_PC="$FM/results/babel/PRISM_e19_raw/evaluation_precomputed/Motion_PRISM_e19_raw_001300000_gscale1.5_debug_s10/00"
MS_PC="$FM/results/babel/MotionStreamer_raw/evaluation_precomputed/Motion_MotionStreamer_raw_001300000_gscale1.5_debug_s10/00"

echo "=== resample PRISM-raw (seg-mode prism) ==="
python3 scripts/eval/resample_prism_precomp_to_canonical.py --precomp-dir "$PRISM_PC" --seg-mode prism
echo "=== resample MS-raw (seg-mode sidecar) ==="
python3 scripts/eval/resample_prism_precomp_to_canonical.py --precomp-dir "$MS_PC" --seg-mode sidecar

cd "$FM"
echo "=== eval PRISM-raw ==="
python3 -m runners.eval --model_path ./results/babel/PRISM_e19_raw/model001300000.pt \
  --dataset babel --eval_mode debug --bpe_denoising_step 125 --guidance_param 1.5 \
  --transition_length 30 > /tmp/prism_raw_eval.log 2>&1
echo "PRISM_RAW_EVAL_EXIT=$?"
echo "=== eval MS-raw ==="
python3 -m runners.eval --model_path ./results/babel/MotionStreamer_raw/model001300000.pt \
  --dataset babel --eval_mode debug --bpe_denoising_step 125 --guidance_param 1.5 \
  --transition_length 30 > /tmp/ms_raw_eval.log 2>&1
echo "MS_RAW_EVAL_EXIT=$?"
echo "POST_BABEL_RAW_EVAL_DONE"
