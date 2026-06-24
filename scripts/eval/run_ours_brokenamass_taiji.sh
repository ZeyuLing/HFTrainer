#!/bin/bash
# Run OUR M2M repair on BrokenAMASS* (Taiji debug machine) via the canonical
# HyMotionM2MPipeline.infer_repair entry point. MODE via $1:
#   identity      -> conversion round-trip sanity (no model), 30 samples
#   strict_sdedit -> proven best: provided(MoGenDIT) mask + lock root + τ=0.5
#   self_denoise  -> ours self-detected mask + lock root + τ=0.5
set -x
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer || exit 1
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

SM_OUT=ref_repo/StableMotion/output
SMRES=$SM_OUT/brokenamass_star_sm_enhanced/results.npy
GT=$SM_OUT/brokenamass_star_clean_v2/results_collected.npy

MODE=${1:-strict_sdedit}
if [ "$MODE" = "identity" ]; then
  python3 scripts/eval/run_ours_repair_brokenamass.py \
    --sm-results $SMRES --gt $GT \
    --output-dir $SM_OUT/brokenamass_star_ours_identity \
    --identity-check --max-samples 30
elif [ "$MODE" = "self_denoise" ]; then
  python3 scripts/eval/run_ours_repair_brokenamass.py \
    --sm-results $SMRES --gt $GT \
    --output-dir $SM_OUT/brokenamass_star_ours_selfdenoise \
    --mask-source self_denoise --translation-mode lock \
    --mask-granularity joint --sdedit-tau 0.5 \
    --max-samples 9999
else  # strict_sdedit (proven best)
  python3 scripts/eval/run_ours_repair_brokenamass.py \
    --sm-results $SMRES --gt $GT \
    --output-dir $SM_OUT/brokenamass_star_ours_strict_sd \
    --mask-source provided --translation-mode lock \
    --mask-granularity joint --sdedit-tau 0.5 \
    --max-samples 9999
fi
echo "=== OURS_DONE ($MODE) ==="
