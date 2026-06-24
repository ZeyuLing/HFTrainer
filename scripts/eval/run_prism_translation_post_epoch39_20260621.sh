#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

RUN_ROOT="outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_translation_ablation_epoch39_20260621"

export RUN_ROOT
export ROLLOUT_NPZ_DIR="$RUN_ROOT/rollout/h3d/depth_driven"
export ABS_OUT="$RUN_ROOT/absolute/h3d"
export STAGE=post
export MS_DEVICE="${MS_DEVICE:-cuda}"
export MC_GPU="${MC_GPU:-0}"
export WORKERS="${WORKERS:-32}"

bash scripts/eval/run_prism_translation_ablation_20260619.sh
