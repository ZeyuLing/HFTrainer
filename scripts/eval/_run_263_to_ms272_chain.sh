#!/usr/bin/env bash
# 263-baseline -> MS-272 evaluation, replicating the validated MDM chain
# (Stage A IK refine-80 + floor-align + row, Stage B 135->272, Stage C MS-272).
set -eo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export HFTRAINER_SKIP_AUTOREGISTER=1

BASE=outputs/evaluation/ms272_from263
declare -A SRC=(
  [t2mgpt]=outputs/evaluation/t2mgpt_h3d263_official/t2mgpt_263
  [momask]=outputs/evaluation/momask_h3d263_official/momask_263
)

for M in t2mgpt momask; do
  DIR263=${SRC[$M]}
  DIR135=$BASE/${M}_smpl135
  DIR272=$BASE/${M}_272
  echo "==================== [$M] Stage A: 263 -> SMPL motion_135 (refine 80) ===================="
  python3 -u scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$DIR263" --out-dir "$DIR135" \
    --model-dir ref_repo/MDM/body_models \
    --source-fps 20 --target-fps 30 --floor-align \
    --refine-iters 80 --refine-lr 0.02 \
    --rot6d-convention row \
    --device cuda --skip-existing
  echo "==================== [$M] Stage B: motion_135 -> 272 ===================="
  python3 -u scripts/data/convert_motion135_to_h3d272.py \
    --in-dir "$DIR135" --out-dir "$DIR272" --workers 8 --skip-existing
  echo "==================== [$M] done converting ($(ls "$DIR272"/*.npy 2>/dev/null | wc -l) files) ===================="
done

echo "==================== Stage C: MS-272 evaluation (t2mgpt, momask) ===================="
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/_ms272_eval_263baselines.py
echo "==================== ALL DONE ===================="
