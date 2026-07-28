#!/usr/bin/env bash
# Generate, retarget, and score capability-matched held-out-layout baselines.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs/AILab_DHA/taijifs_cq11/share_1467498/home/zeyuling/hf_trainer}
METHODS=${METHODS:-"condmdi omnicontrol maskcontrol projflow"}
SETTINGS=${SETTINGS:-"I1 H1 I2 H2 I3 H3"}
MAX_SAMPLES=${MAX_SAMPLES:-16}
RUN_NAME=${RUN_NAME:-pilot16_20260727}
MASKCONTROL_PROFILE=${MASKCONTROL_PROFILE:-paper}
GPU=${GPU:-0}

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HFTRAINER_SKIP_AUTOREGISTER=1

for method in $METHODS; do
  for setting in $SETTINGS; do
    if [[ "$method" == "maskcontrol" && ( "$setting" == "I3" || "$setting" == "H3" ) ]]; then
      echo "[skip] MaskControl cannot receive $setting full-body keyposes"
      continue
    fi

    base="$ROOT/outputs/evaluation/heldout_condition_layout/baselines/$setting/$method/$RUN_NAME"
    mkdir -p "$base/ik_motion135" "$base/eval_npz" "$base/metrics" "$base/logs"
    if [ "$method" = "projflow" ]; then
      prediction_dir="$base/joints22"
    else
      prediction_dir="$base/hml263"
    fi

    CUDA_VISIBLE_DEVICES="$GPU" python3 -u \
      scripts/eval/run_mixed_control_position_baseline_4012.py \
      --method "$method" \
      --setting "$setting" \
      --max-samples "$MAX_SAMPLES" \
      --run-name "$RUN_NAME" \
      --device cuda \
      --maskcontrol-profile "$MASKCONTROL_PROFILE" \
      --skip-existing \
      2>&1 | tee "$base/logs/generation.log"

    CUDA_VISIBLE_DEVICES="$GPU" python3 -u scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "$prediction_dir" \
      --out-dir "$base/ik_motion135" \
      --model-dir ref_repo/MDM/body_models \
      --source-fps 20 \
      --target-fps 30 \
      --device cuda \
      --batch-size 256 \
      --floor-align \
      --refine-iters 0 \
      --rotation-init position_ik \
      --skip-existing \
      --target-length-anno data/eval/m2m_v2/eval_hml3d_official_control_4012.json \
      --num-shards 1 \
      --shard-index 0 \
      2>&1 | tee "$base/logs/ik.log"

    python3 scripts/eval/build_mixed_control_baseline_eval_npz_4012.py \
      --ik-dir "$base/ik_motion135" \
      --setting "$setting" \
      --method "$method" \
      --out-dir "$base/eval_npz" \
      --expected-samples "$MAX_SAMPLES" \
      --max-samples "$MAX_SAMPLES" \
      --skip-existing \
      2>&1 | tee "$base/logs/package.log"

    python3 scripts/eval/score_mixed_control_4012.py \
      --npz-dir "$base/eval_npz" \
      --setting "$setting" \
      --method "$method" \
      --out "$base/metrics/geometry.json" \
      --expected-samples "$MAX_SAMPLES" \
      2>&1 | tee "$base/logs/geometry.log"
  done
done
