#!/usr/bin/env bash
# Compute the unified Table-11 geometry and uTMR metrics after all KIMODO
# position-condition predictions have been generated.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1

RUN_ID=${RUN_ID:-official_20260725_position}
EXPECTED=${EXPECTED_SAMPLES:-4012}
CANONICAL_ROOT=${CANONICAL_ROOT:-outputs/evaluation/body_part/humanml3d_official_test_4012}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
UTMR_JOBS_PER_GPU=${UTMR_JOBS_PER_GPU:-1}
UTMR_LOAD_WORKERS=${UTMR_LOAD_WORKERS:-4}

global_marker="$CANONICAL_ROOT/_KIMODO_POSITION_TABLE11_${RUN_ID}_DONE"
exec 8>"$CANONICAL_ROOT/_KIMODO_POSITION_TABLE11_${RUN_ID}.lock"
flock 8
if [ -s "$global_marker" ]; then
  echo "[skip] KIMODO Table-11 position metrics already complete"
  exit 0
fi

targets=(
  upper lower
  wrist_left wrist_right wrist_both
  elbow_left elbow_right elbow_both
  foot_left foot_right foot_both
  knee_left knee_right knee_both
)
settings=()
for target in "${targets[@]}"; do
  for density in sparse dense; do
    for axes in xz xyz; do
      settings+=("E17_${target}_${density}_${axes}")
    done
  done
done
if [ -n "${SETTINGS_OVERRIDE:-}" ]; then
  IFS=',' read -r -a settings <<< "$SETTINGS_OVERRIDE"
fi

for setting in "${settings[@]}"; do
  base="$CANONICAL_ROOT/$setting/kimodo/$RUN_ID"
  count=$(find -L "$base/npz" -maxdepth 1 -type f -name '*.npz' | wc -l)
  if [ "$count" -ne "$EXPECTED" ]; then
    echo "$setting incomplete: expected=$EXPECTED actual=$count" >&2
    exit 3
  fi
  mkdir -p "$base/metrics" "$base/logs"
done

fail=0
for setting in "${settings[@]}"; do
  while [ "$(jobs -pr | wc -l)" -ge 8 ]; do wait -n || fail=1; done
  base="$CANONICAL_ROOT/$setting/kimodo/$RUN_ID"
  if [ -s "$base/metrics/geometry.json" ]; then
    continue
  fi
  (
    python3 scripts/eval/score_bodypart_position_baseline_4012.py \
      --npz-dir "$base/npz" --setting "$setting" --method kimodo \
      --expected-samples "$EXPECTED" --out "$base/metrics/geometry.json"
  ) > "$base/logs/geometry.log" 2>&1 &
done
wait || fail=1
(( fail == 0 )) || exit 4

IFS=',' read -r -a gpu_ids <<< "$GPUS"
max_jobs=$((${#gpu_ids[@]} * UTMR_JOBS_PER_GPU))
fail=0
for index in "${!settings[@]}"; do
  while [ "$(jobs -pr | wc -l)" -ge "$max_jobs" ]; do wait -n || fail=1; done
  setting=${settings[$index]}
  gpu=${gpu_ids[$((index % ${#gpu_ids[@]}))]}
  base="$CANONICAL_ROOT/$setting/kimodo/$RUN_ID"
  if [ -s "$base/metrics/utmr.json" ]; then
    continue
  fi
  (
    CUDA_VISIBLE_DEVICES="$gpu" env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
      python3 scripts/eval/eval_npz_universal_tmr_fid.py \
        --pred-npz-dir "$base/npz" --tag "kimodo_${setting}" \
        --load-workers "$UTMR_LOAD_WORKERS" \
        --out-json "$base/metrics/utmr.json"
  ) > "$base/logs/utmr.log" 2>&1 &
done
wait || fail=1
(( fail == 0 )) || exit 5

for setting in "${settings[@]}"; do
  base="$CANONICAL_ROOT/$setting/kimodo/$RUN_ID"
  test -s "$base/metrics/geometry.json"
  test -s "$base/metrics/utmr.json"
  date -Is > "$base/DONE"
done
date -Is > "$global_marker"
echo "[done] KIMODO Table-11 position metrics"
