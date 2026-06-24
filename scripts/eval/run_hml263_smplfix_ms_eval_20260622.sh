#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HFTRAINER_SKIP_AUTOREGISTER=1

RUN_ID="${RUN_ID:-table1_hml263_smplfix_20260622}"
SUITE="outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/$RUN_ID"
RES="$SUITE/results"
LOG="$SUITE/logs"
mkdir -p "$RES" "$LOG"

METHODS="${METHODS:-mdm,motiongpt3,flowmdm,motionlab}"
MIN_MOTION_LEN="${MIN_MOTION_LEN:-60}"
MAX_MOTION_LENGTH="${MAX_MOTION_LENGTH:-300}"
SEED="${SEED:-0}"
FORCE_EVAL="${FORCE_EVAL:-1}"

if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
else
  NGPU="${NGPU:-${TJ_GPU_NUM:-1}}"
  GPU_IDS=()
  for ((g=0; g<NGPU; g++)); do
    GPU_IDS+=("$g")
  done
fi
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "[error] empty GPU list" >&2
  exit 2
fi

bash scripts/eval/_cache_272_data.sh > "$LOG/cache_eval.log" 2>&1 || true
if [[ ! -f /dev/shm/eval272_epoch99.ckpt ]]; then
  cp ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt /dev/shm/eval272_epoch99.ckpt 2>/dev/null || true
fi

IFS=',' read -r -a METHOD_ARR <<< "$METHODS"
for method in "${METHOD_ARR[@]}"; do
  pred="$SUITE/prep/$method"
  n="$(find "$pred" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)"
  echo "[coverage] $method prep_npz=$n dir=$pred" | tee -a "$LOG/eval_smplfix.run.log"
  if [[ "$n" -lt 4042 ]]; then
    echo "[error] incomplete prep for $method: $n/4042" >&2
    exit 3
  fi
done

if [[ "$FORCE_EVAL" == "1" || ! -s "$RES/gt_native_272.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python3 scripts/eval/eval_motionstreamer_272.py \
    --tag gt_native_272 \
    --also-refk \
    --seed "$SEED" \
    --min-motion-len "$MIN_MOTION_LEN" \
    --max-motion-length "$MAX_MOTION_LENGTH" \
    --out-json "$RES/gt_native_272.json" \
    > "$LOG/eval_gt_native_272.log" 2>&1
fi

pids=()
idx=0
for method in "${METHOD_ARR[@]}"; do
  pred="$SUITE/prep/$method"
  gpu="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
      --pred-dir "$pred" \
      --tag "$method" \
      --also-refk \
      --seed "$SEED" \
      --min-motion-len "$MIN_MOTION_LEN" \
      --max-motion-length "$MAX_MOTION_LENGTH" \
      --out-json "$RES/${method}.json" \
      > "$LOG/eval_${method}.log" 2>&1
  ) &
  pids+=("$!")
  idx=$((idx + 1))
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done
if [[ "$rc" != "0" ]]; then
  echo "[error] at least one evaluator failed" >&2
  exit "$rc"
fi

python3 scripts/eval/_agg_ms272_tables.py \
  --res-dir "$RES" \
  --out "$SUITE/summary_ms_eval.json" \
  | tee "$SUITE/summary_ms_eval.txt"
