#!/usr/bin/env bash
# Evaluate PRISM epoch-43 BABEL official-val pad360_crop outputs.
# Converts generated SMPL(X) sequences to MotionStreamer-272 and runs the
# BABEL sequential-action evaluator on the official 30 FPS manifest.
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "${ROOT}" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "${ROOT}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

PY=${PY:-python3}
GPU=${GPU:-0}
WORKERS=${WORKERS:-16}
BASE=${BASE:-outputs/evaluation/babel/official_val/msstyle_30fps_gt}
GEN=${GEN:-${BASE}/prism_epoch43_pad360crop_arcond5_depth_driven}
PRED272=${PRED272:-${GEN}_272f}
MANIFEST=${MANIFEST:-${BASE}/manifest.jsonl}
GT_STREAM_DIR=${GT_STREAM_DIR:-${BASE}/gt_272_stream_yup}
METRICS=${METRICS:-${BASE}/metrics}
TAG=${TAG:-prism_epoch43_pad360crop_arcond5_depth}
OUT_JSON=${OUT_JSON:-${METRICS}/${TAG}_ms272_eval_20260628.json}
LOG=${LOG:-${GEN}/eval_logs}
EXPECTED=${EXPECTED:-1295}
SKIP_CACHE=${SKIP_CACHE:-0}

mkdir -p "$PRED272" "$METRICS" "$LOG"

if [ "$SKIP_CACHE" = "1" ]; then
  echo "[cache] skipped by SKIP_CACHE=1" > "$LOG/cache.log"
else
  bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true
fi

src_n=$(find "$GEN" -maxdepth 1 -name 'val_*.npz' 2>/dev/null | wc -l)
if [ "$src_n" -lt "$EXPECTED" ]; then
  echo "[FAIL] source coverage $src_n/$EXPECTED under $GEN" | tee "$LOG/run.log"
  exit 2
fi

cat > "$LOG/command.txt" <<EOF
ROOT=$ROOT
GEN=$GEN
PRED272=$PRED272
MANIFEST=$MANIFEST
TAG=$TAG
OUT_JSON=$OUT_JSON
GT_STREAM_DIR=$GT_STREAM_DIR
EXPECTED=$EXPECTED
WORKERS=$WORKERS
EOF

echo "[start] $(date) gen=$GEN pred272=$PRED272 expected=$EXPECTED" | tee "$LOG/run.log"

"$PY" scripts/eval/smpl_pred_to_272.py \
  --in-dir "$GEN" \
  --out-dir "$PRED272" \
  --skip-existing \
  > "$LOG/smpl_pred_to_272.log" 2>&1

pred_n=$(find "$PRED272" -maxdepth 1 -name 'val_*.npz' 2>/dev/null | wc -l)
if [ "$pred_n" -lt "$EXPECTED" ]; then
  echo "[FAIL] pred272 coverage $pred_n/$EXPECTED under $PRED272" | tee -a "$LOG/run.log"
  exit 3
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval/eval_babel_seq_ms272.py \
  --manifest "$MANIFEST" \
  --pred-dir "$PRED272" \
  --gt-stream-dir "$GT_STREAM_DIR" \
  --tag "$TAG" \
  --out-json "$OUT_JSON" \
  --mean-std humanml \
  --dedup \
  --rprec-batching balanced \
  > "$LOG/eval_babel_seq_ms272.log" 2>&1

touch "$LOG/_DONE"
echo "[done] $(date) -> $OUT_JSON" | tee -a "$LOG/run.log"
