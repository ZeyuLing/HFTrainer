#!/usr/bin/env bash
# End-to-end HumanML3D official-test ViMoGen rerun using DeepSeek-rewritten
# ViMoGen-style captions.
#
# Required:
#   export DEEPSEEK_API_KEY=...
#
# Optional knobs:
#   CAP_TAG=vimogen_deepseek_motion_detailed_20260628
#   RUN_TAG=vimogen_1_3b_deepseek_caption_20260628
#   CUDA_DEVICES=0,1,2,3,4,5,6,7
#   NUM_SHARDS=8
#   REWRITE_EXTRA_ARGS="--max-samples 64"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
CAP_TAG="${CAP_TAG:-vimogen_deepseek_motion_detailed_${DATE_TAG}}"
RUN_TAG="${RUN_TAG:-vimogen_1_3b_deepseek_caption_${DATE_TAG}}"

CAP_DIR="${CAP_DIR:-$ROOT/outputs/evaluation/t2m/humanml3d_official_test/captions/$CAP_TAG}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/evaluation/t2m/humanml3d_official_test/_runs/$RUN_TAG}"
EMBED_DIR="${EMBED_DIR:-$RUN_DIR/text_embeddings}"
EVAL_JSON="${EVAL_JSON:-$RUN_DIR/vimogen_h3d.json}"
EVAL_CAPTION_MAP="${EVAL_CAPTION_MAP:-$RUN_DIR/caption_map.json}"
VIMOGEN276_DIR="${VIMOGEN276_DIR:-$ROOT/outputs/evaluation/t2m/humanml3d_official_test/vimogen276/vimogen_1_3b_deepseek_caption}"
MOTION135_DIR="${MOTION135_DIR:-$ROOT/outputs/evaluation/t2m/humanml3d_official_test/motion135/vimogen_1_3b_deepseek_caption}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"

MODEL_PATH="${MODEL_PATH:-$ROOT/checkpoints/vimogen/hftrainer_1_3b}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
IFS=',' read -r -a DEVICE_LIST <<< "$CUDA_DEVICES"
NUM_SHARDS="${NUM_SHARDS:-${#DEVICE_LIST[@]}}"
TEXT_CUDA_VISIBLE_DEVICES="${TEXT_CUDA_VISIBLE_DEVICES:-${DEVICE_LIST[0]}}"

BATCH_SIZE="${BATCH_SIZE:-64}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-4}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bf16}"
CFG_SCALE="${CFG_SCALE:-5.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
DENOISING_STRENGTH="${DENOISING_STRENGTH:-0.7}"
REWRITE_EXTRA_ARGS="${REWRITE_EXTRA_ARGS:-}"

mkdir -p "$CAP_DIR" "$RUN_DIR" "$EMBED_DIR" "$VIMOGEN276_DIR" "$MOTION135_DIR" "$LOG_DIR"

echo "[1/5] rewrite captions -> $CAP_DIR"
python3 scripts/eval/rewrite_hml3d_captions_vimogen_deepseek.py \
  --out-dir "$CAP_DIR" \
  $REWRITE_EXTRA_ARGS \
  2>&1 | tee "$LOG_DIR/rewrite.log"

echo "[2/5] build ViMoGen eval json -> $EVAL_JSON"
python3 scripts/eval/build_vimogen_eval_json.py \
  --anno-file "$CAP_DIR/test_hml3d_official272_gtlen_vimogen_deepseek_caption.json" \
  --data-dir "$ROOT" \
  --caption-override-json "$CAP_DIR/caption_map.json" \
  --out-json "$EVAL_JSON" \
  --caption-map-json "$EVAL_CAPTION_MAP" \
  --embedding-dir "$EMBED_DIR" \
  --caption-style first \
  2>&1 | tee "$LOG_DIR/build_eval_json.log"

echo "[3/5] encode text embeddings -> $EMBED_DIR"
CUDA_VISIBLE_DEVICES="$TEXT_CUDA_VISIBLE_DEVICES" \
PYTHONPATH="$ROOT/ref_repo/ViMoGen:${PYTHONPATH:-}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
VIMOGEN_TEXT_BATCH_SIZE="$TEXT_BATCH_SIZE" \
python3 ref_repo/ViMoGen/models/transformer/wan/text_encoding_batch.py \
  --json_file "$EVAL_JSON" \
  --text_key prompt \
  --save_dir "$EMBED_DIR" \
  --batch_size "$TEXT_BATCH_SIZE" \
  2>&1 | tee "$LOG_DIR/text_embed.log"

EXPECTED_COUNT="$(python3 - "$EVAL_JSON" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1]))))
PY
)"
EMBED_COUNT="$(find "$EMBED_DIR/prompt" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$EMBED_COUNT" != "$EXPECTED_COUNT" ]]; then
  echo "[error] text embedding incomplete: expected=$EXPECTED_COUNT actual=$EMBED_COUNT dir=$EMBED_DIR/prompt" >&2
  exit 2
fi

echo "[4/5] ViMoGen inference -> $VIMOGEN276_DIR"
pids=()
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
  device="${DEVICE_LIST[$((shard_idx % ${#DEVICE_LIST[@]}))]}"
  log="$LOG_DIR/infer_shard${shard_idx}.log"
  (
    CUDA_VISIBLE_DEVICES="$device" \
    HFTRAINER_SKIP_AUTOREGISTER=1 \
    PYTHONPATH="$ROOT:${PYTHONPATH:-}" \
    python3 scripts/eval/vimogen_t2m_humanml3d.py \
      --eval-json "$EVAL_JSON" \
      --model-path "$MODEL_PATH" \
      --out-dir "$VIMOGEN276_DIR" \
      --batch-size "$BATCH_SIZE" \
      --seed "$SEED" \
      --dtype "$DTYPE" \
      --cfg-scale "$CFG_SCALE" \
      --num-inference-steps "$NUM_INFERENCE_STEPS" \
      --denoising-strength "$DENOISING_STRENGTH" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard_idx" \
      --skip-existing
  ) >"$log" 2>&1 &
  pids+=("$!")
  echo "  shard $shard_idx/$NUM_SHARDS on cuda:$device -> $log"
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

VIMOGEN276_COUNT="$(find "$VIMOGEN276_DIR" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$VIMOGEN276_COUNT" != "$EXPECTED_COUNT" ]]; then
  echo "[error] ViMoGen inference incomplete: expected=$EXPECTED_COUNT actual=$VIMOGEN276_COUNT dir=$VIMOGEN276_DIR" >&2
  exit 3
fi

echo "[5/5] convert 276D -> motion135/SMPL-style -> $MOTION135_DIR"
HFTRAINER_SKIP_AUTOREGISTER=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}" \
python3 scripts/eval/convert_vimogen276_to_motionclip135.py \
  --input-root "$VIMOGEN276_DIR" \
  --out-dir "$MOTION135_DIR" \
  --overwrite \
  --src-fps 20 \
  --dst-fps 30 \
  --max-frames 300 \
  --coord-conversion mbench \
  --translation-source floor_aligned_smpl_transl \
  --rotation-convention row \
  2>&1 | tee "$LOG_DIR/convert_motion135.log"

MOTION135_COUNT="$(find "$MOTION135_DIR" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$MOTION135_COUNT" != "$EXPECTED_COUNT" ]]; then
  echo "[error] motion135 conversion incomplete: expected=$EXPECTED_COUNT actual=$MOTION135_COUNT dir=$MOTION135_DIR" >&2
  exit 4
fi

echo "[done]"
echo "caption_dir=$CAP_DIR"
echo "eval_json=$EVAL_JSON"
echo "vimogen276_dir=$VIMOGEN276_DIR"
echo "motion135_dir=$MOTION135_DIR"
find "$MOTION135_DIR" -maxdepth 1 -type f -name '*.npz' | wc -l | awk '{print "motion135_count="$1}'
