#!/usr/bin/env bash
set -euo pipefail

HF=${HF:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "$HF" ]; then
  HF=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$HF"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HFTRAINER_SKIP_AUTOREGISTER=1
export PYTHONPATH="$HF:${PYTHONPATH:-}"
PYTHON_BIN=${PYTHON_BIN:-python3}

METHOD=${METHOD:?METHOD must be flowmdm or doubletake}
case "$METHOD" in
  flowmdm|doubletake) ;;
  *) echo "unknown METHOD=$METHOD" >&2; exit 2 ;;
esac

NUM_GPUS=${NUM_GPUS:-8}
HOST_RANK=${HOST_RANK:-${INDEX:-${JOB_RANK:-0}}}
MACHINE_NUM=${MACHINE_NUM:-${JOB_COUNT:-1}}
TOTAL_SHARDS=${TOTAL_SHARDS_OVERRIDE:-$((NUM_GPUS * MACHINE_NUM))}
SHARD_OFFSET=${SHARD_OFFSET:-0}
MANIFEST=${MANIFEST:-outputs/evaluation/babel/official_val/msstyle_30fps_gt/manifest.jsonl}
ROOT=${ROOT:-outputs/evaluation/babel/official_val/msstyle_30fps_gt}
OUT_DIR=${OUT_DIR:-$ROOT/${METHOD}_gen}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs/infer_${METHOD}}
mkdir -p "$LOG_ROOT" "$OUT_DIR"

echo "[official-babel-$METHOD] host=$HOST_RANK machines=$MACHINE_NUM gpus=$NUM_GPUS total_shards=$TOTAL_SHARDS shard_offset=$SHARD_OFFSET manifest=$MANIFEST out=$OUT_DIR"

pids=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  shard=$((SHARD_OFFSET + HOST_RANK * NUM_GPUS + gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" -u scripts/eval/gen_vm_babel_official_seq.py \
    --method "$METHOD" \
    --manifest "$MANIFEST" \
    --output-dir "$OUT_DIR" \
    --num-shards "$TOTAL_SHARDS" \
    --shard-index "$shard" \
    --max-episodes "${MAX_EPISODES:-0}" \
    --seed "${SEED:-42}" \
    --flow-guidance-param "${FLOW_GUIDANCE:-1.5}" \
    --flow-bpe-denoising-step "${FLOW_BPE_STEP:-125}" \
    ${FLOW_CHUNKED_ATT:-"--flow-use-chunked-att"} \
    --doubletake-guidance-param "${DOUBLETAKE_GUIDANCE:-2.5}" \
    --handshake-size "${HANDSHAKE_SIZE:-20}" \
    --blend-len "${BLEND_LEN:-20}" \
    --skip-steps-double-take "${SKIP_STEPS_DOUBLE_TAKE:-100}" \
    --skip-existing \
    > "$LOG_ROOT/worker_h${HOST_RANK}_g${gpu}_s${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  echo "[official-babel-$METHOD failed] at least one worker exited non-zero" >&2
  exit "$status"
fi

echo "[official-babel-$METHOD done] host=$HOST_RANK count=$(find "$OUT_DIR" -maxdepth 1 -name 'val_*.npz' 2>/dev/null | wc -l)"
