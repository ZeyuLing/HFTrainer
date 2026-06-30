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

METHOD=${METHOD:?METHOD must be motionstreamer or prism}
NUM_GPUS=${NUM_GPUS:-8}
HOST_RANK=${HOST_RANK:-${INDEX:-${JOB_RANK:-0}}}
MACHINE_NUM=${MACHINE_NUM:-${JOB_COUNT:-1}}
TOTAL_SHARDS=${TOTAL_SHARDS_OVERRIDE:-$((NUM_GPUS * MACHINE_NUM))}
SHARD_OFFSET=${SHARD_OFFSET:-0}
MANIFEST=${MANIFEST:-outputs/evaluation/babel/official_val/msstyle_30fps_gt/manifest.jsonl}
ROOT=${ROOT:-outputs/evaluation/babel/official_val/msstyle_30fps_gt}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs/infer_${METHOD}}
PRISM_OUT_DIR=${PRISM_OUT_DIR:-$ROOT/prism_gen}
mkdir -p "$LOG_ROOT"

echo "[official-babel-$METHOD] host=$HOST_RANK machines=$MACHINE_NUM gpus=$NUM_GPUS total_shards=$TOTAL_SHARDS shard_offset=$SHARD_OFFSET manifest=$MANIFEST"

pids=()
ms_t5_args=()
if [ -n "${MS_T5_MODEL:-}" ]; then
  ms_t5_args=(--t5-model "$MS_T5_MODEL")
fi
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  shard=$((SHARD_OFFSET + HOST_RANK * NUM_GPUS + gpu))
  if [ "$METHOD" = "motionstreamer" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/eval/gen_motionstreamer_babel_official_seq.py \
      --manifest "$MANIFEST" \
      --out-dir "$ROOT/motionstreamer_gen" \
      --num-shards "$TOTAL_SHARDS" \
      --shard-index "$shard" \
      --cfg "${MS_CFG:-4.0}" \
      --context-tokens "${MS_CONTEXT_TOKENS:-16}" \
      --max-episodes "${MAX_EPISODES:-0}" \
      "${ms_t5_args[@]}" \
      --skip-existing \
      > "$LOG_ROOT/worker_h${HOST_RANK}_g${gpu}_s${shard}.log" 2>&1 &
    pids+=("$!")
  elif [ "$METHOD" = "prism" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/eval/gen_prism_babel_official_seq.py \
      --manifest "$MANIFEST" \
      --output-dir "$PRISM_OUT_DIR" \
      --num-shards "$TOTAL_SHARDS" \
      --shard-idx "$shard" \
      --config "${PRISM_CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}" \
      --checkpoint "${PRISM_CHECKPOINT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_16}" \
      --guidance-scale "${PRISM_CFG:-5.0}" \
      --kafs-mode "${PRISM_KAFS_MODE:-none}" \
      --ar-cond-frames "${PRISM_AR_COND_FRAMES:-5}" \
      --max-episodes "${MAX_EPISODES:-0}" \
      --skip-existing \
      > "$LOG_ROOT/worker_h${HOST_RANK}_g${gpu}_s${shard}.log" 2>&1 &
    pids+=("$!")
  else
    echo "unknown METHOD=$METHOD" >&2
    exit 2
  fi
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

if [ "$METHOD" = "prism" ] && [ "${RUN_REPACK:-0}" = "1" ]; then
  mkdir -p "$ROOT/prism_prep135"
  python3 scripts/eval/repack_pred_to_272ids.py \
    --npz-dir "$PRISM_OUT_DIR" \
    --anno-file data/annotation/test_hml3d.json \
    --id-passthrough \
    --out-dir "$ROOT/prism_prep135" \
    --workers "${REPACK_WORKERS:-8}" \
    > "$LOG_ROOT/repack_h${HOST_RANK}.log" 2>&1
fi

echo "[official-babel-$METHOD done] host=$HOST_RANK"
