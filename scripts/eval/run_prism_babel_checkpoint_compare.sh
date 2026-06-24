#!/bin/bash
# Compare BABEL sequential generation between:
#   1) latest KT-RoPE spectral_unified checkpoint
#   2) original no-KT sequential-RoPE iter_15000 checkpoint
#
# Typical Taiji use:
#   PHASE=gen JOB_RANK=<rank> JOB_COUNT=<nodes> bash scripts/eval/run_prism_babel_checkpoint_compare.sh
# After all generation shards finish:
#   PHASE=convert_eval bash scripts/eval/run_prism_babel_checkpoint_compare.sh
set -uo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
if [ ! -d "$ROOT" ]; then
  ROOT=/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
fi
cd "$ROOT"
export PYTHONPATH="$PWD:$PWD/ref_repo/MotionStreamer/MotionStreamer:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

PY=${PY:-python3}
PHASE=${PHASE:-all}  # gen | convert_eval | eval | all
OUT_ROOT=${OUT_ROOT:-outputs/evaluation/babel_seq/ckpt_compare_20260615}
MANIFEST=${MANIFEST:-outputs/evaluation/babel_seq/common_valid_manifest.jsonl}
NUM_GPUS=${NUM_GPUS:-8}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-5.0}
AR_COND_FRAMES=${AR_COND_FRAMES:-5}
KAFS_MODE=${KAFS_MODE:-none}
METHODS=${METHODS:-kt_latest iter15000}
RPREC_BATCH_SIZE=${RPREC_BATCH_SIZE:-8}

KT_CONFIG=${KT_CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
KT_SEQ_CONFIG=${KT_SEQ_CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_seqInfer.py}
KT_DFS_CONFIG=${KT_DFS_CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_dfsInfer.py}
KT_CKPT=${KT_CKPT:-$($PY - <<'PY'
from pathlib import Path
root = Path("work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached")
cands = []
for p in root.glob("checkpoint-epoch_*"):
    try:
        cands.append((int(p.name.rsplit("_", 1)[1]), p))
    except Exception:
        pass
if not cands:
    raise SystemExit("no KT checkpoint found")
print(max(cands)[1])
PY
)}
ITER15_CONFIG=${ITER15_CONFIG:-configs/prism/prism_1b_tp2m_multiframe_iter15k.py}
ITER15_CKPT=${ITER15_CKPT:-work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000}

if [ -n "${JOB_COUNT:-}" ]; then
  HOST_RANK=${JOB_RANK:-0}
  MACHINE_NUM=$JOB_COUNT
else
  HOST_RANK=${INDEX:-0}
  if [ -n "${NODE_LIST:-}" ]; then
    MACHINE_NUM=$($PY -c "import os; print(len(os.environ['NODE_LIST'].split(',')))" 2>/dev/null || echo 1)
  else
    MACHINE_NUM=${MACHINE_NUM:-1}
  fi
fi
TOTAL_SHARDS=$((MACHINE_NUM * NUM_GPUS))

method_config() {
  case "$1" in
    kt_latest|kt_spectral|spectral) echo "$KT_CONFIG|$KT_CKPT" ;;
    kt_seq|seq) echo "$KT_SEQ_CONFIG|$KT_CKPT" ;;
    kt_dfs|dfs) echo "$KT_DFS_CONFIG|$KT_CKPT" ;;
    iter15000) echo "$ITER15_CONFIG|$ITER15_CKPT" ;;
    *) echo "unknown method: $1" >&2; return 1 ;;
  esac
}

run_generation() {
  mkdir -p "$OUT_ROOT/_logs"
  echo "[compare-gen] phase=$PHASE host=$HOST_RANK/$MACHINE_NUM shards=$TOTAL_SHARDS methods=$METHODS"
  for method in $METHODS; do
    IFS='|' read -r cfg ckpt < <(method_config "$method")
    out="$OUT_ROOT/${method}_gen"
    mkdir -p "$out/logs"
    echo "[compare-gen] method=$method out=$out cfg=$cfg ckpt=$ckpt"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
      shard=$((HOST_RANK * NUM_GPUS + i))
      CUDA_VISIBLE_DEVICES=$i "$PY" -u scripts/eval/gen_prism_babel_seq.py \
        --config "$cfg" \
        --checkpoint "$ckpt" \
        --manifest "$MANIFEST" \
        --output-dir "$out" \
        --num-inference-steps "$STEPS" \
        --guidance-scale "$GUIDANCE" \
        --ar-cond-frames "$AR_COND_FRAMES" \
        --kafs-mode "$KAFS_MODE" \
        --rewrite-captions \
        --num-shards "$TOTAL_SHARDS" \
        --shard-idx "$shard" \
        --skip-existing \
        > "$out/logs/gen_h${HOST_RANK}_g${i}.log" 2>&1 &
    done
    wait
    n=$(find "$out" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
    echo "[compare-gen] method=$method host=$HOST_RANK done total_npz=$n"
  done
}

convert_one() {
  local method=$1
  local gen="$OUT_ROOT/${method}_gen"
  local f272="$OUT_ROOT/${method}_272f"
  mkdir -p "$OUT_ROOT/_logs"
  "$PY" scripts/eval/smpl_pred_to_272.py \
    --in-dir "$gen" \
    --out-dir "$f272" \
    --skip-existing \
    > "$OUT_ROOT/_logs/${method}_to272.log" 2>&1
  echo "[compare-convert] method=$method npz=$(find "$f272" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)"
}

eval_one() {
  local method=$1
  local f272="$OUT_ROOT/${method}_272f"
  local res="$OUT_ROOT/results"
  local n272
  n272=$(find "$f272" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  if [ "$n272" -eq 0 ]; then
    echo "[compare-eval] no 272 files for method=$method dir=$f272" >&2
    return 1
  fi
  mkdir -p "$res" "$OUT_ROOT/_logs"
  for proto in unique_dedup balanced_dedup balanced_nodedup random_nodedup; do
    case "$proto" in
      unique_dedup) extra=(--dedup --rprec-batching unique --rprec-batch-size "$RPREC_BATCH_SIZE") ;;
      balanced_dedup) extra=(--dedup --rprec-batching balanced) ;;
      balanced_nodedup) extra=(--no-dedup --rprec-batching balanced) ;;
      random_nodedup) extra=(--no-dedup --rprec-batching random) ;;
    esac
    "$PY" scripts/eval/eval_babel_seq_ms272.py \
      --manifest "$MANIFEST" \
      --pred-dir "$f272" \
      --tag "${method}_${proto}" \
      --out-json "$res/${method}_${proto}.json" \
      --max-total 360 \
      --mean-std humanml \
      --no-rewrite \
      --caption-template 'a person {cap}' \
      "${extra[@]}" \
      > "$OUT_ROOT/_logs/${method}_${proto}_eval.log" 2>&1
    tail -n 3 "$OUT_ROOT/_logs/${method}_${proto}_eval.log"
  done
}

case "$PHASE" in
  gen)
    run_generation
    ;;
  convert_eval)
    for method in $METHODS; do convert_one "$method"; done
    for method in $METHODS; do eval_one "$method"; done
    ;;
  eval)
    for method in $METHODS; do eval_one "$method"; done
    ;;
  all)
    run_generation
    for method in $METHODS; do convert_one "$method"; done
    for method in $METHODS; do eval_one "$method"; done
    ;;
  *)
    echo "unknown PHASE=$PHASE" >&2
    exit 2
    ;;
esac
