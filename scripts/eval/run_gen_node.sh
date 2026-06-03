#!/bin/bash
# Generic single-node sharded PRISM T2M generation for multi-node parallelism.
# Each node runs NGPU local GPUs mapped to global shard indices
# [SHARD_START, SHARD_START+NGPU) out of NSHARDS total shards.
# All nodes for the same (dataset, mode) must use the SAME NSHARDS and write to
# the SAME OUT dir; shard ranges must be disjoint across nodes.
#
# Env (with defaults):
#   CONFIG CKPT MODE ANNO DATA OUT NSHARDS SHARD_START NGPU NUM_INFER
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

CONFIG=${CONFIG:-configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py}
CKPT=${CKPT:-work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_1}
MODE=${MODE:-none}
ANNO=${ANNO:-data/annotation/test_motionhub_t2m.json}
DATA=${DATA:-data/motionhub}
OUT=${OUT:-outputs/evaluation/prism_kt_spectral_epoch1/kafs}
NSHARDS=${NSHARDS:-8}
SHARD_START=${SHARD_START:-0}
NGPU=${NGPU:-8}
NUM_INFER=${NUM_INFER:-50}
# Optional: standalone {motion_id: caption} JSON to override GENERATION captions
# with rewritten ones (main-table protocol: generate on rewritten, eval on original).
REWRITTEN=${REWRITTEN:-}

REWRITTEN_ARG=""
if [ -n "$REWRITTEN" ]; then
    REWRITTEN_ARG="--rewritten-caption-file $REWRITTEN"
fi

# Optional extra CLI args forwarded verbatim to the eval script.
EXTRA_ARGS=${EXTRA_ARGS:-}

LOGDIR=$OUT/_logs
mkdir -p "$LOGDIR"
echo "[gen-node] start $(date) MODE=$MODE OUT=$OUT NSHARDS=$NSHARDS SHARD_START=$SHARD_START NGPU=$NGPU" | tee -a "$LOGDIR/nodes.log"

pids=()
for g in $(seq 0 $((NGPU-1))); do
    sid=$((SHARD_START+g))
    CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/eval_prism_kafs_ablation.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --kafs-mode "$MODE" \
        --anno-file "$ANNO" \
        $REWRITTEN_ARG \
        --data-dir "$DATA" \
        --output-dir "$OUT" \
        --num-inference-steps $NUM_INFER \
        --guidance-scale 5.0 \
        --num-shards $NSHARDS \
        --shard-idx $sid \
        --skip-existing \
        $EXTRA_ARGS \
        > "$LOGDIR/${MODE}_shard${sid}of${NSHARDS}.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

nnpz=$(ls "$OUT/$MODE/"*.npz 2>/dev/null | wc -l)
echo "[gen-node] done $(date) MODE=$MODE shards ${SHARD_START}..$((SHARD_START+NGPU-1))/$NSHARDS npz_total=$nnpz" | tee -a "$LOGDIR/nodes.log"
# Only write the completion marker if generation actually produced output.
# Guards against fast-crash (e.g. wrong python/env) falsely signalling "done".
if [ "$nnpz" -gt 0 ]; then
    touch "$LOGDIR/_done_${MODE}_s${SHARD_START}of${NSHARDS}"
else
    echo "[gen-node] WARNING no NPZ produced; marker NOT written (likely env/python error)" | tee -a "$LOGDIR/nodes.log"
fi
