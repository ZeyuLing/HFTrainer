#!/bin/bash
# Generate KAFS ablation motions for the latest spectral checkpoint (epoch_1),
# sharded across 8 GPUs, one mode at a time (keeps all 8 GPUs saturated).
# Designed to run detached on a Taiji debug machine; logs to CephFS.
#
# Modes: none (baseline), depth_driven (ours), random (control).
# `uniform` is functionally identical to `none`, so we copy none/ -> uniform/
# after generation instead of regenerating it.

set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py
CKPT=work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_1
ANNO=data/annotation/test_motionhub_t2m.json
DATA=data/motionhub
OUT=outputs/evaluation/prism_kt_spectral_epoch1/kafs
NSHARDS=8
LOGDIR=$OUT/_logs
mkdir -p "$LOGDIR"

echo "[launcher] start $(date) ckpt=$CKPT out=$OUT" | tee -a "$LOGDIR/launcher.log"

for MODE in none depth_driven random; do
    echo "[launcher] === MODE=$MODE $(date) ===" | tee -a "$LOGDIR/launcher.log"
    pids=()
    for g in $(seq 0 $((NSHARDS-1))); do
        CUDA_VISIBLE_DEVICES=$g python3 scripts/eval/eval_prism_kafs_ablation.py \
            --config "$CONFIG" \
            --checkpoint "$CKPT" \
            --kafs-mode "$MODE" \
            --anno-file "$ANNO" \
            --data-dir "$DATA" \
            --output-dir "$OUT" \
            --num-inference-steps 50 \
            --guidance-scale 5.0 \
            --num-shards $NSHARDS \
            --shard-idx $g \
            --skip-existing \
            > "$LOGDIR/${MODE}_shard${g}.log" 2>&1 &
        pids+=($!)
    done
    # Block until all 8 shards for this mode finish
    for p in "${pids[@]}"; do wait "$p"; done
    nnpz=$(ls "$OUT/$MODE/"*.npz 2>/dev/null | wc -l)
    echo "[launcher] MODE=$MODE done $(date) npz=$nnpz" | tee -a "$LOGDIR/launcher.log"
done

# uniform == none (alpha=1.0 for all joints); reuse none outputs.
echo "[launcher] copying none/ -> uniform/ $(date)" | tee -a "$LOGDIR/launcher.log"
mkdir -p "$OUT/uniform"
cp -n "$OUT/none/"*.npz "$OUT/uniform/" 2>/dev/null
nnpz=$(ls "$OUT/uniform/"*.npz 2>/dev/null | wc -l)
echo "[launcher] uniform npz=$nnpz" | tee -a "$LOGDIR/launcher.log"

echo "[launcher] ALL DONE $(date)" | tee -a "$LOGDIR/launcher.log"
touch "$OUT/_GEN_COMPLETE"
