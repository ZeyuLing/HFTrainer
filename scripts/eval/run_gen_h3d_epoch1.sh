#!/bin/bash
# Generate HumanML3D T2M motions for the latest spectral checkpoint (epoch_1),
# sharded across 8 GPUs. Only the `none` KAFS mode is needed for HumanML3D:
# it provides the main-table H3D column and the KT-RoPE "Projected Spectral"
# H3D column (the KAFS ablation table itself is MotionHub-only).
#
# Intended to run as the start_cmd of a fresh Taiji 8xV100 node, or detached
# on any idle debug machine. Writes NPZ to shared CephFS.

set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py
CKPT=work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_1
ANNO=data/annotation/test_hml3d.json
DATA=data/motionhub
OUT=outputs/evaluation/prism_kt_spectral_epoch1/h3d
NSHARDS=8
LOGDIR=$OUT/_logs
mkdir -p "$LOGDIR"

echo "[h3d-launcher] start $(date) ckpt=$CKPT out=$OUT" | tee -a "$LOGDIR/launcher.log"

MODE=none
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
for p in "${pids[@]}"; do wait "$p"; done
nnpz=$(ls "$OUT/$MODE/"*.npz 2>/dev/null | wc -l)
echo "[h3d-launcher] MODE=$MODE done $(date) npz=$nnpz" | tee -a "$LOGDIR/launcher.log"
touch "$OUT/_GEN_COMPLETE"
echo "[h3d-launcher] ALL DONE $(date)" | tee -a "$LOGDIR/launcher.log"
