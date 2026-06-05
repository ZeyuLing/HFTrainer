#!/bin/bash
# Supplement the existing epoch-7 HumanML3D PRISM generation on spare GPUs.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:${PYTHONPATH:-}

CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py
CKPT=work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_7
OUT=outputs/evaluation/prism_kt_spectral_epoch7_rw/h3d
LOGDIR=$OUT/_logs
mkdir -p "$LOGDIR"

echo "[supplement] start $(date) ckpt=$CKPT" | tee -a "$LOGDIR/supplement_0604.log"
pids=()
for spec in 0:4 1:5 2:6 3:7; do
    gpu=${spec%%:*}
    sid=${spec##*:}
    (
        CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/eval_prism_kafs_ablation.py \
            --config "$CONFIG" \
            --checkpoint "$CKPT" \
            --kafs-mode depth_driven \
            --anno-file data/annotation/test_hml3d.json \
            --rewritten-caption-file data/annotation/test_hml3d_rewritten.json \
            --data-dir data/motionhub \
            --output-dir "$OUT" \
            --num-inference-steps 50 \
            --guidance-scale 5.0 \
            --num-shards 8 \
            --shard-idx "$sid" \
            --skip-existing \
            > "$LOGDIR/depth_driven_supp_shard${sid}of8.log" 2>&1
        echo "$?" > "$LOGDIR/depth_driven_supp_shard${sid}of8.exit"
    ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
n=$(find "$OUT/depth_driven" -maxdepth 1 -type f -name '*.npz' | wc -l)
echo "[supplement] done $(date) npz=$n" | tee -a "$LOGDIR/supplement_0604.log"
touch "$LOGDIR/_done_supplement_0604"
