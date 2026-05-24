#!/bin/bash
# Compute KAFS ablation metrics after generation is complete.
# Run on taiji machine with GPU (evaluator needs CUDA).
# Usage: bash scripts/eval/run_kafs_metrics.sh

set -e

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

KAFS_DIR="work_dirs/prism_kafs_ablation"

# Verify all modes have completed
for mode in none depth_driven uniform random; do
    manifest="$KAFS_DIR/$mode/manifest.json"
    if [ ! -f "$manifest" ]; then
        count=$(ls "$KAFS_DIR/$mode/"*.npz 2>/dev/null | wc -l)
        echo "[WARN] Mode $mode: manifest.json not found ($count NPZ files generated)"
        echo "       Generation may still be running!"
    else
        echo "[OK] Mode $mode: manifest.json found"
    fi
done

echo ""
echo "Starting metric computation..."

CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/compute_kafs_metrics.py \
    --kafs-dir "$KAFS_DIR" \
    --modes none depth_driven uniform random \
    --anno-file data/annotation/test_motionhub_t2m.json \
    --data-dir data/motionhub \
    --evaluator-ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
    --clip-pretrained checkpoints/clip-vit-base-patch32 \
    --stats-file data/statistic/smplx55_stats_hymotion_aug.json \
    --n-repeats 20 \
    --gpu 0

echo ""
echo "Done! Results in $KAFS_DIR/kafs_metrics_all.json"
