#!/bin/bash
# Compute SMPL-22 MotionCLIP TMR metrics (FID / R-Precision / MM-Dist) for the
# epoch_1 spectral checkpoint: MotionHub KAFS ablation (4 modes) + HumanML3D none.
# Run on a GPU node. MH uses GPU 0, H3D uses GPU 1 (parallel).
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

ROOT=outputs/evaluation/prism_kt_spectral_epoch1
EV="--evaluator-ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq --clip-pretrained checkpoints/clip-vit-base-patch32 --stats-file data/statistic/smplx55_stats_hymotion_aug.json --n-repeats 20"

mkdir -p "$ROOT/kafs/_logs" "$ROOT/h3d/_logs"

# MotionHub KAFS ablation (4 modes) on GPU 0
# NPZ->position conversion already done (*_135d dirs exist) -> skip-convert.
python3 scripts/eval/compute_kafs_metrics.py \
    --kafs-dir "$ROOT/kafs" \
    --modes none depth_driven uniform random \
    --anno-file data/annotation/test_motionhub_t2m.json \
    --data-dir data/motionhub \
    --skip-convert \
    $EV --gpu 0 \
    > "$ROOT/kafs/_logs/metrics_mh.log" 2>&1 &
PID_MH=$!

# HumanML3D (spectral, none) on GPU 1
python3 scripts/eval/compute_kafs_metrics.py \
    --kafs-dir "$ROOT/h3d" \
    --modes none \
    --anno-file data/annotation/test_hml3d.json \
    --data-dir data/motionhub \
    $EV --gpu 1 \
    > "$ROOT/h3d/_logs/metrics_h3d.log" 2>&1 &
PID_H3D=$!

wait $PID_MH; echo "[metrics] MH done $(date)"
wait $PID_H3D; echo "[metrics] H3D done $(date)"
touch "$ROOT/_METRICS_COMPLETE"
echo "[metrics] ALL DONE $(date)"
