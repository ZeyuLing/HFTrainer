#!/bin/bash
# GT-only sanity check: reproduce the paper's "Real" row.
# Hypothesis: paper MH main table uses test_motionhub_1p.json (7569), not
# test_motionhub_t2m.json (1590). Run GT-only on the 1P set and compare to
# paper MH Real (R-P T1=0.667, T3=0.842, MM-D=0.984, Div=22.96).
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:${PYTHONPATH:-}
GPU=${GPU:-6}
OUT=outputs/evaluation/_gtcheck
mkdir -p "$OUT"

run_gt () {
    local tag=$1 anno=$2
    echo "[gtcheck] >>> $tag start $(date)" | tee -a "$OUT/run.log"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval/eval_with_motionclip_evaluator.py \
        --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
        --anno_file "$anno" \
        --data_dir data/motionhub \
        --gt_only \
        --out_json "$OUT/${tag}_gt.json" \
        --n_repeats 20 --seed 42 \
        >"$OUT/${tag}.log" 2>&1
    echo "[gtcheck] <<< $tag done rc=$? $(date)" | tee -a "$OUT/run.log"
}

run_gt mh_1p   data/annotation/test_motionhub_1p.json
run_gt mh_t2m  data/annotation/test_motionhub_t2m.json
run_gt h3d     data/annotation/test_hml3d.json

touch "$OUT/_DONE"
echo "[gtcheck] ALL DONE $(date)" | tee -a "$OUT/run.log"
