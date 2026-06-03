#!/bin/bash
# Reproduce the PAPER main-table "Real" row with the paper protocol:
#   rewritten captions (use_rewriter=True) + R-P pool size 64.
# Target: MH Real T1/T3 = 0.667/0.842 ; H3D Real T1/T3 = 0.778/0.906.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:${PYTHONPATH:-}
GPU=${GPU:-5}
OUT=outputs/evaluation/_gtcheck_paperproto
mkdir -p "$OUT"

run_gt () {
    local tag=$1 anno=$2 rw=$3
    echo "[paperproto] >>> $tag start $(date)" | tee -a "$OUT/run.log"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval/eval_with_motionclip_evaluator.py \
        --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
        --anno_file "$anno" \
        --data_dir data/motionhub \
        --rewritten_caption_file "$rw" \
        --chunk_size 64 \
        --gt_only \
        --out_json "$OUT/${tag}_gt.json" \
        --n_repeats 20 --seed 42 \
        >"$OUT/${tag}.log" 2>&1
    echo "[paperproto] <<< $tag done rc=$? $(date)" | tee -a "$OUT/run.log"
}

run_gt mh   data/annotation/test_motionhub_t2m.json data/annotation/test_motionhub_t2m_rewritten.json
run_gt h3d  data/annotation/test_hml3d.json         data/annotation/test_hml3d_rewritten.json

touch "$OUT/_DONE"
echo "[paperproto] ALL DONE $(date)" | tee -a "$OUT/run.log"
