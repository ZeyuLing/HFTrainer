#!/bin/bash
# Score PRISM epoch_4 (rewritten-generated) depth_driven outputs under the PAPER
# protocol: rewritten captions + R-P pool 64. Apples-to-apples vs paper "ours".
# Paper "ours": H3D T1/T3=0.699/0.893 ; MH T1/T3=0.530/0.772.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer:${PYTHONPATH:-}
GPU=${GPU:-4}
OUT=outputs/evaluation/_pred_paperproto
mkdir -p "$OUT"

run_pred () {
    local tag=$1 anno=$2 rw=$3 preddir=$4
    echo "[predproto] >>> $tag start $(date)" | tee -a "$OUT/run.log"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval/eval_with_motionclip_evaluator.py \
        --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
        --anno_file "$anno" \
        --data_dir data/motionhub \
        --rewritten_caption_file "$rw" \
        --pred_dir "$preddir" \
        --chunk_size 64 \
        --out_json "$OUT/${tag}.json" \
        --n_repeats 20 --seed 42 \
        >"$OUT/${tag}.log" 2>&1
    echo "[predproto] <<< $tag done rc=$? $(date)" | tee -a "$OUT/run.log"
}

run_pred mh  data/annotation/test_motionhub_t2m.json data/annotation/test_motionhub_t2m_rewritten.json \
    outputs/evaluation/prism_kt_spectral_epoch4_rw/mh/depth_driven_135d
run_pred h3d data/annotation/test_hml3d.json         data/annotation/test_hml3d_rewritten.json \
    outputs/evaluation/prism_kt_spectral_epoch4_rw/h3d/depth_driven_135d

touch "$OUT/_DONE"
echo "[predproto] ALL DONE $(date)" | tee -a "$OUT/run.log"
