#!/usr/bin/env bash
# Diagnostic: does the inference-time KAFS depth_driven scaling cause PRISM's
# foot/distal jumps, or is the jitter intrinsic to the model? Generate the SAME
# e15 model on the SAME N samples with kafs=none vs kafs=depth_driven; jerk is
# then compared on the native output (relative comparison, no repack needed).
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"
CONFIG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py
CKPT=work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_15
OUT=outputs/evaluation/prism_e15_kafs_diag
mkdir -p "$OUT"
N=${N:-64}
gen() { # mode gpu
  CUDA_VISIBLE_DEVICES=$2 python3 scripts/eval/eval_prism_kafs_ablation.py \
    --config "$CONFIG" --checkpoint "$CKPT" --kafs-mode "$1" \
    --anno-file data/annotation/test_hml3d.json \
    --rewritten-caption-file data/annotation/test_hml3d_rewritten.json \
    --data-dir data/motionhub --output-dir "$OUT" \
    --num-inference-steps 50 --guidance-scale 5.0 --max-samples "$N" --seed 42 \
    > "$OUT/gen_$1.log" 2>&1
}
# Sequential (one T5 load at a time) avoids the concurrent-load deadlock seen
# when two processes read the kt-spectral T5 shards from CephFS simultaneously.
gen none 0
gen depth_driven 0
echo DONE > "$OUT/_done.txt"
